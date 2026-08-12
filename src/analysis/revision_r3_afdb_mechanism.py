"""Integrated, provenance-strict Revision R3 A-D mechanism analysis."""
from __future__ import annotations
import csv, json, math, os, tempfile, time
from copy import deepcopy
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.analysis.axis_distribution_shift import distribution_statistics, boundary_shift_statistics
from src.analysis.cross_dataset_direction_geometry import (GeometryAccumulator, centroid_distance_matrix, direction_cosine_matrix, prepare_selection)
from src.analysis.head_direction_equivalence import compare_directions, compute_ranking_metrics
from src.analysis.shared_axis_head_comparison import compare_head_to_directions
from src.data.ecg_dataset import ECGWindowDataset
from src.analysis.axis_distribution_shift import load_hidden_selected_rows
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.source_threshold_baseline import threshold_curve
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import environment_snapshot, git_identity, resolve_device, seed_everything, sha256_file

FORBIDDEN={"label","labels","binary_label","rhythm_label"}
WEIGHTINGS=("window_weighted","subject_equal")
def _load(p:Path)->dict: return json.loads(Path(p).read_text(encoding="utf-8"))
def _json(p:Path,x:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp"); q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); q.replace(p)
def _csv(p:Path, rows:list[dict])->None:
    if not rows: raise ValueError("cannot write empty CSV")
    q=p.with_suffix(p.suffix+".tmp")
    with q.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    q.replace(p)
def validate_config(c:dict)->None:
    if c.get("role")!="post_hoc_analysis" or c.get("target_label_usage")!="post-hoc mechanism analysis only": raise ValueError("R3 label-use contract mismatch")
    if sha256_file(Path(c["protocol"]))!=c.get("protocol_sha256"): raise ValueError("R3 protocol hash mismatch")
    if c.get("dataset_order")!=["afdb","cpsc2021","ltafdb","shdb-af"]: raise ValueError("R3 dataset order mismatch")
    if c["datasets"]["ltafdb"].get("dataset_version")!="ltaf_skip_first_hour_v1": raise ValueError("R3 requires LTAFDB-clean1h-v1")
    for n in c["dataset_order"]:
        if sha256_file(Path(c["datasets"][n]["index_path"]))!=c["datasets"][n]["index_sha256"]: raise ValueError(f"R3 index hash mismatch: {n}")
    for k in ("scores","thresholds"):
        if sha256_file(Path(c["r2_oof"][k]))!=c["r2_oof"][k+"_sha256"]: raise ValueError(f"R2 OOF {k} hash mismatch")
    if sha256_file(Path(c["model"]["checkpoint"]))!=c["model"]["checkpoint_sha256"]: raise ValueError("M_AFDB checkpoint hash mismatch")
def _root(c:dict,o:Path|None)->Path: return o or Path(c["output_dir"])
def prepare(c:dict,*,output_override:Path|None=None)->dict:
    validate_config(c); return prepare_selection(c,output_override=_root(c,output_override))
def _archive(p:Path,a:dict[str,np.ndarray])->None:
    if FORBIDDEN & set(a): raise ValueError("R3 frozen archive cannot contain labels")
    if len({len(v) for v in a.values()})!=1: raise ValueError("R3 archive arrays misaligned")
    q=p.with_suffix(p.suffix+".tmp")
    with q.open("wb") as h: np.savez_compressed(h,**a)
    q.replace(p)
def _validate_checkpoint(c:dict,device:torch.device):
    cfg=_load(Path(c["model"]["source_config"])); ck=torch.load(c["model"]["checkpoint"],map_location=device,weights_only=False); p=ck.get("provenance",{})
    if p.get("formal") is not True or p.get("seed")!=c["model"]["required_seed"] or p.get("main_seed") is not True or p.get("target_data_accessed") is not False: raise ValueError("M_AFDB provenance mismatch")
    return cfg,ck
def extract(c:dict,*,device_request:str="auto",output_override:Path|None=None,max_batches_per_dataset:int|None=None)->dict:
    c=deepcopy(c); validate_config(c); root=_root(c,output_override); sel=root/"selected_window_manifest.csv"; art=_load(root/"selection_artifact.json")
    if sha256_file(sel)!=art["selected_window_manifest_sha256"]: raise ValueError("R3 selection hash mismatch")
    if max_batches_per_dataset is not None and output_override is None: raise ValueError("diagnostic extraction requires output override")
    out=root/"extraction"; out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError("R3 extraction output is not empty")
    rows=load_hidden_selected_rows(sel); seed_everything(int(c["seed"])); device=resolve_device(device_request); cfg,ck=_validate_checkpoint(c,device)
    model=SourceMedTSTTT(**cfg["model"]).to(device); model.load_state_dict(ck["model_state"],strict=True); model.eval()
    parts={k:[] for k in ("dataset","subject_id","record_id","window_start","features","head_logit_difference")}; counts={}; started=time.perf_counter(); ex=c["extraction"]
    with torch.inference_mode():
      for name in c["dataset_order"]:
        selected=[r for r in rows if r.dataset==name]
        if max_batches_per_dataset is not None: selected=selected[:max_batches_per_dataset*int(ex["batch_size"])]
        loader=DataLoader(ECGWindowDataset(selected,data_root=Path(c["datasets"][name]["data_root"]),expose_label=False),batch_size=int(ex["batch_size"]),shuffle=False,num_workers=int(ex["num_workers"])); n=0; t=time.perf_counter()
        for bi,b in enumerate(loader,1):
            if not bool((b["y"]==-1).all()): raise ValueError("R3 extraction exposed labels")
            logits,raw=model(b["x"].to(device),return_features=True); feat=F.normalize(raw,dim=-1,eps=1e-12)
            if not torch.isfinite(feat).all() or not torch.isfinite(logits).all(): raise FloatingPointError("R3 extraction produced NaN/Inf")
            m=b["metadata"]; parts["dataset"].extend(map(str,m["dataset"])); parts["subject_id"].extend(map(str,m["subject_id"])); parts["record_id"].extend(map(str,m["record_id"])); parts["window_start"].extend(map(int,m["window_start"].tolist())); parts["features"].append(feat.cpu().numpy().astype(np.float32)); parts["head_logit_difference"].append((logits[:,1]-logits[:,0]).cpu().numpy().astype(np.float32)); n+=len(feat)
            if bi%int(ex["progress_every_batches"])==0: print(f"phase=r3_extract dataset={name} batch={bi} samples={n} seconds={time.perf_counter()-t:.1f}",flush=True)
        counts[name]=n
    arrays={"dataset":np.asarray(parts["dataset"]),"subject_id":np.asarray(parts["subject_id"]),"record_id":np.asarray(parts["record_id"]),"window_start":np.asarray(parts["window_start"],dtype=np.int64),"features":np.concatenate(parts["features"]),"head_logit_difference":np.concatenate(parts["head_logit_difference"])}
    path=out/"features_and_scores.npz"; _archive(path,arrays)
    artifact={"frozen":True,"labels_accessed":False,"selected_manifest_label_fields_parsed":False,"diagnostic_max_batches_per_dataset":max_batches_per_dataset,"counts":counts,"selection_sha256":sha256_file(sel),"checkpoint_sha256":sha256_file(Path(c["model"]["checkpoint"])),"archive_sha256":sha256_file(path)}; _json(out/"score_artifact.json",artifact)
    manifest={"git":git_identity(),"environment":environment_snapshot(device),"config":c,"labels_accessed":False,"score_artifact_sha256":sha256_file(out/"score_artifact.json"),"runtime_seconds":time.perf_counter()-started}; _json(out/"run_manifest.json",manifest); return {**artifact,"runtime_seconds":manifest["runtime_seconds"],"output_dir":str(out)}
def _label_map(p:Path)->dict:
    m={}
    with p.open(encoding="utf-8",newline="") as h:
      for x in csv.DictReader(h):
        k=(x["dataset"],x["subject_id"],x["record_id"],int(x["start_sample"])); y=int(x["binary_label"])
        if k in m or y not in (0,1): raise ValueError("invalid R3 label manifest")
        m[k]=y
    return m
def _join(a,m)->np.ndarray:
    seen=set(); out=[]
    for x in zip(a["dataset"],a["subject_id"],a["record_id"],a["window_start"]):
        k=(str(x[0]),str(x[1]),str(x[2]),int(x[3]))
        if k in seen or k not in m: raise ValueError("R3 identity join failure")
        seen.add(k); out.append(m[k])
    if len(seen)!=len(m): raise ValueError("R3 formal archive lacks full cohort")
    return np.asarray(out,dtype=np.int8)
def decision_gate(rows:list[dict],c:dict)->dict:
    rep=[]; boundary=[]; evidence={}; source=next(r for r in rows if r["dataset"]=="afdb"); rule=c["decision_gate"]
    for r in rows:
        if r["dataset"]=="afdb": continue
        signals={"auroc_below":r["auroc"]<float(rule["representation_auroc_below"]),"relative_auprc_drop":source["auprc"]-r["auprc"]>=float(rule["relative_auprc_drop_at_least"]),"gap_contracted":r["class_gap_ratio"]<float(rule["class_gap_ratio_below"]),"overlap_increased":r["histogram_overlap_coefficient"]-source["histogram_overlap_coefficient"]>=float(rule["overlap_increase_at_least"]),"material_boundary_headroom":r["boundary_headroom"]>=float(rule["boundary_headroom_at_least"])}
        representation=signals["auroc_below"] or ((signals["relative_auprc_drop"] or signals["gap_contracted"] or signals["overlap_increased"]) and not signals["material_boundary_headroom"])
        boundary_signal=(not signals["auroc_below"]) and signals["material_boundary_headroom"]
        if representation: rep.append(r["dataset"])
        if boundary_signal: boundary.append(r["dataset"])
        evidence[r["dataset"]]=signals
    case="C_mixed" if rep and boundary else ("A_representation_first" if rep else ("B_boundary_first" if boundary else "B_boundary_first_weak_headroom"))
    return {"case":case,"representation_limited_targets":rep,"boundary_limited_targets":boundary,"per_target_evidence":evidence,"rule":rule,"historical_stage6_order_inherited":False}
def _plot_heat(root:Path,names:list[str],mats:dict):
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"r3-mpl")); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
    for ax,(k,m) in zip(axs.flat,mats.items()):
        im=ax.imshow(m,cmap="coolwarm" if "cosine" in k else "viridis"); ax.set_xticks(range(4),names,rotation=30,ha="right"); ax.set_yticks(range(4),names); ax.set_title(k.replace("_"," "))
        for i in range(4):
          for j in range(4): ax.text(j,i,f"{m[i,j]:.3f}",ha="center",va="center",fontsize=8)
        fig.colorbar(im,ax=ax)
    fig.savefig(root/"r3b_geometry.png",dpi=180); plt.close(fig)
def analyze(c:dict,*,output_override:Path|None=None)->dict:
    c=deepcopy(c); validate_config(c); root=_root(c,output_override); out=root/"extraction"; artifact=_load(out/"score_artifact.json"); manifest=_load(out/"run_manifest.json"); archive_path=out/"features_and_scores.npz"
    if artifact.get("frozen") is not True or artifact.get("labels_accessed") is not False or artifact.get("diagnostic_max_batches_per_dataset") is not None: raise ValueError("R3 formal analysis rejects extraction")
    if sha256_file(archive_path)!=artifact["archive_sha256"] or sha256_file(out/"score_artifact.json")!=manifest["score_artifact_sha256"]: raise ValueError("R3 extraction artifact changed")
    a=np.load(archive_path); forbidden=FORBIDDEN&set(a.files)
    if forbidden: raise ValueError("R3 archive leaks labels")
    labels=_join(a,_label_map(root/"selected_window_manifest.csv")); names=c["dataset_order"]; features=a["features"].astype(np.float64); datasets=a["dataset"]; subjects=a["subject_id"]
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5): raise ValueError("R3 features invalid")
    summaries={}
    for name in names:
        mask=datasets==name; acc=GeometryAccumulator(features.shape[1]); acc.update(features[mask],labels[mask],subjects[mask]); summaries[name]=acc.finalize()
    dproto=np.asarray(summaries["afdb"]["window_weighted"]["direction"]); midpoint=float((np.asarray(summaries["afdb"]["window_weighted"]["af_prototype"])+np.asarray(summaries["afdb"]["window_weighted"]["nonaf_prototype"]))@dproto/2)
    device=torch.device("cpu"); cfg,ck=_validate_checkpoint(c,device); w=ck["model_state"]["backbone.classification_head.weight"].numpy(); b=ck["model_state"]["backbone.classification_head.bias"].numpy(); comp=compare_directions(dproto,w,b); proto=features@dproto-midpoint; head=a["head_logit_difference"].astype(np.float64)
    per={}
    for name in names:
        mask=datasets==name; pr=pearsonr(proto[mask],head[mask]).statistic; sp=spearmanr(proto[mask],head[mask]).statistic
        per[name]={"support":int(mask.sum()),"pearson":float(pr),"spearman":float(sp),"prototype":compute_ranking_metrics(labels[mask],proto[mask]),"head":compute_ranking_metrics(labels[mask],head[mask])}
    o=np.load(c["r2_oof"]["scores"]); oof={"support":len(o["labels"]),"pearson":float(pearsonr(o["prototype_margin"],o["head_logit_difference"]).statistic),"spearman":float(spearmanr(o["prototype_margin"],o["head_logit_difference"]).statistic),"prototype":compute_ranking_metrics(o["labels"],o["prototype_margin"]),"head":compute_ranking_metrics(o["labels"],o["head_logit_difference"]),"note":"fold-specific OOF models; unbiased AFDB source estimate"}
    eq={"conclusion":"highly_equivalent" if comp["cosine"]>c["equivalence_rule"]["min_direction_cosine"] and min(x["spearman"] for x in per.values())>c["equivalence_rule"]["min_dataset_spearman"] else "related_but_not_equivalent","rule":c["equivalence_rule"]}; r3a={"direction_comparison":comp,"cohort_metrics":per,"r2_oof_metrics":oof,"equivalence":eq}; _json(root/"r3a_prototype_vs_head.json",r3a)
    mats={}
    for weighting in WEIGHTINGS:
        dirs=np.asarray([summaries[n][weighting]["direction"] for n in names]); cents=np.asarray([summaries[n][weighting]["centroid"] for n in names]); mats[f"direction_cosine_{weighting}"]=direction_cosine_matrix(dirs); mats[f"centroid_distance_{weighting}"]=centroid_distance_matrix(cents)
        for kind in ("direction_cosine","centroid_distance"):
            m=mats[f"{kind}_{weighting}"]; _csv(root/f"r3b_{kind}_{weighting}.csv",[{"dataset":n,**{q:float(v) for q,v in zip(names,row)}} for n,row in zip(names,m)])
    r3b={"target_label_usage":c["target_label_usage"],"dataset_order":names,"datasets":summaries,"matrices":{k:v.tolist() for k,v in mats.items()}}; _json(root/"r3b_four_dataset_geometry.json",r3b); _plot_heat(root,names,mats)
    r3c={}
    for weighting in WEIGHTINGS:
        dirs=np.asarray([summaries[n][weighting]["direction"] for n in names]); r3c[weighting]=compare_head_to_directions(comp["head_direction"],dirs,source_index=0)
    _json(root/"r3c_head_vs_shared_axis.json",r3c)
    thresholds=_load(Path(c["r2_oof"]["thresholds"])); p0=0.0; p1=float(thresholds["prototype_margin"]["optimized"]["threshold"]); stats={}; oracle={}
    for name in names:
        mask=datasets==name; stats[name]=distribution_statistics(labels[mask],proto[mask],bins=int(c["statistics"]["histogram_bins"]),score_range=tuple(c["statistics"]["score_range"])); _,oracle[name]=threshold_curve(labels[mask],proto[mask],fixed_threshold=p1)
    rows=[]; operating={}
    for name in names:
        mask=datasets==name; p1m=compute_binary_metrics(labels[mask],proto[mask],threshold=p1); om=compute_binary_metrics(labels[mask],proto[mask],threshold=float(oracle[name]["threshold"])); shift=boundary_shift_statistics(stats[name],stats["afdb"],p0_threshold=p0,p1_threshold=p1,oracle_threshold=float(oracle[name]["threshold"])); row={"dataset":name,**{k:v for k,v in stats[name].items() if k not in ("histogram_bins","histogram_range")},**shift,"p0_threshold":p0,"p1_threshold":p1,"oracle_threshold":float(oracle[name]["threshold"]),"oracle_balanced_accuracy":om["balanced_accuracy"],"p1_balanced_accuracy":p1m["balanced_accuracy"],"boundary_headroom":om["balanced_accuracy"]-p1m["balanced_accuracy"]}; rows.append(row); operating[name]={"P0":compute_binary_metrics(labels[mask],proto[mask],threshold=p0),"P1":p1m,"oracle_post_hoc_only":om}
    gate=decision_gate(rows,c); r3d={"target_label_usage":c["target_label_usage"],"oracle_usage":"post-hoc mechanism only; prohibited for adaptation/model selection","statistics":rows,"operating_metrics":operating,"decision_gate":gate}; _json(root/"r3d_axis_distribution_shift.json",r3d); _csv(root/"r3d_distribution_statistics.csv",rows); _json(root/"decision_gate.json",gate)
    result={"experiment":c["experiment"],"r3a":r3a,"r3b_summary":{"matrices":r3b["matrices"]},"r3c":r3c,"r3d":r3d,"adaptation_time_target_labels_accessed":False,"post_freeze_analysis_labels_accessed":True}; _json(root/"analysis_result.json",result)
    outputs={p.name:sha256_file(p) for p in root.iterdir() if p.is_file()}; _json(root/"run_manifest.json",{"git":git_identity(),"config":c,"input_archive_sha256":artifact["archive_sha256"],"outputs":outputs}); return result
