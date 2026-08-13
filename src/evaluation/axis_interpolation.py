"""Main M1 source-OOF-selected disease-axis interpolation."""
from __future__ import annotations
import csv,json,time
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.data.ecg_dataset import ECGWindowDataset,WindowRow,load_unlabeled_target_rows,load_window_rows
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.source_threshold_baseline import threshold_curve
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.afdb_source_protocol import (
    centered_prototype_margin,
    fold_subject_partitions,
    read_fold_assignments,
    validate_fold_assignments,
)
from src.training.reproducibility import environment_snapshot,git_identity,resolve_device,seed_everything,sha256_file

FORBIDDEN={"label","labels","binary_label","rhythm_label"}
def _load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def _json(p,x):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp"); q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); q.replace(p)
def _npz(p,arrays,*,allow_labels=False):
 if not allow_labels and FORBIDDEN&set(arrays): raise ValueError("frozen target archive cannot contain labels")
 if len({len(v) for v in arrays.values()})!=1: raise ValueError("archive arrays are misaligned")
 p=Path(p); q=p.with_suffix(p.suffix+".tmp");
 with q.open("wb") as h: np.savez_compressed(h,**arrays)
 q.replace(p)
def _csv(p,rows):
 if not rows: raise ValueError("cannot write empty CSV")
 p=Path(p); q=p.with_suffix(p.suffix+".tmp");
 with q.open("w",encoding="utf-8",newline="") as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 q.replace(p)
def _root(c,o): return Path(o or c["output_dir"])
def _alpha_key(a): return f"score_alpha_{a:.2f}".replace(".","p")
def _unit(x):
 x=np.asarray(x,dtype=np.float64); n=float(np.linalg.norm(x))
 if x.ndim!=1 or not np.isfinite(x).all() or n<=1e-12: raise ValueError("direction must be finite nonzero vector")
 return x/n
def interpolate_direction(head,proto,alpha): return _unit((1-float(alpha))*_unit(head)+float(alpha)*_unit(proto))
def validate_config(c):
 if c.get("role")!="source_oof_selected_axis_interpolation" or sha256_file(Path(c["protocol"]))!=c.get("protocol_sha256"): raise ValueError("M1 protocol mismatch")
 for obj,key in ((c["r2"],"config"),(c["r2"],"fold_manifest"),(c["r2"],"fold_summary"),(c["r2"],"oof_archive")):
  if sha256_file(Path(obj[key]))!=obj[key+"_sha256"]: raise ValueError(f"M1 frozen input changed: {key}")
 for e in c["r2"]["fold_checkpoints"]:
  if sha256_file(Path(e["path"]))!=e["sha256"]: raise ValueError(f"fold checkpoint changed: {e['fold']}")
 if sha256_file(Path(c["r2"]["final_checkpoint"]))!=c["r2"]["final_checkpoint_sha256"]: raise ValueError("final checkpoint changed")
 if sha256_file(Path(c["source"]["index_path"]))!=c["source"]["index_sha256"]: raise ValueError("source index changed")
 for name,e in c["targets"].items():
  if sha256_file(Path(e["index_path"]))!=e["index_sha256"]: raise ValueError(f"target index changed: {name}")
 if c["targets"]["ltaf_clean1h"].get("dataset_version")!="ltaf_skip_first_hour_v1": raise ValueError("M1 requires clean1h LTAFDB")
 if list(map(float,c["alphas"])) != [0.0,0.25,0.5,0.75,1.0]: raise ValueError("M1 alpha grid changed")
def _model(c,ckpt_path,device):
 cfg=_load(c["r2"]["config"]); ck=torch.load(ckpt_path,map_location=device,weights_only=False); m=SourceMedTSTTT(**cfg["model"]).to(device); m.load_state_dict(ck["model_state"],strict=True); m.eval(); return m,ck,cfg
def _head_direction(model): return _unit((model.backbone.classification_head.weight[1]-model.backbone.classification_head.weight[0]).detach().cpu().numpy())
def _identities(rows): return [(r.dataset,r.subject_id,r.record_id,r.start_sample) for r in rows]
def _assert_unique(ids,what):
 if len(ids)!=len(set(ids)): raise ValueError(f"{what} contains duplicate identities")
def _diagnostic_source_rows(rows,limit):
 """Select a deterministic two-class source subset for diagnostic extraction."""
 if limit is None: return rows
 per_class=max(1,int(limit)//2); selected=[]
 for label in (0,1): selected.extend([r for r in rows if r.binary_label==label][:per_class])
 selected.sort(key=lambda r:(r.dataset,r.subject_id,r.record_id,r.start_sample))
 if {r.binary_label for r in selected}!={0,1}: raise ValueError("diagnostic source subset lacks both classes")
 return selected
def _extract_features(model,rows,data_root,device,ex,max_batches=None,phase="extract"):
 if max_batches is not None: rows=rows[:max_batches*int(ex["batch_size"])]
 loader=DataLoader(ECGWindowDataset(rows,data_root=Path(data_root),expose_label=False),batch_size=int(ex["batch_size"]),shuffle=False,num_workers=int(ex["num_workers"])); chunks=[]; ids=[]; started=time.perf_counter()
 with torch.inference_mode():
  for bi,b in enumerate(loader,1):
   if not bool((b["y"]==-1).all()): raise ValueError("feature extraction exposed labels")
   z=F.normalize(model.forward_features(b["x"].to(device)),dim=-1,eps=1e-12)
   if not torch.isfinite(z).all(): raise FloatingPointError("nonfinite normalized features")
   chunks.append(z.cpu().numpy().astype(np.float32)); m=b["metadata"]; ids.extend(zip(map(str,m["dataset"]),map(str,m["subject_id"]),map(str,m["record_id"]),map(int,m["window_start"].tolist())))
   if bi%int(ex["progress_every_batches"])==0: print(f"phase={phase} batch={bi} samples={len(ids)} seconds={time.perf_counter()-started:.1f}",flush=True)
 return np.concatenate(chunks),list(ids),rows
def extract_oof(c,*,device_request="auto",output_override=None,max_batches_per_fold=None):
 validate_config(c); diagnostic=max_batches_per_fold is not None
 if diagnostic and output_override is None: raise ValueError("diagnostic OOF requires output override")
 root=_root(c,output_override); out=root/"oof"; out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()): raise FileExistsError("M1 OOF output is not empty")
 seed_everything(int(c["seed"])); device=resolve_device(device_request); cfg=_load(c["r2"]["config"]); assignments=read_fold_assignments(Path(c["r2"]["fold_manifest"])); all_rows=load_window_rows([Path(c["source"]["index_path"])]); validate_fold_assignments(assignments,expected_subjects={r.subject_id for r in all_rows}); summaries={int(x["fold_id"]):x for x in _load(c["r2"]["fold_summary"])["folds"]}; parts={k:[] for k in ("labels","fold_id","features","dataset","subject_id","record_id","window_start")}; fold_dirs=[]; started=time.perf_counter()
 if len(all_rows)!=int(c["source"]["expected_windows"]): raise ValueError("AFDB source window count changed")
 for e in c["r2"]["fold_checkpoints"]:
  k=int(e["fold"]); train_subjects,val_subjects=fold_subject_partitions(assignments,k); rows=load_window_rows([Path(c["source"]["index_path"])],include_subjects=val_subjects); model,ck,_=_model(c,e["path"],device); prov=ck.get("provenance",{}); pcfg=prov.get("config",{})
  if pcfg.get("r2_fold_id")!=k or prov.get("index_sha256")!=c["source"]["index_sha256"] or pcfg.get("fold_manifest_sha256")!=c["r2"]["fold_manifest_sha256"]: raise ValueError("fold provenance mismatch")
  if set(pcfg.get("subject_partitions",{}).get("train",[]))!=train_subjects or set(pcfg.get("subject_partitions",{}).get("validation",[]))!=val_subjects: raise ValueError("fold subject provenance mismatch")
  summary=summaries[k]
  if summary["checkpoint_sha256"]!=e["sha256"] or int(summary["oof_windows"])!=len(rows): raise ValueError("fold summary mismatch")
  train_rows=load_window_rows([Path(c["source"]["index_path"])],include_subjects=train_subjects,max_windows_per_subject_per_class=int(cfg["training"]["max_windows_per_subject_per_class"]),seed=int(c["seed"])); train_used_rows=_diagnostic_source_rows(train_rows,None if not diagnostic else max_batches_per_fold*int(c["extraction"]["batch_size"])); train_z,_,train_used=_extract_features(model,train_used_rows,c["source"]["data_root"],device,c["extraction"],None,phase=f"m1_proto_verify_fold_{k}"); train_y=np.asarray([r.binary_label for r in train_used],dtype=np.int8); recomputed_proto,_,_=centered_prototype_margin(train_z,train_y); frozen_proto=_unit(summary["direction"])
  if not diagnostic and (int(summary["train_windows_for_prototype"])!=len(train_rows) or not np.allclose(recomputed_proto,frozen_proto,atol=2e-6,rtol=0)): raise ValueError("frozen fold prototype verification failed")
  validation_used_rows=_diagnostic_source_rows(rows,None if not diagnostic else max_batches_per_fold*int(c["extraction"]["batch_size"])); z,ids,used=_extract_features(model,validation_used_rows,c["source"]["data_root"],device,c["extraction"],None,phase=f"m1_oof_fold_{k}"); proto=frozen_proto; head=_head_direction(model); labels=np.asarray([r.binary_label for r in used],dtype=np.int8)
  if any(x[1] in train_subjects for x in ids): raise ValueError("OOF subject leakage")
  parts["labels"].append(labels); parts["fold_id"].append(np.full(len(used),k,dtype=np.int8)); parts["features"].append(z)
  for j,name in enumerate(("dataset","subject_id","record_id","window_start")): parts[name].extend(x[j] for x in ids)
  fold_dirs.append({"fold":k,"checkpoint_sha256":e["sha256"],"head_direction":head.tolist(),"prototype_direction":proto.tolist(),"recomputed_prototype_max_abs_error":float(np.max(np.abs(recomputed_proto-frozen_proto))),"head_prototype_cosine":float(head@proto),"prototype_verification_windows":len(train_used),"validation_windows":len(used),"train_subjects":sorted(train_subjects),"validation_subjects":sorted(val_subjects)})
 features=np.concatenate(parts["features"]); labels=np.concatenate(parts["labels"]); folds=np.concatenate(parts["fold_id"]); arrays={"labels":labels,"fold_id":folds,"dataset":np.asarray(parts["dataset"]),"subject_id":np.asarray(parts["subject_id"]),"record_id":np.asarray(parts["record_id"]),"window_start":np.asarray(parts["window_start"],dtype=np.int64),"features":features}
 for alpha in c["alphas"]:
  score=np.empty(len(labels),dtype=np.float32)
  for k,d in enumerate(fold_dirs): score[folds==k]=features[folds==k]@interpolate_direction(d["head_direction"],d["prototype_direction"],alpha).astype(np.float32)
  arrays[_alpha_key(alpha)]=score
 ids=list(zip(map(str,arrays["dataset"]),map(str,arrays["subject_id"]),map(str,arrays["record_id"]),map(int,arrays["window_start"]))); _assert_unique(ids,"M1 OOF")
 if not np.isfinite(features).all() or any(not np.isfinite(arrays[_alpha_key(a)]).all() for a in c["alphas"]): raise FloatingPointError("M1 OOF archive contains NaN/Inf")
 if not diagnostic:
  expected=_identities(all_rows); _assert_unique(expected,"AFDB source index")
  if len(labels)!=int(c["source"]["expected_windows"]) or set(ids)!=set(expected): raise ValueError("M1 OOF exact coverage mismatch")
  expected_labels={i:r.binary_label for i,r in zip(expected,all_rows)}
  if any(expected_labels[i]!=int(y) for i,y in zip(ids,labels)): raise ValueError("M1 OOF label identity mismatch")
 p=out/"normalized_oof_features_and_scores.npz"; _npz(p,arrays,allow_labels=True); _json(out/"fold_directions.json",{"folds":fold_dirs}); manifest={"formal":not diagnostic,"target_data_accessed":False,"source_labels_used":True,"windows":len(labels),"fold_counts":dict(Counter(map(int,folds))),"archive_sha256":sha256_file(p),"git":git_identity(),"environment":environment_snapshot(device),"runtime_seconds":time.perf_counter()-started}; _json(out/"run_manifest.json",manifest); return manifest
def select_alpha(metrics,c):
 tol=float(c["selection"]["numeric_tolerance"]); best=max(x["auroc"] for x in metrics); eligible=[x for x in metrics if x["auroc"]+tol>=best-float(c["selection"]["auroc_tolerance"])]; best_b=max(x["balanced_accuracy"] for x in eligible); tied=[x for x in eligible if np.isclose(x["balanced_accuracy"],best_b,atol=tol,rtol=0)]; return max(tied,key=lambda x:x["alpha"]),eligible
def select_source(c,*,output_override=None):
 validate_config(c); root=_root(c,output_override); out=root/"oof"; man=_load(out/"run_manifest.json"); p=out/"normalized_oof_features_and_scores.npz"
 if (not man.get("formal") and output_override is None) or man.get("target_data_accessed") is not False or sha256_file(p)!=man["archive_sha256"]: raise ValueError("M1 OOF archive is not valid source-only input")
 a=np.load(p); y=a["labels"]; rows=[]; curves=[]
 for alpha in c["alphas"]:
  s=a[_alpha_key(alpha)].astype(np.float64); curve,t=threshold_curve(y,s,fixed_threshold=0.0); m=compute_binary_metrics(y,s,threshold=float(t["threshold"])); row={"alpha":float(alpha),"auroc":float(roc_auc_score(y,s)),"auprc":float(average_precision_score(y,s)),"threshold":float(t["threshold"]),**{k:m[k] for k in ("balanced_accuracy","macro_f1","mcc","sensitivity","specificity","precision","accuracy")}}; rows.append(row); curves.extend({"alpha":float(alpha),**x} for x in curve)
 selected,eligible=select_alpha(rows,c); _csv(out/"alpha_metrics.csv",rows); _csv(out/"threshold_curves.csv",curves); artifact={"frozen":True,"formal":bool(man["formal"]),"target_data_accessed":False,"source_labels_used":True,"selection_rule":"within_0.005_best_auroc_then_max_bacc_then_larger_alpha","alphas":rows,"eligible_alphas":[x["alpha"] for x in eligible],"selected_alpha":selected["alpha"],"selected_threshold":selected["threshold"],"oof_archive_sha256":sha256_file(p),"protocol_sha256":c["protocol_sha256"]}; _json(out/"selection_artifact.json",artifact); _json(out/"selection_manifest.json",{"git":git_identity(),"formal":bool(man["formal"]),"selection_artifact_sha256":sha256_file(out/"selection_artifact.json"),"target_data_accessed":False}); return artifact
def build_final_axis(c,*,device_request="auto",output_override=None,max_batches=None):
 validate_config(c); diagnostic=max_batches is not None
 if diagnostic and output_override is None: raise ValueError("diagnostic final axis requires output override")
 root=_root(c,output_override); sel=_load(root/"oof/selection_artifact.json"); sm=_load(root/"oof/selection_manifest.json")
 if sha256_file(root/"oof/selection_artifact.json")!=sm["selection_artifact_sha256"] or sel.get("target_data_accessed") is not False or (not diagnostic and not sel.get("formal")): raise ValueError("selection artifact invalid")
 out=root/"final_axis"; out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()): raise FileExistsError("final axis output is not empty")
 started=time.perf_counter(); seed_everything(int(c["seed"])); device=resolve_device(device_request); model,ck,cfg=_model(c,c["r2"]["final_checkpoint"],device); rows=load_window_rows([Path(c["source"]["index_path"])],max_windows_per_subject_per_class=int(cfg["training"]["max_windows_per_subject_per_class"]),seed=int(c["seed"])); rows=_diagnostic_source_rows(rows,None if not diagnostic else max_batches*int(c["extraction"]["batch_size"])); z,ids,used=_extract_features(model,rows,c["source"]["data_root"],device,c["extraction"],None,phase="m1_final_source"); y=np.asarray([r.binary_label for r in used],dtype=np.int8); proto=_unit(z[y==1].mean(0)-z[y==0].mean(0)); head=_head_direction(model); directions={str(a):interpolate_direction(head,proto,a).tolist() for a in c["alphas"]}; source_path=out/"source_features.npz"; _npz(source_path,{"labels":y,"dataset":np.asarray([x[0] for x in ids]),"subject_id":np.asarray([x[1] for x in ids]),"record_id":np.asarray([x[2] for x in ids]),"window_start":np.asarray([x[3] for x in ids],dtype=np.int64),"features":z},allow_labels=True); artifact={"frozen":True,"formal":not diagnostic,"target_data_accessed":False,"source_labels_used":True,"selected_alpha":sel["selected_alpha"],"selected_threshold":sel["selected_threshold"],"head_direction":head.tolist(),"prototype_direction":proto.tolist(),"head_prototype_cosine":float(head@proto),"directions":directions,"source_windows":len(used),"source_features_sha256":sha256_file(source_path),"checkpoint_sha256":c["r2"]["final_checkpoint_sha256"],"selection_artifact_sha256":sha256_file(root/"oof/selection_artifact.json")}; _json(out/"final_directions.json",artifact); _json(out/"run_manifest.json",{"git":git_identity(),"environment":environment_snapshot(device),"formal":not diagnostic,"artifact_sha256":sha256_file(out/"final_directions.json"),"runtime_seconds":time.perf_counter()-started}); return artifact
def extract_target(c,*,target,device_request="auto",output_override=None,max_batches=None):
 validate_config(c); diagnostic=max_batches is not None
 if target not in c["targets"]: raise ValueError("unknown M1 target")
 if diagnostic and output_override is None: raise ValueError("diagnostic target extraction requires output override")
 root=_root(c,output_override); fa=root/"final_axis/final_directions.json"; fm=_load(root/"final_axis/run_manifest.json"); directions=_load(fa)
 if sha256_file(fa)!=fm["artifact_sha256"] or not directions.get("frozen") or (not diagnostic and not directions.get("formal")): raise ValueError("final axis artifact invalid")
 out=root/"targets"/target; out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()): raise FileExistsError("target output is not empty")
 entry=c["targets"][target]; rows=load_unlabeled_target_rows([Path(entry["index_path"])],target_split="evaluation")
 if not diagnostic and len(rows)!=entry["expected_evaluation_windows"]: raise ValueError("target evaluation count mismatch")
 seed_everything(int(c["seed"])); device=resolve_device(device_request); model,_,_=_model(c,c["r2"]["final_checkpoint"],device); z,ids,used=_extract_features(model,rows,entry["data_root"],device,c["extraction"],max_batches,phase=f"m1_target_{target}"); arrays={"dataset":np.asarray([x[0] for x in ids]),"subject_id":np.asarray([x[1] for x in ids]),"record_id":np.asarray([x[2] for x in ids]),"window_start":np.asarray([x[3] for x in ids],dtype=np.int64)}
 for alpha in c["alphas"]: arrays[_alpha_key(alpha)]=z@np.asarray(directions["directions"][str(alpha)],dtype=np.float32)
 if not np.isfinite(np.column_stack([arrays[_alpha_key(a)] for a in c["alphas"]])).all(): raise FloatingPointError("target scores contain NaN/Inf")
 ids2=list(zip(map(str,arrays["dataset"]),map(str,arrays["subject_id"]),map(str,arrays["record_id"]),map(int,arrays["window_start"]))); _assert_unique(ids2,f"M1 target {target}")
 p=out/"scores.npz"; _npz(p,arrays); artifact={"frozen":True,"formal":not diagnostic,"target":target,"target_labels_accessed":False,"diagnostic_max_batches":max_batches,"windows":len(used),"target_index_sha256":entry["index_sha256"],"final_direction_sha256":sha256_file(fa),"score_sha256":sha256_file(p)}; _json(out/"score_artifact.json",artifact); _json(out/"extraction_manifest.json",{"git":git_identity(),"environment":environment_snapshot(device),"formal":not diagnostic,"target_labels_accessed":False,"score_artifact_sha256":sha256_file(out/"score_artifact.json")}); return artifact
def _label_map(path):
 rows=load_window_rows([Path(path)],target_split="evaluation"); return {(r.dataset,r.subject_id,r.record_id,r.start_sample):r.binary_label for r in rows}
def m1_decision_status(selected_alpha,means,selected,base):
 """Apply the frozen M1 decision rule without calling an endpoint a success."""
 if np.isclose(float(selected_alpha),0.0,atol=1e-12,rtol=0): return "endpoint_no_axis_utilization",0
 improved_targets=sum((s["auroc"]>=b["auroc"] and s["auprc"]>=b["auprc"] and (s["auroc"]>b["auroc"] or s["auprc"]>b["auprc"])) for s,b in zip(selected,base)); ranking=means["auroc"]["selected"]>means["auroc"]["alpha0"] and means["auprc"]["selected"]>means["auprc"]["alpha0"]; operating=sum(means[k]["selected"]>means[k]["alpha0"] for k in ("balanced_accuracy","macro_f1","mcc")); status="strong_success" if ranking and operating>=2 and improved_targets>=2 else ("partial_success" if ranking else ("failure" if means["auroc"]["selected"]<means["auroc"]["alpha0"] and means["auprc"]["selected"]<means["auprc"]["alpha0"] and operating==0 else "mixed_or_neutral")); return status,int(improved_targets)
def evaluate_targets(c,*,output_override=None):
 validate_config(c); root=_root(c,output_override); selection=_load(root/"oof/selection_artifact.json"); all_rows=[]; per={}
 for target,entry in c["targets"].items():
  out=root/"targets"/target; art=_load(out/"score_artifact.json"); em=_load(out/"extraction_manifest.json"); p=out/"scores.npz"
  if not art.get("formal") or art.get("target_labels_accessed") is not False or sha256_file(p)!=art["score_sha256"] or sha256_file(out/"score_artifact.json")!=em["score_artifact_sha256"]: raise ValueError("target score freeze invalid")
  a=np.load(p); forbidden=FORBIDDEN&set(a.files)
  if forbidden: raise ValueError("target score archive contains labels")
  mapping=_label_map(entry["index_path"]); ids=list(zip(map(str,a["dataset"]),map(str,a["subject_id"]),map(str,a["record_id"]),map(int,a["window_start"]))); 
  if len(ids)!=len(set(ids)) or set(ids)!=set(mapping): raise ValueError("target frozen score coverage mismatch")
  y=np.asarray([mapping[x] for x in ids]); rows=[]
  for x in selection["alphas"]:
   alpha=x["alpha"]; s=a[_alpha_key(alpha)].astype(np.float64); m=compute_binary_metrics(y,s,threshold=float(x["threshold"])); row={"target":target,"alpha":alpha,"selected":alpha==selection["selected_alpha"],**m}; rows.append(row); all_rows.append(row)
  per[target]=rows; _json(out/"metrics.json",{"post_freeze_target_labels_accessed":True,"metrics":rows}); _json(out/"evaluation_manifest.json",{"analysis_git":git_identity(),"score_sha256":art["score_sha256"],"metrics_sha256":sha256_file(out/"metrics.json")})
 selected=[x for x in all_rows if x["selected"]]; base=[x for x in all_rows if x["alpha"]==0.0]; keys=("auroc","auprc","balanced_accuracy","macro_f1","mcc"); means={k:{"alpha0":float(np.mean([x[k] for x in base])),"selected":float(np.mean([x[k] for x in selected]))} for k in keys}; status,improved_targets=m1_decision_status(selection["selected_alpha"],means,selected,base); result={"experiment":c["experiment"],"selected_alpha":selection["selected_alpha"],"source_oof_threshold":selection["selected_threshold"],"target_specific_tuning":False,"target_labels_used_only_after_score_freeze":True,"per_target":per,"three_target_means":means,"targets_with_noninferior_and_improved_ranking":int(improved_targets),"m1_status":status,"status_interpretation":"Source-only selection retained the normalized head endpoint; M1 provides no primary disease-axis-utilization contrast." if status=="endpoint_no_axis_utilization" else "Selected nonzero interpolation is compared with the alpha=0 endpoint under the frozen M1 rule."}; _json(root/"analysis_result.json",result); _csv(root/"target_metrics.csv",all_rows); _json(root/"run_manifest.json",{"git":git_identity(),"analysis_result_sha256":sha256_file(root/"analysis_result.json"),"target_metrics_sha256":sha256_file(root/"target_metrics.csv")}); return result
