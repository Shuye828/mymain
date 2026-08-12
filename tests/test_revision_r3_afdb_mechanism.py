import numpy as np
import pytest
from src.analysis.revision_r3_afdb_mechanism import decision_gate, _archive, _join

def test_r3_archive_rejects_labels(tmp_path):
    with pytest.raises(ValueError,match="cannot contain labels"):
        _archive(tmp_path/"x.npz",{"labels":np.array([0])})

def test_r3_identity_join_requires_full_unique_coverage():
    a={"dataset":np.array(["afdb"]),"subject_id":np.array(["s"]),"record_id":np.array(["r"]),"window_start":np.array([0])}
    assert _join(a,{("afdb","s","r",0):1}).tolist()==[1]
    with pytest.raises(ValueError): _join(a,{("afdb","s","r",0):1,("afdb","x","r",0):0})

def test_r3_decision_gate_cases():
    c={"decision_gate":{"representation_auroc_below":.9,"relative_auprc_drop_at_least":.1,"class_gap_ratio_below":.5,"overlap_increase_at_least":.15,"boundary_headroom_at_least":.03}}
    source={"dataset":"afdb","auroc":.99,"auprc":.98,"class_gap_ratio":1,"histogram_overlap_coefficient":.1,"boundary_headroom":0}
    rep={"dataset":"cpsc2021","auroc":.85,"auprc":.7,"class_gap_ratio":.4,"histogram_overlap_coefficient":.4,"boundary_headroom":.01}
    bound={"dataset":"ltafdb","auroc":.96,"auprc":.93,"class_gap_ratio":.8,"histogram_overlap_coefficient":.2,"boundary_headroom":.08}
    assert decision_gate([source,rep],c)["case"]=="A_representation_first"
    assert decision_gate([source,bound],c)["case"]=="B_boundary_first"
    assert decision_gate([source,rep,bound],c)["case"]=="C_mixed"
    neutral={**bound,"dataset":"shdb-af","boundary_headroom":.001}
    assert decision_gate([source,neutral],c)["case"]=="D_no_major_bottleneck_benchmark_first"
