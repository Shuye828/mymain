import numpy as np
import pytest
from src.evaluation.axis_interpolation import (
    _assert_unique,
    _diagnostic_source_rows,
    _npz,
    interpolate_direction,
    m1_decision_status,
    select_alpha,
)
def test_interpolation_endpoints_and_norm():
 h=np.array([1.,0.]); p=np.array([0.,1.])
 assert np.allclose(interpolate_direction(h,p,0),h)
 assert np.allclose(interpolate_direction(h,p,1),p)
 assert np.isclose(np.linalg.norm(interpolate_direction(h,p,.5)),1)
def test_select_alpha_auroc_band_then_bacc_then_larger():
 c={"selection":{"numeric_tolerance":1e-12,"auroc_tolerance":.005}}
 rows=[{"alpha":0.,"auroc":.9,"balanced_accuracy":.8},{"alpha":.5,"auroc":.896,"balanced_accuracy":.82},{"alpha":1.,"auroc":.895,"balanced_accuracy":.82}]
 s,e=select_alpha(rows,c); assert s["alpha"]==1.; assert len(e)==3
def test_target_archive_rejects_labels(tmp_path):
 with pytest.raises(ValueError,match="cannot contain labels"): _npz(tmp_path/"x.npz",{"labels":np.array([1])})
def test_identity_audit_rejects_duplicates():
 with pytest.raises(ValueError,match="duplicate"):
  _assert_unique([("afdb","s","r",0),("afdb","s","r",0)],"test")
def test_target_archive_rejects_misalignment(tmp_path):
 with pytest.raises(ValueError,match="misaligned"):
  _npz(tmp_path/"x.npz",{"score":np.array([1]),"subject_id":np.array(["a","b"])})
def test_diagnostic_source_rows_cover_both_classes():
 from src.data.ecg_dataset import WindowRow
 rows=[WindowRow("afdb",str(i),"s",i,i+1,250.0,y,"x","train","adapt") for y in (0,1) for i in range(4)]
 selected=_diagnostic_source_rows(rows,4)
 assert len(selected)==4
 assert {r.binary_label for r in selected}=={0,1}
def test_m1_endpoint_is_not_reported_as_success():
 means={k:{"alpha0":0.5,"selected":0.5} for k in ("auroc","auprc","balanced_accuracy","macro_f1","mcc")}
 status,improved=m1_decision_status(0.0,means,[{"auroc":0.5,"auprc":0.5}],[{"auroc":0.5,"auprc":0.5}])
 assert status=="endpoint_no_axis_utilization"
 assert improved==0
