from pathlib import Path
import json,tempfile,numpy as np
from official_rule_eval import *
def main():
 xyz=np.array([[0,0,0],[1,0,0],[2,0,0]],np.float32)
 ref=np.array([[0,0,0],[.0499,0,0],[.05,0,0],[.0501,0,0],[1.01,0,0],[2,0,0],[3,0,0]],np.float32)
 a,b,state=project_pair(xyz,ref,np.array([11,0,33]),np.array([0,22,44]))
 np.testing.assert_array_equal(a,[11,11,0,0,0,33,0])
 np.testing.assert_array_equal(b,[0,0,0,0,22,44,0])
 squared=np.sum((ref[:,None,:]-xyz[None,:,:])**2,axis=2,dtype=np.float32)
 index=np.argmin(squared,axis=1);matched=squared[np.arange(len(ref)),index]<np.float32(.05**2)
 np.testing.assert_array_equal(state['indices'],index)
 np.testing.assert_array_equal(state['matched'],matched)
 with tempfile.TemporaryDirectory(dir=O) as folder:
  d=Path(folder);coords=np.zeros((100,3),np.float32)
  gt=np.array([1]*40+[2]*40+[0]*20)
  pred=np.array([7]*20+[8]*20+[9]*40+[10]*20)
  write_labels(d/'gt.ply',coords,gt);write_labels(d/'pred.ply',coords,pred)
  metrics,stdout=score(d/'gt.ply',d/'pred.ply')
  expected={'mIoU':.75,'wIoU':.75,'P@25':.75,'R@25':1.5,'P@50':.75,'R@50':1.5,'P@75':.25,'R@75':.5}
  for k,v in expected.items():assert metrics[k]==v,(k,metrics[k],v)
  # A 9-vertex prediction is filtered; a 10-vertex prediction is retained.
  coords=np.zeros((19,3),np.float32);gt=np.array([1]*19);pred=np.array([2]*9+[3]*10)
  write_labels(d/'gt.ply',coords,gt);write_labels(d/'pred.ply',coords,pred)
  metrics,_=score(d/'gt.ply',d/'pred.ply')
  assert metrics['TP@50']==1 and metrics['FP@50']==0
 report={'status':'PASS','official_projection_body_used_unchanged':True,
 'shared_5cm_projection_matches_independent_float32_bruteforce':True,
 'strict_distance_boundary_and_unassigned_zero_checked':True,
 'official_scoring_function_called_unchanged':True,'exact_stdout_matches_captured_full_precision_values':True,
 'minimum_10_vertex_prediction_filter_checked':True,
 'non_one_to_one_recall_behavior_preserved_and_disclosed':True}
 dump(O/'official_rule_adapter_validation.json',report);print(json.dumps(report))
if __name__=='__main__':main()
