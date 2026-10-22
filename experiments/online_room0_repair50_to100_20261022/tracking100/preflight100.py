from common100 import *
import sitecustomize,ast
def main():
 started=time.perf_counter()
 assert sitecustomize.settings_path==str(TRACKROOT/'input_guard.json') and sitecustomize.current_cutoff==495
 data=read(TRACKROOT/'inputs_complete.json');assert len(data['raw_rgb_hashes'])==496
 for p,h in data['raw_rgb_hashes'].items():assert digest(p)==h
 model=Path('/home/chenkejun/CVPR/repos/sam2_1/sam2')
 for p,h in data['sam_source_files'].items():assert digest(model/p)==h
 for name in ['gate_v1.py','gate_policy.json','joint_batch_ops.py','objectwise_ops.py','run_case.py']:
  assert digest(C/name)==digest(BASEC/name),name
 a=(BASEC/'track_case.py').read_text().replace('track_config50.json','track_config100.json').replace('assert total == 246','assert total == 496')
 assert a==(C/'track_case.py').read_text()
 def fns(path):
  return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
 a,b=fns(BASEC/'run_batch.py'),fns(C/'run_batch.py')
 for name in ['inputs','freeze_global_association','metrics_for']:assert a[name]==b[name],name
 probes=[]
 os.environ['CVPR_GUARD_TEST']='1'
 raw=Path(next(iter(data['raw_rgb_hashes']))).resolve().parent
 for path in [raw/'frame000500.jpg',Path('/data/chenkejun/CVPR/results/full400_forbidden_probe/surface.ply'),raw.parent/'info_semantic.json']:
  try:
   with path.open('rb') as f:f.read(1)
  except PermissionError as e:probes.append(dict(path=str(path),blocked=True,reason=str(e)))
  else:raise AssertionError(('guard_failed',str(path)))
 os.environ.pop('CVPR_GUARD_TEST',None)
 dump(TRACKROOT/'preflight100.json',dict(status='PASS',seconds=time.perf_counter()-started,guard_active=True,raw_prefix=[0,495],raw_files=496,
  SAM_source_files_checked=len(data['sam_source_files']),same_tracking_function_except_config_name_and_upper_bound=True,
  original_gate_and_metrics_unchanged=True,gate50_equivalence=read(TRACKROOT/'gate50_equivalence.json'),
  current_identity_adapter_sha256=digest(C/'scene100.py'),probe_results=probes,GT_used=False))
 print('PREFLIGHT100_PASS',flush=True)
if __name__=='__main__':main()
