"""Apply the exact archived DBSCAN numerics to isolated full-online outputs."""
from pathlib import Path
import os,sys,json,hashlib,time,importlib.util,fcntl,traceback
import numpy as np
C=Path('/home/chenkejun/CVPR/experiments/trusted_dbscan_20261105')
O=Path('/data/chenkejun/CVPR/results/trusted_dbscan_20261105')
BASE=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
TRUST=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
SC=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101')
DB=BASE/'ablations/dbscan_history_covisible_20261102'
H=SC/'ablations/dbscan_identity_completion_guarded_20261102/run_history_covisible.py'
G=H.with_name('run_guarded_covisible.py')
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
 q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main(mode,scene):
 assert mode in ['control','candidate'] and scene in ['room0','room1','room2','office0','office1','office2','office3','office4']
 dest=O/mode/scene;dest.mkdir(parents=True,exist_ok=True)
 with (dest/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (dest/'final_scene_complete.json').exists():
   assert read(dest/'final_scene_complete.json')['status']=='PASS';return
  started=time.time();root=BASE if mode=='control' else TRUST
  frozen=read(C/'source_lock.json')
  for p,h in frozen['sha256'].items(): assert sha(p)==h,p
  if mode=='candidate':
   controls=read(O/'control_complete.json');assert controls['status']=='PASS' and len(controls['scenes'])==8
  inputs={}
  m=read(root/'published/online_B'/scene/'scene_complete.json')
  for p in [root/'published/online_B'/scene/'scene_complete.json',root/'published/online_B'/scene/'final_B/instance_surface.npz',root/'online'/scene/'400/B/repair/commit_candidate/source_pixel_support.npz']:
   inputs[str(p)]=sha(p)
  for cp in [50,100,200,300,400]:
   b=root/'online'/scene/str(cp)/'B/repair'
   for p in [b/'global_association.json',b/'e1/region.npz',b/f'partial/whole_to_partial/{scene}/native/surface_evidence.npz',b/'commit_complete.json']:
    inputs[str(p)]=sha(p)
  for key,p in m['inputs'].items(): assert sha(p)==m['input_sha256'][key];inputs[p]=m['input_sha256'][key]
  dump(dest/'input_audit.json',dict(status='PASS',mode=mode,scene=scene,input_root=str(root),sha256=inputs,uses_own_lineage=True,only_shared_prefix50_for_candidate=True))
  dump(dest/'status.json',dict(status='RUNNING',scene=scene,stage='exact_original_guarded_pipeline',unix=time.time()))
  sys.path.insert(0,str(SC))
  spec=importlib.util.spec_from_file_location('same_dbscan_history',H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);g=h.g
  # Only dependency roots change. No numerical function or parameter is edited.
  g.O=root;g.T=dest/'base';h.O=root;h.g=g
  assert h.CFG==read(DB/'full8_plan.json')['settings']
  g.protocol()
  before_functions={name:sha(Path(fn.__code__.co_filename)) for name,fn in [('guarded_one',g.one),('build_candidates',g.build_candidates),('replay_votes',g.replay_votes),('fill_gray',g.fill_gray),('history_core',h.core_completion)]}
  g.one('online_B',scene)
  dump(dest/'status.json',dict(status='RUNNING',scene=scene,stage='exact_original_history_completion',unix=time.time()))
  h.core_completion('online_B',scene,g.T/'online_B'/scene,dest)
  actual=load(dest/'instance_surface.npz');original=load(root/'published/online_B'/scene/'final_B/instance_surface.npz')
  for k in ['xyz_m','rgb']:np.testing.assert_array_equal(actual[k],original[k])
  if mode=='control':
   expected=load(DB/'online_B'/scene/'instance_surface.npz')
   for k in ['xyz_m','rgb','instance_id']:np.testing.assert_array_equal(actual[k],expected[k])
  for p,hsh in inputs.items():assert sha(p)==hsh,p
  for p,hsh in frozen['sha256'].items():assert sha(p)==hsh,p
  report=dict(status='PASS',scene=scene,mode=mode,prediction=str(dest/'instance_surface.npz'),prediction_sha256=sha(dest/'instance_surface.npz'),
   control_exact=True,control_receipt=str(O/'control'/scene/'final_scene_complete.json'),input_root=str(root),
   numerical_functions=before_functions,settings=dict(merge=g.CFG,history=h.CFG),no_algorithm_changes=True,GT_used_for_decisions=False,
   same_geometry_RGB=True,input_hashes_unchanged=True,seconds=time.time()-started,unix=time.time())
  dump(dest/'final_scene_complete.json',report);dump(dest/'status.json',report)
  print('SCENE_PASS',mode,scene,round(report['seconds'],1),flush=True)
if __name__=='__main__':
 try:main(*sys.argv[1:])
 except Exception:
  dump(O/sys.argv[1]/sys.argv[2]/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise
