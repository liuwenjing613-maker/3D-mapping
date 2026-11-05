import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/unified_full_depth_20261101')
import write_fence
"""Sequential causal checkpoints within one branch; safe resume at completed stages."""
from pathlib import Path
import os,sys,json,subprocess,time,traceback
C=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/online')/C.name
GROUP=sys.argv[1];assert GROUP in ['A','B']
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def run(n,stage,script,env):
 from source_guard import check
 check(C)
 root=OUT/str(n)/GROUP;root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 with (root/(stage+'.log')).open('a') as log:
  p=subprocess.Popen([PY,'-B','-u',str(C/script)],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  write(OUT/f'status_{GROUP}.json',dict(status='RUNNING',scene=C.name,group=GROUP,checkpoint=n,stage=stage,pid=p.pid,started_unix=time.time()))
  rc=p.wait()
 row=dict(checkpoint=n,stage=stage,seconds=time.perf_counter()-start,returncode=rc)
 with (OUT/f'timings_{GROUP}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 assert rc==0,row
def main():
 for n,old in [(50,0),(100,50),(200,100),(300,200),(400,300)]:
  root=OUT/str(n)/GROUP
  ready=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')/'attributes'/C.name/str(n)/'complete.json'
  while not ready.exists():
   for name in ['native_failure.json','attributes_failure.json']:
    if (Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')/name).exists():raise RuntimeError(name)
   time.sleep(5)
  env={**os.environ,'CVPR_CHECKPOINT':str(n),'CVPR_TRACK_GROUP':GROUP,'CVPR_MAPPING_RUN_ROOT':str(root),
   'CVPR_SURFACE_POLICY':'checkpoint','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CVPR_GPU_POOL':os.environ.get('CVPR_GPU_POOL','0,1,2,3,4,5,6,7'),'CVPR_SCENE':'room1'}
  env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
  if n==50 and not (OUT/'initial'/GROUP/'initial_complete.json').exists():run(n,'initialize','initialize_empty.py',env)
  if not (root/'mapping/mapping_complete.json').exists():
   if not (root/'mapping/mapping_plan.json').exists():run(n,'mapping_setup','map_support.py',env)
   guarded={**env,'CVPR_ONLINE_GUARD':str(root/'mapping/input_guard.json'),'PYTHONPATH':str(C/'map_guard')+':'+str(C)}
   run(n,'mapping','map_continue.py',guarded)
  m=read(root/'mapping/mapping_complete.json');assert m['status']=='PASS' and m['frame_ids']==list(range(0,n*5,5)) and m['surface_rebuild_frames']==[n]
  first=read(root/f'mapping/frame{old+1}_state_audit.json');assert first['status']=='PASS'
  assert first['repaired_state_used']==(old>0)
  if old==0 and not (root/'mapping/independent_initial_audit.json').exists():
   subprocess.run([PY,'-B','/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/verify_initial_map.py','online',C.name,str(n),GROUP],cwd=C,check=True)
  if not (root/'selection_tracking_status.json').exists() or read(root/'selection_tracking_status.json').get('status')!='PASS':run(n,'selection_tracking','selection_run.py',env)
  assert read(root/'selection_tracking_status.json')['status']=='PASS'
  guarded={**env,'CVPR_ONLINE_GUARD':str(root/'input_guard.json'),'PYTHONPATH':str(C/'selection_guard')+':'+str(C)}
  for name,script,marker in [('repair','repair_continue.py','repair/repair_complete.json'),('commit','commit_continue.py','repair/commit_complete.json')]:
   if not (root/marker).exists():run(n,name,script,guarded)
   assert read(root/marker)['status']=='PASS'
  done=read(root/'repair/commit_complete.json');assert done['processed_frames']==n and done['last_raw_frame']==(n-1)*5
  write(OUT/f'checkpoint{n}_{GROUP}_complete.json',dict(status='PASS',scene=C.name,group=GROUP,checkpoint=n,previous_checkpoint=old,
   commit=str(root/'repair/commit_complete.json'),initialized_from_zero=old==0,prior_repaired_state_inherited=old>0,GT_used=False))
 write(OUT/f'status_{GROUP}.json',dict(status='COMPLETE',scene=C.name,group=GROUP,processed_frames=400,last_raw_frame=1995))
if __name__=='__main__':
 try:main()
 except Exception:write(OUT/f'status_{GROUP}.json',dict(status='FAILED',scene=C.name,group=GROUP,traceback=traceback.format_exc()));raise
