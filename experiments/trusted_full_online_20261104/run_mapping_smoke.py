"""Run only the first new mapping interval, reusable by the full branch."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,time,traceback,fcntl
C=Path(__file__).resolve().parent
scene=sys.argv[1];assert scene in ['room0','room2']
code=C/'online'/scene
out=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104/online')/scene/'100/B'
out.mkdir(parents=True,exist_ok=True)
def write(name,obj):
 p=out/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
with (out/'mapping_smoke.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files=[code/n for n in ['map_support.py','map_continue.py','scene_continue.py','commit_continue.py','trusted_online.py','trusted_partition.py']]+[C/n for n in ['aligned_surface.py','write_fence.py']]
 hashes={str(p):sha(p) for p in files};write('mapping_smoke_source_hashes.json',hashes)
 env={**os.environ,'CVPR_SCENE':scene,'CVPR_CHECKPOINT':'100','CVPR_TRACK_GROUP':'B','CVPR_MAPPING_RUN_ROOT':str(out),'CVPR_SURFACE_POLICY':'checkpoint','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4'}
 env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
 try:
  for stage in ['map_support.py','map_continue.py']:
   env2=dict(env)
   if stage=='map_continue.py':env2.update(CVPR_ONLINE_GUARD=str(out/'mapping/input_guard.json'),PYTHONPATH=str(code/'map_guard')+':'+str(code))
   write('mapping_smoke_status.json',dict(status='RUNNING',pid=os.getpid(),scene=scene,stage=stage,unix=time.time()))
   with (out/('mapping_setup.log' if stage=='map_support.py' else 'mapping.log')).open('a') as log:
    subprocess.run([py,'-B','-u',str(code/stage)],cwd=code,env=env2,stdout=log,stderr=subprocess.STDOUT,check=True)
  for p,h in hashes.items():assert sha(Path(p))==h,p
  done=json.loads((out/'mapping/mapping_complete.json').read_text());assert done['status']=='PASS'
  audit=json.loads((out/'mapping/partition_audit.json').read_text())
  assert len(list((out/'mapping/partition_layers').glob('f*.png')))==50
  modes={}
  for row in audit:modes[row['mode']]=modes.get(row['mode'],0)+1
  write('mapping_smoke_status.json',dict(status='PASS',scene=scene,modes=modes,partition_frames=50,source_hashes_unchanged=True,unix=time.time()))
 except Exception:
  write('mapping_smoke_status.json',dict(status='FAILED',scene=scene,traceback=traceback.format_exc(),unix=time.time()))
  raise
