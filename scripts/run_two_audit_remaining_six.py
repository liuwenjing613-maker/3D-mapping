import os,json,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104')
SCENES=['room1','room2','office0','office1','office3','office4']
PY=sys.executable
LOG=ROOT/'run_logs/replica8';LOG.mkdir(exist_ok=True)
def step(scene,name,args):
 with (LOG/(scene+'_'+name+'.log')).open('a') as f:
  print(scene,name,'START',flush=True);subprocess.run([PY,'-u']+args,stdout=f,stderr=subprocess.STDOUT,check=True);print(scene,name,'DONE',flush=True)
def cpu(scene):
 old=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/main')/scene/'400/A/mapping'
 cache=ROOT/'observations'/scene;track=ROOT/'stable_tracks'/scene
 if not (cache/'manifest.json').exists():step(scene,'cache',['-m','instance_repair.two_audit_cache','build','--mapping-root',str(old),'--output',str(cache),'--mode','offline','--checkpoint','400'])
 if not (track/'manifest.json').exists():step(scene,'tracks',['-m','instance_repair.two_audit_tracks','build','--cache',str(cache),'--output',str(track)])
 step(scene,'prepare',['-c',"from scripts.two_audit_seed_filter import prepare;prepare("+repr(scene)+")"])
 return scene
failures={}
with ThreadPoolExecutor(max_workers=3) as pool:
 jobs={pool.submit(cpu,s):s for s in SCENES}
 for job in as_completed(jobs):
  scene=jobs[job]
  try:job.result()
  except Exception as e:failures[scene]=str(e);print('FAILED',scene,str(e),flush=True)
for scene in SCENES:
 if scene in failures:continue
 try:step(scene,'filter',['scripts/two_audit_seed_filter.py','--scene',scene])
 except Exception as e:failures[scene]=str(e);print('FAILED',scene,str(e),flush=True)
(ROOT/'replica8_run_status.json').write_text(json.dumps(dict(status='FAILED' if failures else 'PASS',failures=failures,scenes=SCENES),indent=2))
print('FINISHED',failures,flush=True)
