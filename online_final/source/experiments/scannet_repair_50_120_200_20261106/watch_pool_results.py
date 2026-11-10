from pathlib import Path
import json,time,sys,os,subprocess,traceback
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106');O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
v=sys.argv[1];base=v.replace('_DBSCAN','');isdb=v.endswith('_DBSCAN');status=O/'aggregate'/v/'watch_status.json';status.parent.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def state(s,**kw):
 q=status.with_suffix('.tmp');q.write_text(json.dumps(dict(status=s,variant=v,unix=time.time(),**kw),indent=2));q.replace(status)
try:
 while True:
  ready=[]
  for scene in ['scene0011_00','scene0050_00']:
   d=O/('dbscan' if isdb else 'evaluation')/base/scene
   complete=d/'complete.json';report=(d/'status.json') if isdb else O/'published'/base/scene/'finish_status.json'
   if report.exists() and read(report)['status']=='FAILED':raise RuntimeError('Required producer failed: '+str(report))
   if complete.exists() and read(complete)['status']=='PASS':ready.append(scene)
  if len(ready)==2:break
  state('WAITING',ready_scenes=ready);time.sleep(15)
 state('RUNNING',stage='pool_current_main_R2')
 with (status.parent/'run.log').open('a') as f:subprocess.run([sys.executable,'-B','-u',str(C/'pool_scannet_r2.py'),v],cwd=C,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert read(status.parent/'scope_audit.json')['status']=='PASS';state('PASS')
except Exception:state('FAILED',traceback=traceback.format_exc());raise
