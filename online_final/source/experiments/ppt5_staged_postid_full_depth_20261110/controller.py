from pathlib import Path
import os,sys,json,time,subprocess,traceback,fcntl
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110')
scene=sys.argv[1];base=R/'online'/scene;branch=C/'branches'/scene;status=base/'optimization_status.json'
lock=(base/'full_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time()
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def rd(p):return json.loads(p.read_text())
def state(status_value,**kw):
 q=status.with_suffix('.tmp');q.write_text(json.dumps(dict(status=status_value,baseline='PPT online-v2 B@a15f7612',filter='ppt5-full-depth-postid-uniform025-new3500-v1',mode='actual online',scene=scene,schedule=[50,100,200,300,400],elapsed_seconds=time.time()-started,pid=os.getpid(),**kw),indent=2));q.replace(status)
def run(stage,args,n):
 state('RUNNING',stage=stage,checkpoint=n)
 env={**os.environ,'CVPR_CHECKPOINT':str(n),'CVPR_TRACK_GROUP':'B','CVPR_SCENE':scene,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
 env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with (base/(stage+'.log')).open('ab') as log:subprocess.run([PY,'-B','-u',*map(str,args)],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
 while True:
  p=base/'50/B/controller_status.json';d=rd(p) if p.exists() else {'status':'WAITING'}
  if d['status']=='PASS':break
  if d['status']=='FAILED':raise RuntimeError('first50 failed: '+str(d))
  state('WAITING_FIRST50_ACTUAL_REPAIR',checkpoint=50);time.sleep(15)
 run('geometry50',[C/'verify_geometry.py',scene,50],50)
 for n,old in [(100,50),(200,100),(300,200),(400,300)]:
  root=base/str(n)/'B';root.mkdir(parents=True,exist_ok=True)
  p=root/'mapping_driver_status.json'
  if not p.exists() or rd(p)['status']!='PASS':run('mapping'+str(n),[C/'next_mapping_wait.py',root,branch,base/str(old)/'B'],n)
  run('geometry'+str(n),[C/'verify_geometry.py',scene,n],n)
  p=root/'candidate_chain_status.json'
  if not p.exists() or rd(p)['status']!='PASS':run('repair_chain'+str(n),[C/'next_chain.py',root,branch,C],n)
  cp=rd(root/'repair/commit_complete.json');assert cp['status']=='PASS' and cp['processed_frames']==n
 run('finalB',[C/'finish_finalB.py',scene],400)
 run('PPT_history_DBSCAN',[C/'terminal_dbscan.py',scene],400)
 state('PASS_REPAIR_AND_PPT_POSTPROCESS',stage='ALL5_ACTUAL_COMMITS_FINALB_PPT_HISTORY_DBSCAN',evaluation_complete=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
