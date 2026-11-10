from pathlib import Path
import sys,os,json,time,subprocess,traceback,fcntl
code=Path(sys.argv[1]);variant=code.name;scene=sys.argv[2]
R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')/variant/scene
C=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online');branch=code/'branches'/scene
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.time();status=R/'optimization_status.json'
def read(p):return json.loads(p.read_text())
def state(s,**kw):
 d={'status':s,'variant':variant,'scene':scene,'schedule':[100,300,400],'elapsed_seconds':time.time()-started,'GT_used_for_repair':False,**kw}
 t=status.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(status)
def wait(p):
 while True:
  if p.exists():
   d=read(p)
   if d['status']=='FAILED':raise RuntimeError(str(p)+' '+str(d))
   if d['status']=='PASS':return
  state('WAITING',stage=str(p));time.sleep(15)
def run(stage,cmd,n):
 env=dict(os.environ,CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP='B',CVPR_SCENE=scene,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 state('RUNNING',stage=stage,checkpoint=n)
 with (R/(stage+'.log')).open('ab') as log:
  p=subprocess.run([PY,'-B','-u',*map(str,cmd)],cwd=branch,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(stage+' failed; '+str(R/(stage+'.log')))
try:
 wait(R/'100/B/threshold_chain_status.json')
 for n,previous in [(300,100),(400,300)]:
  root=R/str(n)/'B'
  if not (root/'mapping_driver_status.json').exists() or read(root/'mapping_driver_status.json')['status']!='PASS':
   run('mapping'+str(n),[C/'next_mapping_wait.py',root,branch,R/str(previous)/'B'],n)
  if not (root/'candidate_chain_status.json').exists() or read(root/'candidate_chain_status.json')['status']!='PASS':
   run('repair_chain'+str(n),[code/'next_chain.py',root,branch,code],n)
  assert read(root/'repair/commit_complete.json')['processed_frames']==n
 root=R/'400/B'
 run('postprocess400',[C/'replica_online_postprocess.py',root,branch],400)
 run('evaluate400',[code/'replica_online_final_eval.py',scene],400)
 assert read(root/'evaluation_current_r2/status.json')['status']=='PASS'
 state('PASS',stage='REAL_ONLINE_100_300_400_COMMIT_FINALB_DBSCAN_MAIN_R2')
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
