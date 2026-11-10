from pathlib import Path
import os,sys,json,time,subprocess,traceback
root=Path(sys.argv[1]);branch=Path(sys.argv[2]);code=Path(sys.argv[3]);scene=root.parts[-3];py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python';status=root/'candidate_chain_status.json';start=time.time()
def read(p):return json.loads(Path(p).read_text())
def state(s,**kw):q=status.with_suffix('.tmp');q.write_text(json.dumps(dict(status=s,elapsed_seconds=time.time()-start,**kw),indent=2));q.replace(status)
def wait(p):
 while True:
  x=read(p) if p.exists() else {'status':'WAITING'}
  if x['status']=='FAILED':raise RuntimeError(str(p)+' failed '+str(x))
  if x['status']=='PASS':return
  state('WAITING',stage=p.name);time.sleep(10)
def run(stage,path,args=(),guard=False):
 state('RUNNING',stage=stage);env=dict(os.environ,CVPR_CHECKPOINT='50',CVPR_TRACK_GROUP='B',CVPR_SCENE=scene,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 if guard:env.update(PYTHONPATH=str(branch/'selection_guard')+':'+str(branch),CVPR_ONLINE_GUARD=str(root/'input_guard.json'))
 with (root/('first50_'+stage+'.log')).open('ab') as f:rc=subprocess.call([py,'-B','-u',str(path),*map(str,args)],cwd=path.parent,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 if rc:raise RuntimeError(stage+' failed: '+str(root/('first50_'+stage+'.log')))
env=dict(os.environ,CVPR_CHECKPOINT='50',CVPR_TRACK_GROUP='B',CVPR_SCENE=scene,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
try:
 run('groups',branch/'first50_groups_driver.py')
 run('medoids',code/'first100_medoids.py',[scene]);run('background',code/'first100_background070.py',[scene]);run('fixedH',code/'first100_fixedh_driver.py',[scene]);run('SAM3',code/'first100_sam3_scheduler.py',[root]);run('original_gate',Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/first100_gate_driver.py'),[root]);run('repair_commit',branch/'repair_supervisor.py',[root,branch])
 d=read(root/'actual_repair_driver_status.json');assert d['status']=='PASS' and d['stage']=='REAL_REPAIR_AND_COMMIT';state('PASS',stage='50_REAL_REPAIR_AND_COMMIT',GT_used=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
