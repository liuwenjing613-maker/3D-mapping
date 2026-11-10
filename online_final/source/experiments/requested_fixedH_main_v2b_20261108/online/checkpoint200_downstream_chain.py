from pathlib import Path
import os,sys,json,time,subprocess,traceback
scene=sys.argv[1];D=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online');B=D/'branches'/scene;R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online')/scene/'200/B';out=R/'downstream_chain_status.json';start=time.time();py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python';env={**os.environ,'CVPR_CHECKPOINT':'200','CVPR_TRACK_GROUP':'B','CVPR_SCENE':scene,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
def put(d):q=out.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(out)
def state(s,**kw):put(dict(status=s,elapsed_seconds=time.time()-start,**kw))
def wait(p):
 while True:
  d=json.loads(p.read_text()) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError(str(p)+' failed '+str(d))
  if d['status']=='PASS':return
  state('WAITING',stage='ORIGINAL_GROUPING');time.sleep(15)
def run(stage,script,args):
 state('RUNNING',stage=stage,repair_complete=False)
 with (R/('downstream_'+stage+'.log')).open('ab') as log:rc=subprocess.call([py,'-B','-u',str(script),*args],cwd=B,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 if rc:raise RuntimeError(stage+' failed exit '+str(rc)+' see '+str(R/('downstream_'+stage+'.log')))
try:
 wait(R/'candidate_chain_status.json')
 run('medoids',D/'checkpoint200_medoids.py',[scene]);run('background',D/'checkpoint200_background070.py',[scene]);run('fixedH',D/'checkpoint200_fixedh_driver.py',[scene]);run('sam3_tracking',D/'checkpoint200_sam3_scheduler.py',[str(R)]);run('original_gate',D/'first100_gate_driver.py',[str(R)]);run('real_repair_and_commit',B/'repair_supervisor.py',[str(R),str(B)])
 status=json.loads((R/'actual_repair_driver_status.json').read_text());assert status['status']=='PASS' and status['stage']=='REAL_REPAIR_AND_COMMIT';state('PASS',stage='REAL_REPAIR_AND_COMMIT',repair_complete=True)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
