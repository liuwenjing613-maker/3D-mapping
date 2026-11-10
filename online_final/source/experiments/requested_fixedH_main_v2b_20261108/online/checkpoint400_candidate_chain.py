from pathlib import Path
import os,sys,time,json,subprocess,traceback
scene=sys.argv[1];E=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/branches')/scene;R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online')/scene/'400/B';out=R/'candidate_chain_status.json';start=time.time();py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def put(x):q=out.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(out)
def state(status,**kw):put(dict(status=status,elapsed_seconds=time.time()-start,tracking_or_repair=False,**kw))
def wait(p):
 while True:
  d=json.loads(p.read_text()) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError(str(p)+' failed: '+str(d))
  if d['status']=='PASS':return
  state('WAITING',stage=p.name);time.sleep(15)
try:
 wait(R/'candidate_prepare_status.json');env={**os.environ,'CVPR_CHECKPOINT':'400','CVPR_TRACK_GROUP':'B','CVPR_SCENE':scene,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CVPR_GPU_POOL':'0,1,2,3,4,5,6,7','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 procs=[]
 for name,args in [('model',['sam3']),('model',['cropformer']),('summary',[]),('evidence',[])]:
  script=E/('checkpoint400_'+name+'_driver.py');log=(R/('checkpoint400_'+name+('_'+args[0] if args else '')+'.log')).open('ab');p=subprocess.Popen([py,'-B','-u',str(script),*args],cwd=E,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append((p,log))
 state('RUNNING',stage='CANDIDATE_MODELS_AND_UPSTREAM',pids=[p.pid for p,l in procs]);wait(R/'candidates/upstream_evidence/driver_status.json')
 for name in ['versions','semantic','groups']:
  script=E/('checkpoint400_'+name+'_driver.py');state('RUNNING',stage=name)
  with (R/('checkpoint400_'+name+'.log')).open('ab') as log:rc=subprocess.call([py,'-B','-u',str(script)],cwd=E,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(name+' failed exit '+str(rc))
 state('PASS',stage='ORIGINAL_GROUPS_ONLY_NOT_MEDOIDS')
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
