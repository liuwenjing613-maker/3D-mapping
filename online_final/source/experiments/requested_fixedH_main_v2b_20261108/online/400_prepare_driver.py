from pathlib import Path
import os,sys,json,time,subprocess,traceback
scene=sys.argv[1];E=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/branches')/scene;R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online')/scene/'400/B';sc=E/'selection_code/400/B';status=R/'candidate_prepare_status.json';start=time.time();py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def put(x):q=status.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(status)
try:
 while True:
  x=json.loads((R/'mapping_driver_status.json').read_text()) if (R/'mapping_driver_status.json').exists() else {'status':'WAITING'}
  if x['status']=='FAILED':raise RuntimeError('causal mapping failed '+str(x))
  if x['status']=='PASS':break
  put({'status':'WAITING_MAPPING','elapsed_seconds':time.time()-start});time.sleep(15)
 prior=json.loads((R.parent.parent/'200/B/repair/commit_complete.json').read_text());assert prior['status']=='PASS' and prior['processed_frames']==200
 env={**os.environ,'CVPR_CHECKPOINT':'400','CVPR_TRACK_GROUP':'B','CVPR_SCENE':scene,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 for stage,script,args in [('setup',E/'selection_setup.py',[]),('input_fence',E/'selection_fence.py',[]),('catalog',sc/'catalog/build_candidates.py',[scene]),('prepare',sc/'frontend/prepare_inputs.py',[])]:
  put({'status':'RUNNING','stage':stage,'elapsed_seconds':time.time()-start,'tracking_or_repair':False});e=dict(env)
  if stage!='setup':e.update(PYTHONPATH=str(E/'selection_guard')+':'+str(E),CVPR_ONLINE_GUARD=str(R/'input_guard.json'))
  with (R/(stage+'_prepare.log')).open('ab') as f:rc=subprocess.call([py,'-B','-u',str(script),*args],cwd=script.parent,env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(stage+' failed exit '+str(rc)+' see '+str(R/(stage+'_prepare.log')))
 put({'status':'PASS','stage':'CANDIDATE_INPUTS_ONLY','elapsed_seconds':time.time()-start,'tracking_or_repair':False})
except BaseException:put({'status':'FAILED','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
