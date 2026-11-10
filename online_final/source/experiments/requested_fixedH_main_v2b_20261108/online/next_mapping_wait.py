from pathlib import Path
import os,sys,json,time,subprocess,traceback
root=Path(sys.argv[1]);branch=Path(sys.argv[2]);prior=Path(sys.argv[3]);n=int(root.parent.name);root.mkdir(parents=True,exist_ok=True);status=root/'mapping_driver_status.json';start=time.time()
def put(d):q=status.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(status)
try:
 while True:
  p=prior/'actual_repair_driver_status.json';d=json.loads(p.read_text()) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError('previous real repair failed: '+str(d))
  if d['status']=='PASS':break
  put({'status':'WAITING_PREVIOUS_COMMIT','elapsed_seconds':time.time()-start});time.sleep(15)
 commit=json.loads((prior/'repair/commit_complete.json').read_text());assert commit['status']=='PASS' and commit['processed_frames']==int(prior.parent.name)
 env={**os.environ,'CVPR_CHECKPOINT':str(n),'CVPR_TRACK_GROUP':'B','CVPR_MAPPING_RUN_ROOT':str(root),'CVPR_SURFACE_POLICY':'checkpoint','CVPR_SCENE':root.parts[-3],'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 for stage,name in [('mapping_setup','map_support.py'),('mapping','map_continue.py')]:
  e=dict(env)
  if stage=='mapping':e.update(CVPR_ONLINE_GUARD=str(root/'mapping/input_guard.json'),PYTHONPATH=str(branch/'map_guard')+':'+str(branch))
  put({'status':'RUNNING','stage':stage,'elapsed_seconds':time.time()-start})
  with (root/(stage+'.log')).open('ab') as f:rc=subprocess.call(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(branch/name)],cwd=branch,env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(stage+' failed '+str(rc))
 done=json.loads((root/'mapping/mapping_complete.json').read_text());assert done['status']=='PASS' and done['frame_count']==n and not done['GT_used'] and not done['future_inputs_used'] and done['prior_commit_sha256']
 put({'status':'PASS','stage':'CAUSAL_MAPPING_ONLY_AFTER_PREVIOUS_REPAIR','elapsed_seconds':time.time()-start,'GT_used':False,'repair_executed':False})
except BaseException:put({'status':'FAILED','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
