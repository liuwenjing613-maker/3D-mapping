from pathlib import Path
import os,sys,json,time,subprocess,traceback
root=Path(sys.argv[1]);branch=Path(sys.argv[2]);s=root.parts[-3];st=root/'mapping_driver_status.json';t=time.time()
def state(status,**kw):q=st.with_suffix('.tmp');q.write_text(json.dumps(dict(status=status,elapsed_seconds=time.time()-t,**kw),indent=2));q.replace(st)
env={**os.environ,'CVPR_CHECKPOINT':'50','CVPR_TRACK_GROUP':'B','CVPR_MAPPING_RUN_ROOT':str(root),'CVPR_SURFACE_POLICY':'checkpoint','CVPR_SCENE':s,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None);py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
try:
 for stage,name in [('mapping_setup','map_support.py'),('mapping','map_continue.py')]:
  state('RUNNING',stage=stage);e=dict(env)
  if stage=='mapping':e.update(CVPR_ONLINE_GUARD=str(root/'mapping/input_guard.json'),PYTHONPATH=str(branch/'map_guard')+':'+str(branch))
  with (root/('first50_'+stage+'.log')).open('ab') as f:rc=subprocess.call([py,'-B','-u',str(branch/name)],cwd=branch,env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(stage+' failed '+str(rc))
 d=json.loads((root/'mapping/mapping_complete.json').read_text());assert d['status']=='PASS' and d['frame_count']==50 and not d['GT_used'] and not d['future_inputs_used'];state('PASS',stage='50_CAUSAL_MAPPING_ONLY',repair_executed=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
