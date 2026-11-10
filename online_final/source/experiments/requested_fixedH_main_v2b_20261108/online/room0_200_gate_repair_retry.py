from pathlib import Path
import os,json,time,subprocess,traceback
R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/room0/200/B');E=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online');B=E/'branches/room0';out=R/'downstream_chain_status.json';start=time.time();py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python';env={**os.environ,'CVPR_CHECKPOINT':'200','CVPR_TRACK_GROUP':'B','CVPR_SCENE':'room0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
def put(d):q=out.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(out)
try:
 for stage,cmd in [('original_gate',[py,'-B','-u',str(E/'first100_gate_driver.py'),str(R)]),('real_repair_and_commit',[py,'-B','-u',str(B/'repair_supervisor.py'),str(R),str(B)])]:
  put({'status':'RUNNING','stage':stage,'fallback':'full pixel projection only for zero stride-2 seed support','elapsed_seconds':time.time()-start})
  with (R/('downstream_retry_'+stage+'.log')).open('ab') as f:rc=subprocess.call(cmd,cwd=B,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(stage+' failed: exit '+str(rc))
 assert json.loads((R/'actual_repair_driver_status.json').read_text())['status']=='PASS';put({'status':'PASS','stage':'REAL_REPAIR_AND_COMMIT','elapsed_seconds':time.time()-start,'repair_complete':True})
except BaseException:put({'status':'FAILED','traceback':traceback.format_exc()});raise
