from pathlib import Path
import os,sys,json,time,traceback,subprocess
root=Path(sys.argv[1]);p=root/'gate_driver_status.json';start=time.time()
def write(d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
try:
 while True:
  q=root/'tracking_sam3/driver_status.json';d=json.loads(q.read_text()) if q.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError('SAM3 tracking failed: '+str(d))
  if d['status']=='PASS':break
  write({'status':'WAITING_TRACKING','elapsed_seconds':time.time()-start});time.sleep(15)
 write({'status':'RUNNING','stage':'ORIGINAL_GATE','elapsed_seconds':time.time()-start})
 with (root/'gate_driver.log').open('ab') as f:rc=subprocess.call(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u','/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/first100_original_gate.py',str(root)],env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 if rc:raise RuntimeError('gate failed see gate_sam3/status.json')
 write({'status':'PASS','stage':'ORIGINAL_GATE_ONLY_NOT_REPAIR','elapsed_seconds':time.time()-start})
except BaseException:write({'status':'FAILED','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
