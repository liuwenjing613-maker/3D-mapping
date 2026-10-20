from pathlib import Path
import json,subprocess,os,time,hashlib,traceback
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/room1/e1_repair')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 assert json.loads((R/'tests_complete.json').read_text())['status']=='PASS'
 codes={str(p):sha(p) for p in C.glob('*.py')}
 save('pipeline_code_freeze.json',{'status':'FROZEN_BEFORE_MAPPING_AND_GT','files':codes})
 completed=[]
 env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
 for stage in ['identity','wait_for_tracking','raw_repair','strict_repair','history_core','expand','E1']:
  if stage=='wait_for_tracking':
   while not (R/'tracking_queue_complete.json').exists():
    if (R/'tracking_queue_failure.json').exists():raise RuntimeError('tracking queue failed')
    save('pipeline_status.json',{'status':'RUNNING','stage':stage,'completed':completed,'unix':time.time()});time.sleep(15)
   assert json.loads((R/'tracking_queue_complete.json').read_text())['status']=='PASS'
   continue
  for p,h in codes.items():assert sha(Path(p))==h,p
  log=R/(stage+'.log');started=time.time()
  with log.open('xb') as f:
   q=subprocess.Popen([PY,'-u',str(C/'run_stage.py'),stage],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   save('pipeline_status.json',{'status':'RUNNING','stage':stage,'pid':q.pid,'completed':completed,'log':str(log),'unix':time.time()})
   assert q.wait()==0,stage+' failed; inspect '+str(log)
  completed.append({'stage':stage,'seconds':time.time()-started});print(json.dumps(completed[-1]),flush=True)
 save('pipeline_complete.json',{'status':'PASS','stages':completed,'GT_used':False})
 save('pipeline_status.json',{'status':'PASS','stage':'mapping_complete','completed':completed,'unix':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:save('pipeline_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
