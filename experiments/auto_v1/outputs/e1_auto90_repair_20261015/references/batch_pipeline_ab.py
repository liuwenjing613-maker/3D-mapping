from pathlib import Path
import os,json,time,subprocess,traceback,hashlib
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/room2_strict_manual_ab_20261010/batch_20fd547a5808')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def write(n,x):
 p=R/n;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 codes={str(p):sha(p) for p in C.glob('*.py')};write('pipeline_code_freeze.json',{'status':'FROZEN_BEFORE_MAPPING_AND_GT','files':codes})
 while not (R/'tracking_queue_complete.json').exists():
  if (R/'tracking_queue_failure.json').exists():raise RuntimeError('tracking failure; inspect queue receipt')
  write('pipeline_status.json',{'stage':'waiting_for_shared_tracking','status':'RUNNING'});time.sleep(15)
 assert json.loads((R/'tracking_queue_complete.json').read_text())['status']=='PASS'
 while not (R/'alias_review.json').exists():
  write('pipeline_status.json',{'stage':'waiting_for_GT_free_alias_review','status':'RUNNING'});time.sleep(15)
 stages=[];env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
 for stage,script in [('tracking_crosscheck','cross_seed_diagnostics.py'),('raw_observation_repair','run_batch.py'),('strict_original_repair','run_strict_repair.py'),('history_core_modified_repair','run_history_core.py'),('revision2_evaluation','evaluate_ab_r2.py')]:
  for p,h in codes.items():assert sha(Path(p))==h,p
  start=time.time();log=R/(stage+'.log')
  with log.open('xb') as f:
   p=subprocess.Popen([PY,'-u',str(C/script)],env=env,stdout=f,stderr=subprocess.STDOUT)
   write('pipeline_status.json',{'status':'RUNNING','stage':stage,'pid':p.pid,'completed':stages,'log':str(log)})
   assert p.wait()==0,stage+' failed: '+str(log)
  stages.append({'stage':stage,'seconds':time.time()-start})
 write('pipeline_complete.json',{'status':'PASS','stages':stages});write('pipeline_status.json',{'status':'PASS','stage':'complete','completed':stages})
if __name__=='__main__':
 try:main()
 except Exception:write('pipeline_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
