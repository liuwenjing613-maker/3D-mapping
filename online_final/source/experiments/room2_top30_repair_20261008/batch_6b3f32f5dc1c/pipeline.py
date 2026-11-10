"""Wait for frozen bidirectional results, then repair, evaluate and export once."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
CODE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 p=ROOT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def main():
 started=time.monotonic();env=dict(os.environ,OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
 names=['pipeline.py','cross_seed_diagnostics.py','alias_review.py','run_batch.py','joint_batch_ops.py','evaluate_batch.py','export_review.py']
 codes={str(CODE/n):sha(CODE/n) for n in names}
 write('pipeline_code_freeze.json',{'status':'FROZEN_BEFORE_REPAIR_AND_GT','code_sha256':codes,'GT_used':False})
 while not (ROOT/'tracking_queue_complete.json').exists():
  if (ROOT/'tracking_queue_failure.json').exists():raise RuntimeError('Tracking queue failed; preserve records and inspect failure')
  write('pipeline_status.json',{'status':'RUNNING','stage':'waiting_for_tracking','elapsed_seconds':round(time.monotonic()-started,2)})
  time.sleep(15)
 assert json.loads((ROOT/'tracking_queue_complete.json').read_text())['status']=='PASS'
 stages=[]
 for stage,script,receipt in [('cross_seed','cross_seed_diagnostics.py','cross_seed_diagnostics.json'),
  ('identity_review','alias_review.py','alias_review.json'),('repair','run_batch.py','repair_complete.json'),
  ('evaluation','evaluate_batch.py','evaluation_summary.json'),('export','export_review.py','review_bundle.json')]:
  for p,h in codes.items():assert sha(p)==h
  t=time.monotonic();log=ROOT/(stage+'.log')
  with log.open('xb') as stream:
   process=subprocess.Popen([PY,'-u',str(CODE/script)],stdout=stream,stderr=subprocess.STDOUT,env=env)
   write('pipeline_status.json',{'status':'RUNNING','stage':stage,'pid':process.pid,'log':str(log),'completed_stages':stages,'elapsed_seconds':round(time.monotonic()-started,2)})
   code=process.wait()
  assert code==0,f'{stage} exit {code}; inspect {log}'
  result=json.loads((ROOT/receipt).read_text());assert result['status'] in ['PASS','FROZEN']
  stages.append({'stage':stage,'seconds':round(time.monotonic()-t,2),'receipt_sha256':sha(ROOT/receipt)})
 write('pipeline_complete.json',{'status':'PASS','stages':stages,'elapsed_seconds':round(time.monotonic()-started,2)})
 write('pipeline_status.json',{'status':'PASS','stage':'complete','completed_stages':stages})
if __name__=='__main__':
 try:main()
 except Exception:write('pipeline_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
