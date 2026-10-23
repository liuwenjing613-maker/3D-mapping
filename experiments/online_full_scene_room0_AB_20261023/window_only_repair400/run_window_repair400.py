"""Run already-authorized full A/B checkpoint400 repairs, without changing kernels."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common_continue import *
from selection_runtime import run

def main():
 started=time.perf_counter();timings=[]
 steps=[('input_fence','selection_fence.py',ROOT/'input_fence_audit.json'),('selection_qa','selection_qa.py',CANDIDATE_ROOT/'online_qa.json'),
  ('prepare','track_prepare.py',R/'prepare_complete.json'),('queue','track_queue_with_cache.py',R/'tracking_queue_complete.json'),
  ('gate','track_gate.py',R/'gate_complete.json'),('repair','repair_continue.py',ROOT/'repair/repair_complete.json'),('commit','commit_continue.py',ROOT/'repair/commit_complete.json')]
 for name,script,marker in steps:
  if marker.exists():
   assert read(marker)['status']=='PASS';continue
  dump(ROOT/'status.json',dict(status='RUNNING',stage=name,pid=os.getpid(),started_unix=time.time(),seconds=time.perf_counter()-started))
  timings.append(run(name,HERE/script,out=ROOT/'driver_logs'))
  assert read(marker)['status']=='PASS'
  if name=='prepare':
   import track_queue_with_cache as q
   cases=read(R/'seed_diagnostics.json')['cases']
   hits=[dict(case_uid=c['case_uid'],match=q.prior_case(c) is not None) for c in cases]
   dump(ROOT/'tracking_cache_plan.json',dict(status='PASS',cases=hits,matching_cases=sum(c['match'] for c in hits),total_cases=len(hits)))
 queue=read(R/'tracking_queue_complete.json');gate=read(R/'gate_complete.json');commit=read(ROOT/'repair/commit_complete.json')
 assert commit['processed_frames']==400 and commit['last_raw_frame']==1995
 for p,h in read(ROOT/'run_input_freeze.json')['source_sha256'].items():assert digest(p)==h,p
 for row in read(ROOT.parent.parent/'adapter_audit.json')['unchanged_numerical_files']:
  assert digest(HERE/row['file'])==row['source_sha256']==digest(row['source'])
 dump(R/'complete.json',dict(status='PASS',checkpoint=400,group=GROUP,cases=gate['cases'],objects=gate['objects'],
  inferred_cases=queue['inferred_cases'],reused_cases=queue['reused_cases'],tracking_queue_seconds=queue['seconds'],gate_seconds=gate['gate_seconds'],
  numerical_rules_unchanged=True,cache_mode='strict fingerprint only; Gate and repair recomputed'))
 dump(ROOT/'repair_pipeline_complete.json',dict(status='PASS',group=GROUP,seconds=time.perf_counter()-started,timings=timings,original_inputs_unchanged=True))
 dump(ROOT/'status.json',dict(status='REPAIR_COMPLETE',group=GROUP,seconds=time.perf_counter()-started,final_commit=str(ROOT/'repair/commit_complete.json')))
 print('REPAIR_PIPELINE_COMPLETE',GROUP,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'status.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise
