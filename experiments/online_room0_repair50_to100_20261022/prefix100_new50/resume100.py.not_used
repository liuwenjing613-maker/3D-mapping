from online_common import *
import run100 as runner
import time,traceback,os
def main():
 start=time.perf_counter();timings=[]
 timings.append(runner.run('selection_resume','run_selection.py',['--start','v2']))
 assert read(R/'selection_complete.json')['status']=='PASS'
 O=R/'all_edge_20261021'
 if not (O/'v1/audit.json').exists():timings.append(runner.run('all_edge_prepare','all_edge/screen.py',['100','prepare']))
 if not (O/'complete.json').exists():timings.append(runner.run('all_edge_select','all_edge/run_variant.py',['100','GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129']))
 timings.append(runner.run('all_edge_render','all_edge/render.py',['100']))
 write(R/'resume_elapsed.json',dict(status='PASS',seconds=time.perf_counter()-start,stages=timings,resumed_at='v2',model_inference_repeated=False,second_repair_applied=False))
 for f in ('workflow_failure.json','selection_failure.json','v2/failure.json'):
  p=R/f
  if p.exists():
   v=read(p);v.update(resolved=True,resolution='ROI digest frame-token false positive and missing inherited containment config resolved; resumed v2 without model rerun',resolved_at=time.time());write(p,v)
 write(R/'workflow_status.json',dict(status='PASS',stage='100_masks_ready_second_repair_not_started',unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:write(R/'workflow_failure.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
