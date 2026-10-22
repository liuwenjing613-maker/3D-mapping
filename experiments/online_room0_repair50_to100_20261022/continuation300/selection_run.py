"""Wait for this branch's real mapping checkpoint, then candidates and dense tracking/Gate."""
from common_continue import *
from selection_runtime import run,environment
import subprocess,traceback
SC=CONT/'selection_code'/str(CHECKPOINT)/GROUP
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 while not (B/'mapping_complete.json').exists() or read(B/'mapping_complete.json').get('status')!='PASS':
  dump(ROOT/'selection_tracking_status.json',dict(status='WAITING_MAPPING',checkpoint=CHECKPOINT,group=GROUP,unix=time.time()))
  time.sleep(10)
 # Setup runs before the guard exists; it only links/hashes exact arrived input files.
 if not (CANDIDATE_ROOT/'setup_complete.json').exists():
  start=time.perf_counter()
  subprocess.run([PY,'-B','-u',str(CONT/'selection_setup.py')],cwd=CONT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},check=True)
 if not (ROOT/'input_fence_audit.json').exists():run('input_fence',CONT/'selection_fence.py',out=CANDIDATE_ROOT)
 if not (CANDIDATE_ROOT/'selection_complete.json').exists():
  started=time.perf_counter();timings=[]
  steps=[
   ('catalog',SC/'catalog/build_candidates.py',['room0'],PY,False, CANDIDATE_ROOT/'catalog/room0/candidate_manifest.json'),
   ('prepare',SC/'frontend/prepare_inputs.py',[],PY,False, CANDIDATE_ROOT/'frontend/input_manifest.json'),
   ('exact_model_cache',CONT/'selection_cache.py',[],PY,False, CANDIDATE_ROOT/'exact_model_cache_audit.json'),
   ('cropformer',SC/'frontend/infer_cropformer.py',[],CROP,True, CANDIDATE_ROOT/'frontend/cropformer/audit.json'),
   ('sam',SC/'frontend/infer_sam.py',[],SAM,True, CANDIDATE_ROOT/'frontend/sam2_1/audit.json'),
   ('summary',SC/'frontend/summarize_frontend.py',[],PY,False, CANDIDATE_ROOT/'frontend/pipeline_complete.json'),
   ('overmerge',SC/'overmerge/score.py',[],PY,False, CANDIDATE_ROOT/'overmerge/audit.json'),
   ('intersection',SC/'intersection/compute.py',[],PY,False, CANDIDATE_ROOT/'intersection/web/intersection.json'),
   ('versions',SC/'versions/compute.py',[],PY,False, CANDIDATE_ROOT/'versions/selection.json'),
   ('semantic_classify',SC/'semantic_route/classify.py',[],PY,True, CANDIDATE_ROOT/'semantic_route/semantic.json'),
   ('semantic_cabinet',SC/'semantic_route/refine_cabinet.py',[],PY,True, CANDIDATE_ROOT/'semantic_route/cabinet_context_rule.json'),
   ('semantic_route',SC/'semantic_route/route.py',[],PY,False, CANDIDATE_ROOT/'semantic_route/selection.json'),
   ('grouping_r2',SC/'grouping_r2/compute.py',[],PY,False, CANDIDATE_ROOT/'grouping_r2/audit.json'),
   ('grouping_r3',SC/'grouping_r3/compute.py',[],PY,False, CANDIDATE_ROOT/'grouping_r3/audit.json'),
   ('v1',SC/'v1/compute.py',[],PY,False, CANDIDATE_ROOT/'v1/audit.json'),
   ('semantic_background',SC/'v2/semantic/run.py',['--manifest',CANDIDATE_ROOT/'v1/seed_manifest.json','--raw-config',SC/'config.json','--out',CANDIDATE_ROOT/'v2/semantic','--config',SC/'v2/semantic/config.json'],PY,True,CANDIDATE_ROOT/'v2/semantic/classification.json'),
   ('v2',SC/'v2/measure_and_select.py',[],PY,False,CANDIDATE_ROOT/'v2/complete.json'),
   ('all_edge_prepare',SC/'all_edge/screen.py',[CHECKPOINT,'prepare'],PY,False,CANDIDATE_ROOT/'all_edge_20261021/prepare_complete.json'),
   ('all_edge_semantic',SC/'v2/semantic/run.py',['--manifest',CANDIDATE_ROOT/'all_edge_20261021/v1/seed_manifest.json','--raw-config',SC/'config.json','--out',CANDIDATE_ROOT/'all_edge_20261021/v2/semantic','--config',SC/'v2/semantic/config.json'],PY,True,CANDIDATE_ROOT/'all_edge_20261021/v2/semantic/classification.json'),
   ('all_edge_select',SC/'all_edge/screen.py',[CHECKPOINT,'select'],PY,False,CANDIDATE_ROOT/'all_edge_20261021/complete.json'),
   ('all_edge_render',SC/'all_edge/render.py',[CHECKPOINT],PY,False,CANDIDATE_ROOT/'all_edge_20261021/review_complete.json'),
   ('qa',CONT/'selection_qa.py',[],PY,False,CANDIDATE_ROOT/'online_qa.json')]
  skipped=[]
  for name,script,args,python,gpu,marker in steps:
   events=CANDIDATE_ROOT/'process_timing_events.jsonl'
   previous=[json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
   if marker.exists() and any(x['stage']==name and x['status']=='PASS' for x in previous):skipped.append(name);continue
   timings.append(run(name,script,args,python=python,gpu=gpu,out=CANDIDATE_ROOT))
  audit=read(CANDIDATE_ROOT/'online_qa.json');assert audit['status']=='PASS'
  allrows=[json.loads(x) for x in (CANDIDATE_ROOT/'process_timing_events.jsonl').read_text().splitlines()]
  successful=[x for x in allrows if x['status']=='PASS' and x['stage']!='input_fence']
  production=[x for x in successful if x['stage'] not in ['qa','all_edge_render']]
  dump(CANDIDATE_ROOT/'selection_complete.json',dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,retained=audit['retained'],seconds=time.perf_counter()-started,
   timing_scope='this invocation wall clock including GPU waiting; after input setup; excludes previous attempts',production_seconds=sum(x['seconds'] for x in production),
   gpu_wait_seconds=sum(x['gpu_wait_seconds'] for x in successful),process_timings=successful,skipped_previously_complete_stages=skipped,
   source_manifest=audit['source_manifest'],candidate_raw_frame_range=audit['candidate_raw_frame_range'],scoring_history=[1,CHECKPOINT],current_checkpoint_repaired=False))
  dump(CANDIDATE_ROOT/'selection_progress.json',dict(status='PASS',stage='complete',retained=audit['retained'],checkpoint=CHECKPOINT,group=GROUP))
 if not (R/'complete.json').exists():
  started=time.perf_counter();timings=[]
  for name,script,marker in [('prepare',CONT/'track_prepare.py',R/'prepare_complete.json'),('queue',CONT/'track_queue.py',R/'tracking_queue_complete.json'),('gate',CONT/'track_gate.py',R/'gate_complete.json')]:
   if marker.exists():continue
   timings.append(run(name,script,out=R))
  gate=read(R/'gate_complete.json');queue=read(R/'tracking_queue_complete.json')
  dump(R/'complete.json',dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,seconds=time.perf_counter()-started,
   timing_scope='this tracking orchestration invocation; includes GPU waiting; subprocess times are nested',
   process_timings=timings,cases=gate['cases'],objects=gate['objects'],tracking_queue_seconds=queue['seconds'],
   inferred_cases=queue['inferred_cases'],reused_cases=queue['reused_cases'],gate_seconds=gate['gate_seconds'],
   raw_frames=RAW_COUNT,raw_cutoff=CURRENT_RAW,current_checkpoint_repaired=False))
 dump(ROOT/'selection_tracking_status.json',dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,stage='gate_ready_for_repair',unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:
  dump(ROOT/'selection_tracking_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise
