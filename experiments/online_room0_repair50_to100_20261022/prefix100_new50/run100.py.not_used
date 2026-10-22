from online_common import *
import os,subprocess,time,traceback
TOP=R.parent
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONPATH':str(C)}
def run(name,path,args=(),guard=True):
 start=time.perf_counter();started_unix=time.time();env=dict(ENV);p=None;code=None
 if guard:env.update(PYTHONPATH=str(C/'guard')+':'+str(C),CVPR_ONLINE_GUARD=str(R/'input_guard.json'))
 try:
  with (R/(name+'.log')).open('a') as log:
   p=subprocess.Popen([PY,'-B','-u',str(C/path),*args],env=env,cwd=C,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   write(R/'workflow_status.json',dict(stage=name,pid=p.pid,unix=time.time()))
   code=p.wait()
   if code:raise RuntimeError(name+' failed')
 finally:
  # Persist each completed/failed child immediately, even if a later stage fails.
  record=dict(stage=name,seconds=time.perf_counter()-start,started_unix=started_unix,
              ended_unix=time.time(),pid=p.pid if p else None,returncode=code,
              status='PASS' if code==0 else 'FAILED_OR_INTERRUPTED',scope='whole_child_process_wall')
  with (R/'process_timing_events.jsonl').open('a') as journal:
   journal.write(__import__('json').dumps(record,ensure_ascii=False)+'\n')
   journal.flush();os.fsync(journal.fileno())
 return record
def main():
 while not (TOP/'committed50.json').exists():
  write(R/'workflow_status.json',dict(stage='waiting_for_verified_50_commit',unix=time.time()));time.sleep(10)
 assert read(TOP/'committed50.json')['status']=='COMMITTED'
 timings=[]
 if not (R/'inputs_manifest.json').exists():timings.append(run('prepare_prefix','prepare_prefix.py',guard=False))
 if not (R/'mapping_complete.json').exists():timings.append(run('map_online','map_online.py'))
 assert read(R/'mapping_complete.json')['stopped_before_second_repair']
 timings.append(run('export_prefix_ply','export_prefix_ply.py'))
 if not (R/'selection_complete.json').exists():timings.append(run('selection','run_selection.py'))
 O=R/'all_edge_20261021'
 if not (O/'v1/audit.json').exists():timings.append(run('all_edge_prepare','all_edge/screen.py',['100','prepare']))
 if not (O/'complete.json').exists():timings.append(run('all_edge_select','all_edge/run_variant.py',['100','GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129']))
 timings.append(run('all_edge_render','all_edge/render.py',['100']))
 write(R/'timings.json',timings)
 write(R/'workflow_status.json',dict(status='PASS',stage='100_masks_ready_second_repair_not_started',unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:write(R/'workflow_failure.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
