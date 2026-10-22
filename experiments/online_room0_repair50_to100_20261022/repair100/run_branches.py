from pathlib import Path
import os,json,sys,time,subprocess,traceback
C=Path(__file__).resolve().parent
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
R=TOP/'repair100';PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def write(path,x):
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2));temp.replace(path)
def main():
 R.mkdir(exist_ok=True)
 guard=json.loads((TOP/'tracking100/input_guard.json').read_text())
 guard['root']=str(R)
 guard['allowed_prefix_cache_roots']=list(dict.fromkeys([*guard['allowed_prefix_cache_roots'],str(TOP/'tracking100')]))
 write(R/'input_guard.json',guard)
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4',
 'PYTHONPATH':str(C.parent/'tracking100/guard'),'CVPR_ONLINE_GUARD':str(R/'input_guard.json')}
 launched=[]
 for group in ['A','B']:
  out=R/group;out.mkdir(exist_ok=True);log=(out/'pipeline.log').open('a')
  started=time.perf_counter();unix=time.time()
  proc=subprocess.Popen([PY,'-B','-u',str(C/'pipeline.py')],cwd=C,env={**env,'CVPR_TRACK_GROUP':group},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  launched.append((group,proc,log,started,unix))
  write(out/'process_status.json',dict(status='RUNNING',pid=proc.pid,started_unix=unix))
 pending=list(launched)
 while pending:
  for item in pending[:]:
   group,proc,log,started,unix=item
   code=proc.poll()
   if code is None:continue
   log.close();pending.remove(item)
   write(R/group/'process_status.json',dict(status='PASS' if code==0 else 'FAILED',pid=proc.pid,returncode=code,started_unix=unix,ended_unix=time.time(),seconds=time.perf_counter()-started,scope='pipeline_process_including_import_setup_no_tracking_gate_commit_or_export'))
  if pending:time.sleep(1)
 write(R/'pipeline_processes_complete.json',dict(status='PASS' if all(x[1].returncode==0 for x in launched) else 'FAILED',groups={x[0]:json.loads((R/x[0]/'process_status.json').read_text()) for x in launched}))
if __name__=='__main__':
 try:main()
 except Exception:
  R.mkdir(exist_ok=True);write(R/'controller_failure.json',dict(traceback=traceback.format_exc()));raise
