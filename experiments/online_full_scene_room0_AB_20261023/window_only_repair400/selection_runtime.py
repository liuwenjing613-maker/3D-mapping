"""Timed child processes and cooperative GPU reservation; no other job is stopped."""
from common_continue import *
import subprocess,fcntl,contextlib,traceback
LIB='/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'
def environment(extra=None):
 return {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4',
  'CUDA_DEVICE_ORDER':'PCI_BUS_ID','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
  'PYTHONPATH':str(CONT/'selection_guard')+':'+str(CONT)+':'+str(C)+':/home/chenkejun/CVPR/repos/sam2_1',
  'CVPR_ONLINE_GUARD':str(ROOT/'input_guard.json'),'LD_LIBRARY_PATH':LIB,
  'XDG_CACHE_HOME':str(ROOT/'runtime_cache'),'CUDA_CACHE_PATH':str(ROOT/'runtime_cache/cuda'),
  'MPLCONFIGDIR':str(ROOT/'runtime_cache/mpl'),**(extra or {})}
WRAPPER="import sitecustomize,os; assert sitecustomize.settings_path and sitecustomize.current_cutoff==(int(os.environ['CVPR_CHECKPOINT'])-1)*5; import sys,runpy; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
@contextlib.contextmanager
def gpu_slot(status_root):
 locks=TOP/'continuation300/gpu_locks';locks.mkdir(parents=True,exist_ok=True)
 started=time.perf_counter();failed={};chosen=None
 while chosen is None:
  output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  candidates=[]
  for line in output.splitlines():
   index,uid,free,used,util=[x.strip() for x in line.split(',')]
   if int(free)>=14000 and int(util)<=20 and time.time()-failed.get(uid,0)>600:candidates.append((int(used),int(util),-int(free),int(index),uid))
  for used,util,free,index,uid in sorted(candidates):
   lock=(locks/(uid+'.lock')).open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   test=subprocess.run([PY,'-B','-c','import torch; torch.zeros(1,device="cuda"); print(torch.cuda.get_device_name())'],
     env=environment({'CUDA_VISIBLE_DEVICES':uid}),capture_output=True,text=True,timeout=45)
   if test.returncode:
    failed[uid]=time.time()
    with (locks/'device_failures.jsonl').open('a') as f:f.write(json.dumps(dict(unix=time.time(),uuid=uid,index=index,error=test.stderr[-2000:]))+'\n')
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();continue
   chosen=(uid,lock);break
  if chosen is None:
   dump(status_root/'gpu_wait.json',dict(status='WAITING',seconds=time.perf_counter()-started,requirement='verifiedCUDA,free>=14000MiB,util<=20%; never stops other jobs',failed_devices=list(failed)))
   time.sleep(15)
 uid,lock=chosen;wait=time.perf_counter()-started
 dump(status_root/'gpu_wait.json',dict(status='ACQUIRED',uuid=uid,wait_seconds=wait))
 try:yield uid,wait
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
def run(stage,script,args=(),python=PY,gpu=False,out=None):
 out=Path(out or CANDIDATE_ROOT);out.mkdir(parents=True,exist_ok=True)
 def execute(device=None,waiting=0):
  started=time.perf_counter();unix=time.time();code=None;proc=None
  try:
   with (out/(stage+'.log')).open('a') as log:
    proc=subprocess.Popen([python,'-B','-u','-c',WRAPPER,str(script),*map(str,args)],cwd=Path(script).parent,
      env=environment({'CUDA_VISIBLE_DEVICES':device} if device else {}),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    dump(out/('selection_progress.json' if out==CANDIDATE_ROOT else 'process_progress.json'),
      dict(status='RUNNING',stage=stage,pid=proc.pid,gpu=device,started_unix=unix,checkpoint=CHECKPOINT,group=GROUP))
    print('STAGE',CHECKPOINT,GROUP,stage,proc.pid,flush=True);code=proc.wait()
   if code:raise RuntimeError(f'{stage} failed exit{code}; {out/stage}.log')
  finally:
   row=dict(stage=stage,checkpoint=CHECKPOINT,group=GROUP,seconds=time.perf_counter()-started,gpu_wait_seconds=waiting,
    started_unix=unix,ended_unix=time.time(),pid=proc.pid if proc else None,gpu=device,returncode=code,status='PASS' if code==0 else 'FAILED_OR_INTERRUPTED')
   with (out/'process_timing_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  return row
 if gpu:
  with gpu_slot(out) as (device,waiting):return execute(device,waiting)
 return execute()
