import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/online_final_trusted_20261110')
import write_fence
"""Timed child processes and cooperative GPU reservation; no other job is stopped."""
from common_continue import *
import subprocess,fcntl,contextlib,traceback
LIB='/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'
def environment(extra=None):
 return {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4',
  'CUDA_DEVICE_ORDER':'PCI_BUS_ID','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
  'PYTHONPATH':str(CONT/'selection_guard')+':'+str(CONT)+':'+str(C)+':/home/chenkejun/CVPR/repos/sam2_1:/data/chenkejun/CVPR/envs/sam3_compat:/home/chenkejun/CVPR/repos/sam3-main',
  'CVPR_ONLINE_GUARD':str(ROOT/'input_guard.json'),'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/envs/sam3_cuda_link:'+LIB,
  'XDG_CACHE_HOME':str(ROOT/'runtime_cache'),'CUDA_CACHE_PATH':str(ROOT/'runtime_cache/cuda'),
  'MPLCONFIGDIR':str(ROOT/'runtime_cache/mpl'),**(extra or {})}
WRAPPER='import sitecustomize,os; assert sitecustomize.settings_path and sitecustomize.current_cutoff==(int(os.environ[\'CVPR_CHECKPOINT\'])-1)*5; exec("if os.environ.get(\'CVPR_GPU_CAP_MIB\'):\\n import torch\\n torch.cuda.set_per_process_memory_fraction(int(os.environ[\'CVPR_GPU_CAP_MIB\'])*1048576/torch.cuda.get_device_properties(0).total_memory)"); import sys,runpy; sys.path.extend([\'/home/chenkejun/beauty/conceptgraphs/envs/cg-main/lib/python3.10/site-packages\',\'/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/lib/python3.10/site-packages\'] if sys.version_info[:2]==(3,10) else []); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name=\'__main__\')'
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
from gpu_memory_scheduler import acquire

def run(stage,script,args=(),python=PY,gpu=False,out=None):
 out=Path(out or CANDIDATE_ROOT);out.mkdir(parents=True,exist_ok=True)
 def execute(device=None,waiting=0,lease=None):
  started=time.perf_counter();unix=time.time();code=None;proc=None
  try:
   with (out/(stage+'.log')).open('a') as log:
    proc=subprocess.Popen([python,'-B','-u','-c',WRAPPER,str(script),*map(str,args)],cwd=Path(script).parent,
      env=environment({**({'CUDA_VISIBLE_DEVICES':device,'CVPR_GPU_CAP_MIB':str(lease.cap_mib)} if device else {}),**({'PYTHONPATH':str(CONT/'selection_guard')+':'+str(CONT)+':'+str(C)} if python==CROP else {})}),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if lease:lease.attach(proc.pid)
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
  budget={'sam3':14000,'cropformer':40000,'classify':8000,'cabinet':16000}.get(stage,22000)
  for attempt in range(2):
   lease=acquire(budget,stage+':'+str(out),out)
   try:return execute(lease.uid,lease.data['wait_seconds'],lease)
   except RuntimeError:
    if attempt or 'out of memory' not in (out/(stage+'.log')).read_text(errors='replace').lower():raise
    budget=min(47000,budget+6000)
    dump(out/'shared_retry.json',dict(status='RETRY_LARGER_MEMORY_BUDGET',stage=stage,budget_mib=budget))
   finally:lease.close()

 return execute()
