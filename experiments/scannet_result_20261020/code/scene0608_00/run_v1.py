"""Execute the accepted upstream sequence in an isolated scene0608_00 experiment."""
from pathlib import Path
import os,json,subprocess,time,traceback
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020/scene0608_00')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
SAM='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
CROP='/data/chenkejun/CVPR/runtime/cropformer_env/bin/python'
ENV={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020/shared:/home/chenkejun/CVPR/repos/sam2_1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XDG_CACHE_HOME':str(R/'runtime_cache'),'CUDA_CACHE_PATH':str(R/'cuda_cache'),'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
def save(name,v):
 p=R/(name+'.json');q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def run(stage,script,python=PY,gpu=None):
 log=R/(stage+'_'+Path(script).stem+'.log');env=dict(ENV)
 if gpu:env['CUDA_VISIBLE_DEVICES']=gpu
 with log.open('ab') as f:
  p=subprocess.Popen([python,'-B','-u',str(C/stage/script)],cwd=C/stage,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 save('v1_status',dict(stage=stage,script=script,pid=p.pid,gpu=gpu,time=time.time()))
 return p
def check(p):
 assert p.wait()==0, 'Stage failed; see stage log'
def gpu(excluded=()):
 while True:
  rows=[]
  busy={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines() if line.strip()}
  for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
   index,uid,free,util=[x.strip() for x in line.split(',')]
   if uid not in excluded and int(free)>24000 and int(util)<35:rows.append((int(free),-int(util),uid,int(index)))
  for free,util,uid,index in sorted(rows,reverse=True):
   env={**ENV,'CUDA_VISIBLE_DEVICES':uid}
   test=subprocess.run([SAM,'-B','-c','import torch; x=torch.zeros(1,device="cuda"); print(torch.cuda.get_device_name(0))'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
   with (R/'gpu_checks.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),index=index,uuid=uid,free_MiB=free,exit_code=test.returncode,stderr=test.stderr[-1600:]))+'\n')
   if test.returncode==0:return uid
  save('v1_status',dict(stage='waiting_for_available_GPU',time=time.time()))
  time.sleep(30)
def main():
 try:
  check(run('frontend','prepare_inputs.py'))
  g1=gpu();check(run('frontend','infer_cropformer.py',CROP,g1))
  g2=gpu();check(run('frontend','infer_sam.py',SAM,g2))
  check(run('frontend','summarize_frontend.py'))
  check(run('overmerge','score.py'))
  check(run('intersection','compute.py'))
  check(run('versions','compute.py'))
  g=gpu();check(run('semantic_route','classify.py',PY,g));check(run('semantic_route','refine_cabinet.py',PY,g))
  check(run('semantic_route','route.py'))
  for stage in ['grouping_r2','grouping_r3','v1']:check(run(stage,'compute.py'))
  audit=json.loads((R/'v1/audit.json').read_text());assert audit['status']=='PASS'
  save('v1_complete',dict(status='PASS',stats=audit['stats'],time=time.time()))
  save('v1_status',dict(stage='complete',time=time.time()))
 except Exception:
  save('v1_failure',dict(error=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
