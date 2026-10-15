from pathlib import Path
import json,sys,os,subprocess,collections,time
import numpy as np
from PIL import Image
R=Path('/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4')
g=json.loads((R/'seed_manifest.json').read_text())['groups']
rows=[];batches={};overlaps=[]
for i,group in enumerate(g,1):
 p=group['primary'];m=np.asarray(Image.open(R/p['mask']))>0;assert int(m.sum())==p['area']
 slots=batches.setdefault(p['frame'],[])
 for old in slots:
  for oid,mask in old:
   n=int((mask&m).sum())
   if n:overlaps.append({'a':oid,'b':i,'pixels':n})
 assigned=False
 for slot in slots:
  if not any((mask&m).any() for _,mask in slot):slot.append((i,m));assigned=True;break
 if not assigned:slots.append([(i,m)])
print('BATCHES',json.dumps({'count':sum(map(len,batches.values())),'frames':len(batches),'overlaps':overlaps,'batch_members':{str(k):[[i for i,_ in x] for x in v] for k,v in sorted(batches.items())}}),flush=True)
print('RESOURCES',subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),flush=True)
print('MEM',Path('/proc/meminfo').read_text().splitlines()[:4],flush=True)
print('DISK',subprocess.check_output(['df','-h','/data/chenkejun/CVPR'],text=True),flush=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'3','LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
q=subprocess.run(['/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python','-c','import torch; a=torch.ones(5,device="cuda"); print(a.sum().item(),torch.cuda.get_device_name())'],env=env,text=True,capture_output=True,timeout=30)
print('GPU3_PROBE',q.returncode,q.stdout,q.stderr[-2000:],flush=True)
old=Path('/data/chenkejun/CVPR/results/room2_strict_manual_ab_20261010/batch_20fd547a5808/tracking_queue_complete.json')
if old.exists():
 d=json.loads(old.read_text());print('OLD_RUNTIME',{k:v for k,v in d.items() if k!='tasks'},flush=True)
