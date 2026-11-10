from pathlib import Path
import hashlib,json,os,time
import numpy as np
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/ovimap_aligned_main_onlinev2_20261101')
OLD=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
FILL=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_fill_20261031')
OLD_C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031')
FILL_C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_fill_20261031')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
SRC=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src')
P='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 q.write_text(json.dumps(d,ensure_ascii=True,indent=2,allow_nan=False));q.replace(p)
def save(p,**d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp.npz')
 np.savez_compressed(q,**d);q.replace(p)
