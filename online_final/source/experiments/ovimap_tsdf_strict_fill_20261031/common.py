from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_fill_20261031')
OLD=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
OLD_C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031')
SRC=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
P='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
def save(p,**d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp.npz');np.savez_compressed(q,**d);q.replace(p)
