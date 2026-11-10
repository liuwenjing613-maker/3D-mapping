from pathlib import Path
import os,json,hashlib,subprocess,sys
import numpy as np
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
SRC=C/'mapping_snapshot/revisable_instance_map/src'
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
COUNTS={'scene0011_00':200,'scene0050_00':200}
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(p,**kw):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**kw)
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def protocol():
 sys.path.insert(0,str(MAIN))
 from unified_eval.current_protocol import CURRENT_CONFIG,require_current_protocol
 require_current_protocol(CURRENT_CONFIG)
 return read(MAIN/'unified_eval/configs/current_protocol.json')
def run(args,log):
 args=list(map(str,args));protocol()
 env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with Path(log).open('a') as f:subprocess.run([sys.executable,'-B',str(Path('/home/chenkejun/CVPR/experiments/scannet_ovimap_strict_dbscan_20261104')/'evaluate_surface.py'),*args],cwd=MAIN,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
