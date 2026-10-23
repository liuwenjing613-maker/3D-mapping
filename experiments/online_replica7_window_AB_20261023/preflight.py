"""Read-only availability checks for the seven requested Replica scenes."""
from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002')
scenes=['room1','room2','office0','office1','office2','office3','office4']
rows=[]
for scene in scenes:
 p=R/'configs'/f'{scene}.json';c=json.loads(p.read_text());source=Path(c['source']['scene_root']).resolve();masks=Path(c['source']['mask_root']).resolve();frozen=Path('/data/chenkejun/CVPR/datasets/cropformer_frozen')/scene
 missing=[]
 for fid in range(1996):
  files=[source/f'results/frame{fid:06d}.jpg']
  if fid%5==0:files += [source/f'results/depth{fid:06d}.png',masks/f'frame{fid:06d}.png',frozen/f'frame{fid:06d}.png']
  missing.extend(str(x) for x in files if not x.is_file())
 traj=source/c['source']['trajectory'];frames=frozen/'frames.jsonl'
 rows.append(dict(scene=scene,status='PASS' if not missing and traj.is_file() and frames.is_file() else 'FAIL',config=str(p),source=str(source),masks=str(masks),frozen=str(frozen),missing=missing[:8],missing_count=len(missing),trajectory_rows=sum(1 for _ in traj.open()),config_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(rows,indent=2))
print(subprocess.check_output(['free','-h'],text=True))
