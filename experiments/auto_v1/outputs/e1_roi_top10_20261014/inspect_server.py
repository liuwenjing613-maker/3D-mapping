from pathlib import Path
import json, hashlib
R=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
for p in [R/'instance_repair/frozen_e1_20261012/room2/reference.json',R/'instance_repair/configs/current_architecture.json']:
 if p.exists(): print(str(p),p.read_text()[:6000])
for p in (R/'instance_repair/frozen_e1_20261012').rglob('*.py'):
 text=p.read_text()
 lines=[l for l in text.splitlines() if 'p1a1_one_vote_20261009' in l or 'strict_baseline' in l]
 if lines: print(str(p), '\n'.join(lines)[:1200])
for p in (R/'instance_repair/frozen_e1_20261012').glob('*'):print('FROZEN',p.name)
P=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/room2')
for model in ['cropformer','sam2_1']:
 print(model,json.loads((P/model/'audit.json').read_text()))
