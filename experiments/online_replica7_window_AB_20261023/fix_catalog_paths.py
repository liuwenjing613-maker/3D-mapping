"""Correct generated per-scene catalog references; preserve all numeric outputs."""
from pathlib import Path
import ast,json,hashlib,time
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
rows=[]
for scene in ['room1','room2','office0','office1','office2','office3','office4']:
 for p in (C/scene/'selection_code').rglob('*.py'):
  old=p.read_text();new=old.replace('/candidates/catalog/room0','/candidates/catalog/'+scene)
  if old==new:continue
  ast.parse(new);p.write_text(new)
  rows.append(dict(file=str(p),before=hashlib.sha256(old.encode()).hexdigest(),after=hashlib.sha256(new.encode()).hexdigest()))
(O/'catalog_path_adapter_fix.json').write_text(json.dumps(dict(status='PASS',unix=time.time(),files=rows),indent=2)+'\n')
print('CATALOG_PATHS_FIXED',len(rows))
