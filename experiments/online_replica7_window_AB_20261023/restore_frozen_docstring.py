"""Undo a scene-name replacement in a frozen kernel's module docstring only."""
from pathlib import Path
import ast,json,hashlib
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
base=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/prefix100/v2/semantic_exemption.py')
src=base.read_text();reference=ast.parse(src);reference.body.pop(0);rows=[]
for scene in ['room1','room2','office0','office1','office2','office3','office4']:
 for p in (C/scene/'selection_code').glob('*/?/v2/semantic_exemption.py'):
  prior=p.read_text();tree=ast.parse(prior);tree.body.pop(0)
  assert ast.dump(tree)==ast.dump(reference)
  if prior==src:continue
  p.write_bytes(base.read_bytes());rows.append(str(p))
(O/'frozen_docstring_restore.json').write_text(json.dumps(dict(status='PASS',numeric_AST_exact=True,files=rows,source_sha256=hashlib.sha256(base.read_bytes()).hexdigest()),indent=2)+'\n')
print('FROZEN_DOCSTRING_RESTORED',len(rows))
