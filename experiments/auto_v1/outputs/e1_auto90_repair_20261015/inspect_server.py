from pathlib import Path
import json,subprocess
root=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
for p in [Path('/home/chenkejun/CVPR/AGENTS.md'),root/'AGENTS.md',root/'instance_repair/configs/current_architecture.json',root/'unified_eval/configs/current_protocol.json']:
    print('FILE',p)
    if p.exists():
        print('\n'.join(s for s in p.read_text().splitlines() if not any(k in s.lower() for k in ['password','密码','secret','token='])))
for folder in [root/'instance_repair',Path('/home/chenkejun/CVPR/experiments/mask_gate_no_foreign_v2_20261010'),Path('/home/chenkejun/CVPR/experiments/room2_strict_manual_ab_20261010')]:
    print('FILES',folder)
    print('\n'.join(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md'] and '__pycache__' not in str(p)))
print('GIT',subprocess.run(['git','status','--short'],cwd=root,text=True,capture_output=True).stdout)
print('GPU',subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],text=True,capture_output=True).stdout)
for folder in [Path('/data/chenkejun/CVPR/results/mask_gate_no_foreign_v2_20261010/room2'),Path('/data/chenkejun/CVPR/results/room2_strict_manual_ab_20261010')]:
    print('RESULT_FILES',folder)
    print('\n'.join(str(p.relative_to(folder)) for p in folder.glob('*')))
