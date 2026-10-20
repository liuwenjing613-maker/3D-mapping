"""Resolve all numerical E1 dependencies to the verified main snapshot."""
from pathlib import Path
import json,sys
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office3/e1_repair')
S=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office3/v1')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources,LockedModules,load_script,digest
verify_sources()
F=MAIN/'instance_repair/frozen_e1_20261012/room0'
mapping=json.loads((F/'module_map.json').read_text())
for n in ('run_history_core','run_case','objectwise_ops','gate_v1','strict_vote_ops'):
 if (C/(n+'.py')).exists():mapping[n]=str(C/(n+'.py'))
sys.meta_path.insert(0,LockedModules(F,mapping))
sys.path.insert(0,str(C))
