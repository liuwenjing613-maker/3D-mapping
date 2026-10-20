"""Resolve current original E1 sources, without changing frozen numerical kernels."""
from pathlib import Path
import json,os,sys
C=Path(__file__).resolve().parent
SCENE='office3'
SOURCE=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020/office3/e1_repair')
SOURCE_CODE=Path('/home/chenkejun/CVPR/experiments/replica8_swap34_20261020/office3/e1_repair')
COUNT=len(json.loads((SOURCE/'confirmed_selection.json').read_text())['included'])
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020/office3/partial')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources,LockedModules
verify_sources()
F=MAIN/'instance_repair/frozen_e1_20261012'/('room2' if SCENE=='room2' else 'room0')
mapping=json.loads((F/'module_map.json').read_text())
mapping['run_history_core']=str(SOURCE_CODE/'run_history_core.py')
sys.meta_path.insert(0,LockedModules(F,mapping))
sys.path.insert(0,str(C))
