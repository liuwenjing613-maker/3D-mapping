"""Resolve current original E1 sources, without changing frozen numerical kernels."""
from pathlib import Path
import json,os,sys
C=Path(__file__).resolve().parent
SCENE=os.environ['PARTIAL_SCENE']
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
assert SCENE in SCENES
if SCENE=='room0':source_result=source_code='auto_v2_room0_pr35_20261019/e1_repair'
elif SCENE=='room2':source_result=source_code='e1_auto35_original_20261019'
else:source_result=source_code='auto_v2_replica6_20261019/'+SCENE+'/e1_repair'
SOURCE=Path('/data/chenkejun/CVPR/results')/source_result
SOURCE_CODE=Path('/home/chenkejun/CVPR/experiments')/source_code
COUNT=len(json.loads((SOURCE/'confirmed_selection.json').read_text())['included'])
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica8_partial_20261019')/SCENE
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources,LockedModules
verify_sources()
F=MAIN/'instance_repair/frozen_e1_20261012'/('room2' if SCENE=='room2' else 'room0')
mapping=json.loads((F/'module_map.json').read_text())
mapping['run_history_core']=str(SOURCE_CODE/'run_history_core.py')
sys.meta_path.insert(0,LockedModules(F,mapping))
sys.path.insert(0,str(C))
