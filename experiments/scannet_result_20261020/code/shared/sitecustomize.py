from pathlib import Path
import sys,json,os
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import LockedModules
F=MAIN/'instance_repair/frozen_e1_20261012/room0'
mapping=json.loads((F/'module_map.json').read_text())
mapping['revisable_instance_map.frame_io']='/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020/shared/frame_io.py'
mapping['run_repairs']='/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020/shared/run_repairs.py'
sys.meta_path.insert(0,LockedModules(F,mapping))
