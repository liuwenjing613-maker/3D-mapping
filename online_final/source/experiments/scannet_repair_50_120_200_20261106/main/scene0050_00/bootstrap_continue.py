import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
import write_fence
from common_continue import *
import sys
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources,LockedModules,load_script
verify_sources()
F=MAIN/('instance_repair/frozen_e1_20261012/'+'room0')
mapping=read(F/'module_map.json')
for n in ('run_case','objectwise_ops','gate_v1','strict_vote_ops'):
 assert (C/(n+'.py')).exists(),str(C/(n+'.py'))
 mapping[n]=str(C/(n+'.py'))
# Only the camera input reader is dataset-specific; all other frozen modules retain their source locks.
mapping['revisable_instance_map.frame_io']='/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106/mapping_snapshot/revisable_instance_map/src/revisable_instance_map/frame_io.py'
sys.meta_path.insert(0,LockedModules(F,mapping))
sys.path.insert(0,str(C))
sys.path.insert(0,str(CONT))
import run_repairs as r
from scene_continue import SceneContinue
r.Scene=SceneContinue

from aligned_surface import assign_surface as aligned_assign_surface
r.assign_surface=aligned_assign_surface
