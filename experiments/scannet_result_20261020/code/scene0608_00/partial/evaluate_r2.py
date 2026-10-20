from pathlib import Path
import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020')
from evaluate import evaluate_scene
evaluate_scene('scene0608_00', include_partial=True)
