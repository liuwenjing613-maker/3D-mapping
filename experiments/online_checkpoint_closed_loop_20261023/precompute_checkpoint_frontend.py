"""Prepare independent optimized-branch RGB jobs while reference repair is running."""
from pathlib import Path
import os,sys
code=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_checkpoint_closed_loop_20261023/checkpoint')
os.environ.update(CVPR_CHECKPOINT='200',CVPR_TRACK_GROUP='A',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
sys.path.insert(0,str(code))
from selection_runtime import run
run('prepare',code/'selection_code/200/A/frontend/prepare_inputs.py')
run('exact_model_cache',code/'selection_cache.py')
