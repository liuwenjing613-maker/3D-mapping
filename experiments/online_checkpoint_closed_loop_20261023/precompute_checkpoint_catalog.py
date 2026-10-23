"""Independently prepare the other branch's ROI while the first runs later stages."""
from pathlib import Path
import os,subprocess,sys
HERE=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_checkpoint_closed_loop_20261023')
code=HERE/'checkpoint'
os.environ.update(CVPR_CHECKPOINT='200',CVPR_TRACK_GROUP='A',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
subprocess.run([PY,'-B',str(code/'selection_setup.py')],cwd=code,check=True)
sys.path.insert(0,str(code))
from selection_runtime import run
run('input_fence',code/'selection_fence.py')
run('catalog',code/'selection_code/200/A/catalog/build_candidates.py',['room0'])
