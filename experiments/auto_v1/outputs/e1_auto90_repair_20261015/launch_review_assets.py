from pathlib import Path
import os,json,subprocess
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015');R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
p=R/'review_assets_process.json';assert not p.exists()
with (R/'review_assets.log').open('xb') as f:q=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-u',str(C/'prepare_review_assets.py')],env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
p.write_text(json.dumps({'pid':q.pid}));print(p.read_text())
