from pathlib import Path
import hashlib,json
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023/office0/selection_code/400/A')
R=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023/office0/400/A/candidates')
FIDS=tuple(range(0,1996,5))
CHECKPOINT=400
GROUP='A'
CANDIDATE_START=0
CURRENT_RAW=1995
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
SAM='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
CROP='/data/chenkejun/CVPR/runtime/cropformer_env/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
