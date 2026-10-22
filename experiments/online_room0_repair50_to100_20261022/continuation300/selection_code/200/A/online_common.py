from pathlib import Path
import hashlib,json
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300/selection_code/200/A')
R=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/A/candidates')
FIDS=tuple(range(0,996,5))
CHECKPOINT=200
GROUP='A'
CANDIDATE_START=0
CURRENT_RAW=995
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
