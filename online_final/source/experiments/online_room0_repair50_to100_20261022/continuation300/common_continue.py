"""Authorized causal continuation of the original A/B checkpoint strategy."""
from pathlib import Path
import json,hashlib,os,time
CHECKPOINT=int(os.environ.get('CVPR_CHECKPOINT','200'))
GROUP=os.environ.get('CVPR_TRACK_GROUP','A')
assert CHECKPOINT in (200,300) and GROUP in ('A','B')
OLD_COUNT=CHECKPOINT-100
CURRENT_RAW=(CHECKPOINT-1)*5
RAW_COUNT=CURRENT_RAW+1
BASEC=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
CONT=BASEC/'continuation300'
C=CONT/'tracking_code'/str(CHECKPOINT)/GROUP
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
ROOT=TOP/'continuation300'/str(CHECKPOINT)/GROUP
TRACKROOT=ROOT
R=ROOT/'tracking'
B=ROOT/'mapping'
CONFIG=B/'config.json'
CANDIDATE_ROOT=ROOT/'candidates'
S=CANDIDATE_ROOT/'all_edge_20261021/v1'
CHOSEN=CANDIDATE_ROOT/'all_edge_20261021/v2/seed_manifest.json'
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
SAM='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
CROP='/data/chenkejun/CVPR/runtime/cropformer_env/bin/python'
FIDS=list(range(0,RAW_COUNT,5))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def emit(stage,**x):
 row=dict(status='RUNNING',group=GROUP,checkpoint=CHECKPOINT,stage=stage,unix=time.time(),**x)
 dump(R/'progress.json',row);print(json.dumps(row),flush=True)
