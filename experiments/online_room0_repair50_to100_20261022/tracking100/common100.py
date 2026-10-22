from pathlib import Path
import json,hashlib,os,time
GROUP=os.environ.get('CVPR_TRACK_GROUP','A')
assert GROUP in ('A','B')
C=Path(__file__).resolve().parent
BASEC=C.parent
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
TRACKROOT=TOP/'tracking100'
R=TRACKROOT/GROUP
B=TOP/'prefix100'
CANDIDATE_ROOT=TOP/('prefix100' if GROUP=='A' else 'prefix100_new50')
S=CANDIDATE_ROOT/'all_edge_20261021/v1'
CHOSEN=CANDIDATE_ROOT/'all_edge_20261021/v2/seed_manifest.json'
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
SAM='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
FIDS=list(range(0,496,5))
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
 row=dict(status='RUNNING',group=GROUP,stage=stage,unix=time.time(),**x)
 dump(R/'progress.json',row);print(json.dumps(row),flush=True)
