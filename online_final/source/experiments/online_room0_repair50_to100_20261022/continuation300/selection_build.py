"""Build isolated candidate adapters; original checkpoint-100 files remain immutable."""
from pathlib import Path
import json,hashlib,ast,re
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
CONT=BASE/'continuation300';TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
SRC=BASE/'prefix100';SR=TOP/'prefix100'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_required(text,a,b):
 assert a in text,a
 return text.replace(a,b)
def main():
 audit=[]
 dirs=['catalog','frontend','overmerge','intersection','versions','semantic_route','grouping_r2','grouping_r3','v1','v2','all_edge']
 for n in [200,300]:
  for g in ['A','B']:
   code=CONT/'selection_code'/str(n)/g;out=TOP/'continuation300'/str(n)/g/'candidates'
   raw=(n-1)*5;rawcount=raw+1;start=0 if g=='A' else (n-100)*5;first=start//5+1
   policy=f'all_observed_mapping_frames_1_to{n}' if g=='A' else f'new_mapping_frames_{first}_to{n}'
   code.mkdir(parents=True,exist_ok=True)
   sources=[p for d in dirs for p in (SRC/d).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.css','.js','.html'] and '__pycache__' not in p.parts]
   sources += [SRC/'containment_approval.json']
   source_hashes={str(p):sha(p) for p in sources}
   for p in sources:
    rel=p.relative_to(SRC);s=p.read_text()
    s=s.replace(str(SRC),str(code)).replace(str(SR),str(out))
    s=s.replace('geometry_prefix100.json',f'geometry_prefix{n}.json')
    if rel==Path('catalog/build_candidates.py'):
     s=replace_required(s,'list(range(0,496,5))',f'list(range(0,{rawcount},5))')
     s=replace_required(s,'fids=np.arange(0,len(poses),5,dtype=int)',f'fids=np.arange({start},{rawcount},5,dtype=int);assert len(poses)=={rawcount}')
     s=replace_required(s,'for fid in range(0,496,5):',f'for fid in range({start},{rawcount},5):')
     s=replace_required(s,"e['all_frames_checked']=100",f"e['all_frames_checked']={n if g=='A' else 100}")
     s=s.replace('前100个处理帧中',f'候选来源处理帧{first}至{n}中')
     s=s.replace('after committed repair50 and continuation to100',f'after committed checkpoint{n-100} branch {g}, continuation to{n}')
     s=replace_required(s,"'history_frames':100",f"'history_frames':{n},'checkpoint':{n},'branch':'{g}','candidate_source_policy':'{policy}','candidate_processed_frame_range':[{first},{n}],'candidate_raw_frame_range':[{start},{raw}],'candidate_search_frame_ids':list(range({start},{rawcount},5)),'scoring_history_frame_ids':list(range(0,{rawcount},5))")
    if rel==Path('frontend/prepare_inputs.py'):
     s=s.replace("catalog['repair50_applied'] and not catalog['repair100_applied']","catalog['repair50_applied'] and catalog['repair100_applied'] and not catalog['current_checkpoint_repaired']")
     # The historical implementation asserts only max prefix here; select views are separately constrained.
     s=s.replace("all(v['frame']<=495 for v in e['views'])",f"all(v['frame'] in range({start},{rawcount},5) for v in e['views'])")
     s=s.replace("all(v['frame'] in range(0,496,5) for v in e['views'])",f"all(v['frame'] in range({start},{rawcount},5) for v in e['views'])")
     s=s.replace("history_frames=100",f"history_frames={n}")
     s=re.sub(r"NAME='[^']+'",f"NAME='room0-prefix{n}-{g}-after-repair{n-100}-online-ROI'",s)
    if rel==Path('all_edge/screen.py'):
     s=replace_required(s,'assert n==100',f'assert n=={n}')
     s=s.replace("'all_observed_mapping_frames_1_to100'",repr(policy))
     s=s.replace('candidate_raw_frame_range=[0,495]',f'candidate_raw_frame_range=[{start},{raw}]')
     s=s.replace('candidate_processed_frame_range=[1,100]',f'candidate_processed_frame_range=[{first},{n}]')
     s=s.replace("selection['candidate_raw_frame_range']=[0,495]",f"selection['candidate_raw_frame_range']=[{start},{raw}]")
     s=s.replace("selection['candidate_processed_frame_range']=[1,100]",f"selection['candidate_processed_frame_range']=[{first},{n}]")
     s=s.replace("selection['repair100_applied']=False",f"selection['repair100_applied']=True;selection['current_checkpoint_repaired']=False;selection['previous_committed_checkpoint']={n-100};selection['scoring_history']=[1,{n}]")
     s=s.replace("assert 0<=s['frame']<=n*5-5",f"assert {start}<=s['frame']<=n*5-5")
     s=s.replace("started=time.time()", "started=time.perf_counter()").replace("seconds=time.time()-started","seconds=time.perf_counter()-started")
    if rel==Path('v2/measure_and_select.py'):
     s=re.sub(r"CASE='[^']+'",f"CASE='room0_online_prefix{n}_{g}_20261022'",s)
    # Metadata only: prior checkpoint100 is committed; this trigger not repaired yet.
    s=s.replace("'repair100_applied':False",f"'repair100_applied':True,'current_checkpoint_repaired':False,'previous_committed_checkpoint':{n-100}")
    s=s.replace('repair100_applied=False',f'repair100_applied=True,current_checkpoint_repaired=False,previous_committed_checkpoint={n-100}')
    if p.suffix=='.json':
     d=json.loads(s)
     if rel==Path('containment_approval.json'):
      d.update(case_id=f'room0_online_prefix{n}_{g}_20261022',basis='User confirmed continuation of original dual95 selection, direct containment, and first full/crop border exclusion',scope=f'branch{g} checkpoint{n}; candidate frames{first}..{n}; scoring1..{n}')
     s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
    if p.suffix=='.py':ast.parse(s)
    dest=code/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_text()==s,('nonidentical rerun',str(dest))
    else:dest.write_text(s)
   online=f"""from pathlib import Path
import hashlib,json
C=Path({str(code)!r})
R=Path({str(out)!r})
FIDS=tuple(range(0,{rawcount},5))
CHECKPOINT={n}
GROUP={g!r}
CANDIDATE_START={start}
CURRENT_RAW={raw}
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
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\\n');t.replace(p)
"""
   (code/'online_common.py').write_text(online)
   audit.append(dict(checkpoint=n,group=g,code=str(code),results=str(out),candidate_raw_range=[start,raw],scoring_history=[1,n],original_source_sha256=source_hashes,adapter_sha256={str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()}))
 guard=(SRC/'guard/sitecustomize.py').read_text().replace('current_cutoff = 495',"current_cutoff = (int(os.environ.get('CVPR_CHECKPOINT','200'))-1)*5")
 guard=guard.replace("settings = json.load(stream)","settings = json.load(stream)\n    current_cutoff = int(settings['raw_cutoff'])")
 guard=guard.replace('(?<![A-Za-z0-9])(?:frame|depth|f)(\\d{6})(?!\\d)','(?<![A-Za-z0-9])(?:frame|depth|f)(\\d{6})(?=$|[._-])')
 dest=CONT/'selection_guard/sitecustomize.py';dest.parent.mkdir(exist_ok=True);dest.write_text(guard)
 (CONT/'selection_adapter_audit.json').write_text(json.dumps(dict(status='PASS',rules='original dual95 and existing stage5 exemptions unchanged',adapters=audit),indent=2))
 print('BUILT',len(audit),'candidate code adapters')
if __name__=='__main__':main()
