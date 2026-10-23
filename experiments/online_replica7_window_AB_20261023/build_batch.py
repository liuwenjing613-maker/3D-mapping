"""Scene/checkpoint/path adapters for the user-authorized seven-scene online run.
Numerical kernels and thresholds are unchanged; first checkpoint starts empty.
"""
from pathlib import Path
import ast,hashlib,json,re,runpy,shutil
REPO=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
SRC=REPO/'experiments/online_full_scene_room0_AB_20261023'
HERE=REPO/'experiments/online_replica7_window_AB_20261023'
OUT=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
SCENES=['room1','room2','office0','office1','office2','office3','office4']
COUNTS=[50,100,200,300,400];PREV={50:0,100:50,200:100,300:200,400:300}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def replace(s,a,b):assert a in s,a;return s.replace(a,b)
def scene_names(s,scene):
 # Absolute references to existing frozen source modules remain references.
 protected={}
 def keep(m):
  key=f'__UNCHANGED_SOURCE_PATH_{len(protected)}__';protected[key]=m.group(0);return key
 s=re.sub(r'''(['"])/(?:home|data)/[^'"\n]*\1''',keep,s)
 s=s.replace('room0',scene)
 for key,v in protected.items():s=s.replace(key,v)
 return s
def put(p,s):
 if p.suffix=='.py':ast.parse(s)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)

def main():
 assert not (OUT/'run_plan.json').exists(),'Prepared batch is immutable; use an explicit repair for adaptation errors'
 OUT.mkdir(parents=True,exist_ok=True);audit=[]
 for scene in SCENES:
  c=HERE/scene;o=OUT/scene;c.mkdir(parents=True,exist_ok=True)
  names=['common_continue.py','map_support.py','map_continue.py','bootstrap_continue.py','scene_continue.py','selection_setup.py','selection_runtime.py',
   'selection_run.py','selection_cache.py','selection_qa.py','selection_fence.py','track_prepare.py','track_queue.py','track_gate.py',
   'track_batch.py','track_pure_gate.py','repair_continue.py','commit_continue.py']
  for name in names:
   source=SRC/name;s=source.read_text()
   if name=='common_continue.py':
    s=replace(s,'import json,hashlib,os,time\n','import json,hashlib,os,time,threading\n')
    s=replace(s,"p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')","p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')")
    s=replace(s,'CHECKPOINT in (200,300,400)','CHECKPOINT in (50,100,200,300,400)')
    s=replace(s,'OLD_COUNT=CHECKPOINT-100','OLD_COUNT={50:0,100:50,200:100,300:200,400:300}[CHECKPOINT]')
    s=s.replace(str(SRC),str(c)).replace('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023',str(o))
    s+='\nSCENE='+repr(scene)+'\n'
   elif name=='map_support.py':
    s=s.replace(str(SRC),str(c));s=replace(s,'TARGET in (200,300,400)','TARGET in (50,100,200,300,400)')
    s=replace(s,'PRIOR_COUNT=TARGET-100','PRIOR_COUNT={50:0,100:50,200:100,300:200,400:300}[TARGET]')
    s=replace(s,'PRIOR_RAW=(PRIOR_COUNT-1)*5','PRIOR_RAW=max(-1,(PRIOR_COUNT-1)*5)')
    start=s.index('def prior_paths():');end=s.index('\ndef setup():',start)
    s=s[:start]+f'''def prior_paths():
 out=Path({str(o)!r})
 if PRIOR_COUNT==0:return out/'initial'/GROUP/'initial_complete.json',out/'initial'/GROUP
 return out/str(PRIOR_COUNT)/GROUP/'repair/commit_complete.json',out/str(PRIOR_COUNT)/GROUP/'mapping'
'''+s[end:]
    cfg=f'/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs/{scene}.json'
    s=replace(s,"config=read(BASEC/'prefix100/config.json')",f'config=read(Path({cfg!r}))')
    s=s.replace('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs/room0.json',cfg)
    s=s.replace('/data/chenkejun/CVPR/datasets/cropformer_frozen/room0',f'/data/chenkejun/CVPR/datasets/cropformer_frozen/{scene}')
    s=s.replace("str(TOP/'prefix100')", "str(previous.resolve())")
    s=replace(s,"config['frame_selection']['stop_exclusive']=PRIOR_RAW+1","config['frame_selection']['stop_exclusive']=PRIOR_RAW+1")
   elif name=='map_continue.py':
    start=s.index(' geometry.grid=');end=s.index(" manifest=read(prior['journal_manifest'])",start)
    block=s[start:end];s=s[:start]+' if PRIOR_COUNT:\n'+''.join(' '+line+'\n' for line in block.splitlines())+' else:\n  grid_restore_error=0.;geometry_duplicate_points=0\n'+s[end:]
    s=replace(s,"max(x['frame_id'] for x in decisions)","max((x['frame_id'] for x in decisions),default=-1)")
    s=replace(s,"source=ReplicaFrameSource(R/'config.json');assert len(source.frame_ids)==PRIOR_COUNT\n frames=[source.load_frame(fid) for fid in FIDS[:PRIOR_COUNT]]", "frames=[]\n if PRIOR_COUNT:\n  source=ReplicaFrameSource(R/'config.json');assert len(source.frame_ids)==PRIOR_COUNT\n  frames=[source.load_frame(fid) for fid in FIDS[:PRIOR_COUNT]]")
    s=s.replace('prior_repair_applied=True','prior_repair_applied=PRIOR_COUNT>0').replace('repaired_state_used=True','repaired_state_used=PRIOR_COUNT>0,initial_empty_state=PRIOR_COUNT==0')
    s=s.replace('future_baseline_used=False,repair_applied=True','future_baseline_used=False,repair_applied=PRIOR_COUNT>0')
   elif name=='commit_continue.py':s=replace(s,'PRIOR_RAW = (OLD_COUNT - 1) * 5','PRIOR_RAW = max(-1,(OLD_COUNT - 1) * 5)')
   elif name=='bootstrap_continue.py':
    # The frozen room0 module package contains the shared numerical kernels.
    s=s.replace("F=MAIN/'instance_repair/frozen_e1_20261012/room0'", "F=MAIN/('instance_repair/frozen_e1_20261012/'+'__FROZEN_PACKAGE__')")
   elif name=='selection_setup.py':
    start=s.index(' caches=list(');end=s.index(' allowed=set(',start)
    s=s[:start]+f''' caches=list(old['allowed_prefix_cache_roots'])+[str(ROOT),str(B.resolve()),str(TOP/'tracking100'),str(TOP/'prefix100')]
 for n in {COUNTS!r}:
  if n>CHECKPOINT:continue
  for g in ['A','B']:
   peer=Path({str(o)!r})/str(n)/g
   if peer!=ROOT:caches.extend(str(peer/k) for k in ['candidates','tracking'])
'''+s[end:]
   elif name=='selection_runtime.py':
    s=replace(s,"if int(free)>=14000", "if int(index) in set(map(int,os.environ.get('CVPR_GPU_POOL','0,1,2,3').split(','))) and int(free)>=14000")
   elif name=='selection_run.py':
    s=replace(s,"('semantic_route',SC/'semantic_route/route.py'", "('window_source_policy',CONT/'window_source_policy.py',[],PY,False,CANDIDATE_ROOT/'window_source_policy.json'),\n   ('semantic_route',SC/'semantic_route/route.py'")
    s=s.replace("CONT/'track_queue_with_cache.py'","CONT/'track_queue.py'")
   elif name=='selection_cache.py':
    start=s.index('def source_pairs():');end=s.index('\ndef link(',start)
    s=s[:start]+f'''def source_pairs():
 result=[]
 for n in {COUNTS!r}:
  if n>CHECKPOINT:continue
  for g in ['A','B']:
   root=Path({str(o)!r})/str(n)/g/'candidates'
   if root!=CANDIDATE_ROOT:result.append((CONT/'selection_code'/str(n)/g,root))
 return result
'''+s[end:]
   elif name=='track_queue.py':
    s=replace(s,"peer=TOP/'continuation300'/str(CHECKPOINT)/('B' if GROUP=='A' else 'A')/'tracking'","peer=ROOT.parent/('B' if GROUP=='A' else 'A')/'tracking'")
   elif name=='selection_qa.py':
    s=replace(s,"assert cat['repair100_applied'] and not cat['current_checkpoint_repaired']", "assert cat['repair100_applied']==(OLD_COUNT>=100) and cat['repair50_applied']==(OLD_COUNT>=50) and not cat['current_checkpoint_repaired']")
   elif name=='repair_continue.py':
    s=s.replace("BASEC/'run_history_core.py'","CONT/'helper_code/run_history_core.py'").replace("BASEC/'observation_ops.py'","CONT/'helper_code/observation_ops.py'")
    s=s.replace("BASEC/'expand_room0.py'","CONT/'helper_code/expand_kernel.py'").replace('e.C=BASEC',"e.C=CONT/'helper_code'")
   if name not in ['common_continue.py','map_support.py','selection_runtime.py','track_queue.py','track_pure_gate.py']:
    s=scene_names(s,scene)
   if name=='bootstrap_continue.py':s=s.replace('__FROZEN_PACKAGE__','room0')
   # Historical marker names are metadata; state truth follows the actual previous checkpoint.
   if name in ['track_prepare.py','track_gate.py']:
    s=s.replace('repair50_applied=True','repair50_applied=OLD_COUNT>=50').replace('repair100_applied=True','repair100_applied=OLD_COUNT>=100')
   put(c/name,s);audit.append(dict(scene=scene,file=name,source=str(source),source_sha256=sha(source),adapted_sha256=sha(c/name)))
  for sub in ['map_guard','selection_guard']:
   for p in (SRC/sub).glob('*.py'):put(c/sub/p.name,p.read_text());assert sha(p)==sha(c/sub/p.name)
  for name in ['initialize_empty.py','window_source_policy.py','run_branch.py']:shutil.copyfile(HERE/name,c/name)
  for name in ['run_history_core.py','observation_ops.py','adaptive_core.py','expand_room0.py']:
   s=scene_names((BASE/name).read_text(),scene)
   if name=='expand_room0.py':
    # Provenance refers to the actual adapted copy; formulas are unchanged.
    s=s.replace(f"'expand_{scene}.py'","'expand_kernel.py'")
   put(c/'helper_code'/('expand_kernel.py' if name=='expand_room0.py' else name),s)
  for n in COUNTS:
   for g in ['A','B']:
    tc=c/'tracking_code'/str(n)/g;tc.mkdir(parents=True,exist_ok=True)
    for p in (SRC/'tracking_code/400'/g).iterdir():
     if p.is_file():shutil.copyfile(p,tc/p.name)
  # Reuse the accepted candidate adapter generator, extending its frame bounds.
  gen=(SRC/'selection_build.py').read_text().replace(str(SRC),str(c)).replace('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023',str(o))
  gen=replace(gen,'for n in [200,300,400]:',f'for n in {COUNTS!r}:')
  gen=replace(gen,"code=CONT/'selection_code'/str(n)/g;out=TOP/str(n)/g/'candidates'", "old={50:0,100:50,200:100,300:200,400:300}[n]\n   code=CONT/'selection_code'/str(n)/g;out=TOP/str(n)/g/'candidates'")
  gen=gen.replace('(n-100)*5','old*5').replace('{n-100}','{old}').replace("{n if g=='A' else 100}","{n if g=='A' else n-old}")
  gen=gen.replace("if dest.exists():assert dest.read_text()==s,('nonidentical rerun',str(dest))\n    else:dest.write_text(s)","dest.write_text(s)")
  put(c/'selection_build.py',gen);runpy.run_path(str(c/'selection_build.py'),run_name='__main__')
  for n in COUNTS:
   for g in ['A','B']:
    sc=c/'selection_code'/str(n)/g;old=PREV[n]
    for p in sc.rglob('*'):
     if not p.is_file() or p.suffix not in ['.py','.json','.js','.html','.css']:continue
     if str(p.relative_to(sc)) in ['v2/filter_rules.py','v2/semantic_exemption.py','v1/noise_filter.py','v1/crop_filter.py','v1/source_policy.py']:
      p.write_bytes((BASE/'prefix100'/p.relative_to(sc)).read_bytes());continue
     s=scene_names(p.read_text(),scene)
     s=s.replace('/candidates/catalog/room0','/candidates/catalog/'+scene)
     s=s.replace("catalog['repair50_applied'] and catalog['repair100_applied'] and not catalog['current_checkpoint_repaired']",f"catalog['previous_committed_checkpoint']=={old} and not catalog['current_checkpoint_repaired']")
     s=s.replace("'repair50_applied':True",f"'repair50_applied':{old>=50}").replace('repair50_applied=True',f'repair50_applied={old>=50}').replace("['repair50_applied']=True",f"['repair50_applied']={old>=50}")
     s=s.replace("'repair100_applied':True",f"'repair100_applied':{old>=100}").replace('repair100_applied=True',f'repair100_applied={old>=100}').replace("['repair100_applied']=True",f"['repair100_applied']={old>=100}")
     # No room0 hand-picked source overrides are transferred to another scene.
     if p.name=='source_overrides.json':
      d=json.loads(s);assert not d.get('overrides'),d
     put(p,s)
  assert sha(c/'track_pure_gate.py')==sha(SRC/'track_pure_gate.py')
  for n in COUNTS:
   for g in ['A','B']:
    for name in ['track_case.py','gate_v1.py','objectwise_ops.py','strict_vote_ops.py','joint_batch_ops.py','tracking_policy.json','gate_policy.json']:
     assert sha(c/'tracking_code'/str(n)/g/name)==sha(SRC/'tracking_code/400'/g/name)
 save(OUT/'adapter_audit.json',dict(status='PASS',adapters=audit,scope='scene/path/checkpoint/zero-state adapters; authorized window-only source routing',
  frozen_SAM_Gate_objectwise_vote_policy_files_exact=True,prior_room0_outputs_used_as_history=False))
 save(OUT/'run_plan.json',dict(status='READY',scenes=SCENES,branches=['A','B'],checkpoints=COUNTS,processed_frames=400,raw_mapping_ids=list(range(0,2000,5)),
  policy='window-only SAM source route; cabinet ordinary candidate selection; original PR exemptions',surface_policy='checkpoint',
  initialization='zero frames, empty TSDF and association; no inherited room0 history',A='all arrived mapping frames',B='new mapping frames since prior checkpoint',
  GT_used_by_mapper=False,evaluation='main current object_observed_repair revision2, final400 only',GPU_pool=[0,1,2,3]))
 print('BATCH_ADAPTERS_READY',SCENES)
if __name__=='__main__':main()
