"""User-requested source-policy ablation, fixed checkpoints and cached model inputs.

Numerical selection kernels stay identical. Only cabinet direct-route hits are
disabled. Legacy v1/v2 outputs are read-only feature caches/review metadata for
the production all_edge path, which selects directly from fresh grouping_r2.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ast,copy,hashlib,json,os,shutil,subprocess,time,traceback

REPO=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
SRC=REPO/'experiments/online_full_scene_room0_AB_20261023/selection_code/400'
CODE=REPO/'experiments/online_full_scene_room0_AB_20261023/window_only_mask_ablation'
BASE=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023/400')
OUT=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023/window_only_mask_ablation')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def link(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.symlink_to(Path(s).resolve(),target_is_directory=Path(s).is_dir())

def prepare(branch,variant):
 old=BASE/branch/'candidates';sc=SRC/branch;c=CODE/branch/variant;r=OUT/branch/variant
 assert not r.exists(),r
 r.mkdir(parents=True);c.mkdir(parents=True)
 # Inputs and legacy metadata are never written. Original model outputs are reused.
 for name in ['overmerge','baseline','catalog','frontend','intersection','versions','v1','v2']:
  link(r/name,old/name)
 changed=[]
 for folder in ['semantic_route','grouping_r2','all_edge']:
  for p in (sc/folder).glob('*'):
   if not p.is_file() or p.suffix not in ['.py','.json']:continue
   dest=c/folder/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
   text=p.read_text()
   if folder=='semantic_route':text=text.replace(str(old/'semantic_route'),str(r/'semantic_route'))
   elif folder=='grouping_r2':
    for stage in ['semantic_route','grouping_r2']:
     text=text.replace(str(old/stage),str(r/stage))
   elif folder=='all_edge':
    text=text.replace(str(old),str(r)).replace(str(sc),str(c))
   if variant=='window_only' and p.suffix=='.json':
    text=text.replace('window/cabinet SAM-only','window SAM-only; cabinet uses ordinary selection')
   if p.suffix=='.py':ast.parse(text)
   dest.write_text(text)
   changed.append(dict(path=str(dest),source=str(p),source_sha256=sha(p),adapted_sha256=sha(dest),path_adapted=text!=p.read_text()))
 # all_edge imports these byte-identical rules; run.py can consume new masks
 # while checking full RGB/model/postprocess/cache fingerprints.
 for name in ['v1','v2']:
  for p in (sc/name).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json']:
    dest=c/p.relative_to(sc);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    assert sha(dest)==sha(p)
 sem=read(old/'semantic_route/semantic.json');disabled=[]
 if variant=='window_only':
  original=copy.deepcopy(sem)
  for cid,row in sem['candidates'].items():
   if row['hit'] and row['target_family']=='cabinet':
    disabled.append(cid);row['hit_before_window_only']=True;row['hit']=False
    row['decision_reason']='CABINET_RETURNS_TO_ORDINARY_SELECTION'
  for cid,row in original['candidates'].items():
   if cid not in disabled:assert row==sem['candidates'][cid]
  sem['stats']['hit_cabinet']=0;sem['stats']['not_hit']+=len(disabled)
  sem['config']['source_route_ablation']=dict(direct_SAM_families=['window'],cabinet='ordinary CropFormer/SAM selection',classification_scores_unchanged=True)
 save(r/'semantic_route/semantic.json',sem)
 for sub in ['v2/semantic','all_edge_20261021/v2/semantic']:
  cache=old/sub/'semantic_cache'
  for p in cache.glob('*'):
   if p.suffix in ['.npz','.json']:link(r/'all_edge_20261021/v2/semantic/semantic_cache'/p.name,p)
 locked={str(p):sha(p) for p in [old/'semantic_route/semantic.json',old/'semantic_route/selection.json',old/'grouping_r2/grouping.json',old/'all_edge_20261021/v2/seed_manifest.json',old/'baseline/final/instance_surface.npz']}
 save(r/'experiment.json',dict(branch=branch,variant=variant,checkpoint=400,disabled_cabinet_hits=disabled,
  semantic_window_decisions_unchanged=True,SegFormer_PR_window_cabinet_exemption_unchanged=True,
  production_path='semantic_route -> grouping_r2 -> all_edge boundary/noise/grouping -> cached SegFormer -> original five-stage selection',
  legacy_v1_v2='read-only exact-mask morphology cache, semantic cache, and review metadata; not filter decisions',
  no_tracking_or_repair_run=True,no_GT_used=True,source_locks=locked,code_adapters=changed))
 return c,r,old,locked

def execute(branch,variant,gpu):
 c,r,old,locked=prepare(branch,variant);started=time.perf_counter();timings=[]
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
 steps=[('route',[c/'semantic_route/route.py']),('grouping_r2',[c/'grouping_r2/compute.py']),
  ('all_edge_prepare',[c/'all_edge/screen.py',400,'prepare']),
  ('semantic',[c/'v2/semantic/run.py','--manifest',r/'all_edge_20261021/v1/seed_manifest.json','--raw-config',SRC/branch/'config.json','--out',r/'all_edge_20261021/v2/semantic','--config',c/'v2/semantic/config.json']),
  ('all_edge_select',[c/'all_edge/screen.py',400,'select'])]
 for name,args in steps:
  save(r/'status.json',dict(status='RUNNING',stage=name,seconds=time.perf_counter()-started))
  tick=time.perf_counter()
  with (r/(name+'.log')).open('w') as log:
   subprocess.run([PY,'-B','-u']+list(map(str,args)),env=env,cwd=c,stdout=log,stderr=subprocess.STDOUT,check=True)
  timings.append(dict(stage=name,seconds=time.perf_counter()-tick))
 for p,h in locked.items():assert sha(p)==h,('source changed',p)
 sem=read(r/'all_edge_20261021/v2/semantic/classification.json')
 result=dict(status='PASS',branch=branch,variant=variant,seconds=time.perf_counter()-started,timings=timings,
  semantic_cache_hits=sem['stats']['cached_frames'],semantic_newly_inferred=sem['stats']['newly_inferred_frames'],
  final_retained=read(r/'all_edge_20261021/complete.json')['retained'],all_original_sources_unchanged=True)
 if variant=='control':
  for stage in ['all_edge_20261021/v1','all_edge_20261021/v2']:
   a=read(old/stage/'seed_manifest.json');b=read(r/stage/'seed_manifest.json')
   assert [(g['id'],g['primary']['digest'],g['primary']['support_sha256'],g['auxiliary_ids']) for g in a['groups']]==[(g['id'],g['primary']['digest'],g['primary']['support_sha256'],g['auxiliary_ids']) for g in b['groups']],stage
  assert read(old/'all_edge_20261021/v2/selection.json')['stages']==read(r/'all_edge_20261021/v2/selection.json')['stages']
  result['control_reproduces_original_final_masks_and_stage_decisions']=True
 save(r/'complete.json',result);save(r/'status.json',result);print(json.dumps(result),flush=True)
 return result

def branch_run(branch,gpu):
 try:
  return [execute(branch,v,gpu) for v in ['control','window_only']]
 except Exception:
  save(OUT/branch/'failure.json',dict(status='FAILED',error=traceback.format_exc()));raise

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(branch_run,b,gpu) for b,gpu in [('A',0),('B',1)]]
  result=[f.result() for f in fs]
 save(OUT/'complete.json',dict(status='PASS',branches=result))
