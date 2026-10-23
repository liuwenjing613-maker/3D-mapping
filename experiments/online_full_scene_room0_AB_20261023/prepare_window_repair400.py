"""Isolate the authorized window-only source policy full checkpoint400 repair.

All tracking/Gate/partial/E1/commit implementations are copied byte-for-byte.
Only execution roots and the exact-cache search roots are adapted.
"""
from pathlib import Path
import ast,hashlib,json,shutil

SRC=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_full_scene_room0_AB_20261023')
BASE=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
C=SRC/'window_only_repair400';O=BASE/'window_only_repair400'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def link(p,s):
 p=Path(p);s=Path(s);assert s.exists(),s;p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),p;p.symlink_to(s.resolve(),target_is_directory=s.is_dir())

def main():
 assert not O.exists(),'Use a fresh isolated result directory'
 C.mkdir(parents=True,exist_ok=True);O.mkdir()
 files=['common_continue.py','bootstrap_continue.py','scene_continue.py','selection_runtime.py','selection_qa.py','selection_fence.py',
  'track_prepare.py','track_queue.py','track_queue_with_cache.py','track_gate.py','track_batch.py','track_pure_gate.py','repair_continue.py','commit_continue.py']
 audit=[]
 for name in files:
  p=SRC/name;text=p.read_text();original=text
  if name=='common_continue.py':
   text=text.replace(str(SRC),str(C)).replace(str(BASE),str(O))
  elif name=='track_queue_with_cache.py':
   start=text.index('SOURCES=');end=text.index('\n\n',start)
   text=text[:start]+f"SOURCES=[Path({str(BASE)!r})/'400'/g/'tracking' for g in [GROUP,('B' if GROUP=='A' else 'A')]]+[ROOT.parent/('B' if GROUP=='A' else 'A')/'tracking']"+text[end:]
  ast.parse(text);(C/name).write_text(text)
  if name not in ['common_continue.py','track_queue_with_cache.py']:assert sha(C/name)==sha(p)
  audit.append(dict(file=name,source=str(p),source_sha256=sha(p),executed_sha256=sha(C/name),only_path_or_cache_adapter=text!=original))
 for p in (SRC/'selection_guard').glob('*.py'):
  q=C/'selection_guard'/p.name;q.parent.mkdir(exist_ok=True);shutil.copyfile(p,q);assert sha(q)==sha(p)
 for branch in ['A','B']:
  original=BASE/'400'/branch;root=O/'400'/branch;root.mkdir(parents=True)
  variant=BASE/'window_only_mask_ablation'/branch/'window_only'
  assert read(variant/'complete.json')['status']=='PASS'
  assert read(BASE/'window_only_mask_ablation'/branch/'control/complete.json')['control_reproduces_original_final_masks_and_stage_decisions']
  link(root/'mapping',original/'mapping')
  cand=root/'candidates';cand.mkdir()
  for p in variant.iterdir():
   if p.name in ['online_qa.json']:continue
   link(cand/p.name,p)
  for name in ['mapping_input_freeze.json','exact_model_cache_audit.json']:
   link(cand/name,original/'candidates'/name)
  # Read-only original RGB and model provenance, same dense 0..1995 input.
  shutil.copyfile(original/'inputs_complete.json',root/'inputs_complete.json')
  track=root/'tracking';track.mkdir();link(track/'inputs',original/'tracking/inputs')
  shutil.copyfile(original/'tracking/track_config.json',track/'track_config.json')
  tc=C/'tracking_code/400'/branch;tc.mkdir(parents=True)
  for p in (SRC/'tracking_code/400'/branch).iterdir():
   if not p.is_file():continue
   shutil.copyfile(p,tc/p.name);assert sha(p)==sha(tc/p.name)
   audit.append(dict(file=str((tc/p.name).relative_to(C)),source=str(p),source_sha256=sha(p),executed_sha256=sha(tc/p.name),only_path_or_cache_adapter=False))
  link(C/'selection_code/400'/branch,SRC/'selection_code/400'/branch)
  guard=read(original/'input_guard.json');guard['root']=str(root)
  more=[str(BASE/'400'/g) for g in ['A','B']]+[str(variant),str(root),str(O/'400'/('B' if branch=='A' else 'A')/'tracking')]
  guard['allowed_prefix_cache_roots']=sorted(set(guard['allowed_prefix_cache_roots']+more))
  guard['read_only_result_roots']=sorted(set(guard.get('read_only_result_roots',[])+[str(original),str(variant),str(original/'mapping')]))
  save(root/'input_guard.json',guard)
  paths=[original/'repair/commit_complete.json',original/'repair/repair_complete.json',original/'candidates/all_edge_20261021/v2/seed_manifest.json',
   original/'mapping/baseline/final/instance_surface.npz',original/'mapping/baseline/surface_evidence.npz',original/'mapping/geometry/surface.ply',original/'mapping/prior_commit.json',
   variant/'all_edge_20261021/v2/seed_manifest.json',variant/'semantic_route/semantic.json']
  commit=read(original/'repair/commit_complete.json');paths.append(Path(commit['final_map']))
  save(root/'run_input_freeze.json',dict(status='FROZEN',branch=branch,source_sha256={str(p):sha(p) for p in paths},
   mapping_origin=str(original/'mapping'),prior_repaired_checkpoint=300,current_mapping_checkpoint=400,
   candidate_source=str(variant),only_policy_change='cabinet SAM-only direct/replacement disabled; window route retained',
   PR_exemption_unchanged=True,GT_used=False,original_result_preserved=True))
 save(O/'adapter_audit.json',dict(status='PASS',files=audit,unchanged_numerical_files=[r for r in audit if not r['only_path_or_cache_adapter']]))
 save(O/'run_plan.json',dict(status='READY',branches=['A','B'],checkpoint=400,scope='Replay full repair from each original pre-repair400 checkpoint; earlier committed repairs unchanged',
  code=str(C),results=str(O),policy='window-only SAM source policy; cabinet ordinary selection',
  numeric_changes=[],strict_original_cache_fingerprint_required=True,tracking_misses_infer_normally=True,
  stages=['source/selection audit','exact-mask track prepare','dense SAM2 or exact cache','original Gate','partial + core + E1','durable repair commit','main default R2','full-resolution PLY comparison']))
 print(json.dumps(dict(status='READY',code=str(C),results=str(O))))
if __name__=='__main__':main()
