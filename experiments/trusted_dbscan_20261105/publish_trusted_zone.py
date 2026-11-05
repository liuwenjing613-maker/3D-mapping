from pathlib import Path
import json,hashlib,shutil,ast,subprocess,re,collections
W=Path('/home/chenkejun/CVPR/worktrees/trusted-zone-20261105');HOME=Path('/home/chenkejun/CVPR');DOC=W/'docs/trusted_zone_20261105';DOC.mkdir(parents=True,exist_ok=True)
manifest={};locks=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def copy(src,rel=None,expected=None):
 src=Path(src);rel=Path(rel) if rel else src.relative_to(HOME);dst=W/rel
 assert dst.resolve().is_relative_to(W.resolve()),str(dst)
 b=src.read_bytes()
 if expected:assert sha(b)==expected,str(src)
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.read_bytes()==b,'Base source mismatch: '+str(rel)
 else:dst.write_bytes(b)
 manifest[str(rel)]={'source':str(src),'sha256':sha(b),'bytes':len(b)}
for name in ['trusted_full_online_20261104','trusted_dbscan_20261105']:
 C=HOME/'experiments'/name;lock=json.loads((C/'source_lock.json').read_text());locks.append((name,len(lock['sha256'])))
 for p,h in lock['sha256'].items():copy(p,expected=h)
 for p in C.iterdir():
  if p.is_file() and p.suffix in ['.py','.json']:copy(p)
 for p in (C/'viewer').rglob('*'):
  if p.is_file() and p.suffix in ['.html','.js','.json']:copy(p)
# Preserve baseline lock used by the DBSCAN wrapper, including declared dependencies.
C=HOME/'experiments/unified_full_depth_20261101'
for n in ['source_lock.json','alignment_common.py','finish_scene.py']:
 copy(C/n)
# Only small, explicitly selected numerical reports; no masks, GT, PLY, weights or logs.
R=Path('/data/chenkejun/CVPR/results/trusted_dbscan_20261105')
selected=['comparison/comparison.json','comparison/per_gt.csv','comparison/original_severe_cases.csv','comparison/new_severe_vs_old_dbscan.csv','comparison/input_audit.json','diagnostics/severe_case_postprocess_audit.json','diagnostics/metric_paradox/analysis.json','diagnostics/metric_paradox/validation.json','control_complete.json','candidate_complete.json','final_source_audit.json','plan.json']
for f in selected:copy(R/f,Path('docs/trusted_zone_20261105/evidence')/f)
for s in ['room0','room1','room2','office0','office1','office2','office3','office4']:
 for mode in ['control','candidate']:copy(R/mode/s/'final_scene_complete.json',Path('docs/trusted_zone_20261105/evidence/receipts')/mode/s/'final_scene_complete.json')
T=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
copy(T/'plan.json','docs/trusted_zone_20261105/evidence/online_plan.json')
copy(T/'comparison/comparison.json','docs/trusted_zone_20261105/evidence/raw_online_comparison.json')
# Avoid newline normalization: locks describe deployed bytes.
for n in ['trusted_full_online_20261104','trusted_dbscan_20261105','unified_full_depth_20261101']:
 p=W/'experiments'/n/'.gitattributes'
 if not p.exists():p.write_text('* -text\n')
(DOC/'source_manifest.json').write_text(json.dumps({'status':'PASS','base_branch':'origin/online-v2','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),'source_locks':locks,'files':manifest},ensure_ascii=False,indent=2)+'\n')
print('COPIED',len(manifest),'bytes',sum(r['bytes'] for r in manifest.values()))
