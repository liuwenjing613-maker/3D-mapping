"""Read existing frozen results; export a new provenance receipt without rerunning mapping."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
OUT=Path('/data/chenkejun/CVPR/results/partial_e1_swap_baseline_20261021')
ARCHIVE=Path('/home/chenkejun/CVPR/worktrees/auto-v3-swap')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
read=lambda p:json.loads(Path(p).read_text())
files={}; copies={}; started=time.time()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(p,expected=None):
 p=Path(p);key=str(p)
 if key not in files:files[key]={'sha256':digest(p),'bytes':p.stat().st_size}
 if expected is not None:assert files[key]['sha256']==expected,('hash mismatch',key)
 return files[key]
def copy(p,rel):
 p=Path(p);meta=check(p);dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 data=p.read_bytes()
 if dst.exists():assert dst.read_bytes()==data,str(dst)
 else:dst.write_bytes(data)
 copies[rel]={'source':str(p),**meta}
OUT.mkdir(parents=True,exist_ok=True)
run=subprocess.run([sys.executable,'-B','experiments/auto_v3_swap/verify.py','--server-sources'],cwd=ARCHIVE,text=True,capture_output=True)
assert run.returncode==0,(run.stdout,run.stderr)
upstream=json.loads(run.stdout);print('UPSTREAM',json.dumps(upstream),flush=True)
comp=read(R/'comparison.json');check(R/'comparison.json','110b1ab9436515df2749790f8b03c6081bc5a6bb4c3fea2bfacc4783401e46a5')
assert comp['status']=='PASS' and comp['new_total']==196 and comp['GT_count']==350
protocol=read(MAIN/'unified_eval/configs/current_protocol.json')
check(MAIN/'unified_eval/configs'/protocol['config_file'],protocol['config_sha256'])
assert protocol['frozen'] is False and protocol['profile_revision']==2
for name,expected in protocol['evaluator_code_sha256'].items():check(MAIN/'unified_eval'/name,expected)
KEYS=['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256','profile_semantics_sha256','evaluator_code_sha256']
scenes=[]
for s in comp['scenes']:
 scene=s['scene'];root=R/scene/'partial';freeze=read(root/'predictions_freeze.json')
 assert freeze['GT_read_by_mapper'] is False and freeze['baseline_modified'] is False
 checks={}
 for section in ['source_sha256','code_sha256','output_sha256']:
  entries=freeze[section]
  for path,expected in entries.items():check(path,expected)
  checks[section]=len(entries)
 for name in ['predictions_freeze.json','policy.json','input_selection_audit.json','tests_complete.json','mapping_complete.json','run_complete.json','evaluation_complete.json','frame_receipts.json']:
  if (root/name).exists():copy(root/name,f'evidence/{scene}/{name}')
 copy(root/'evaluation_r2/comparison.json',f'evidence/{scene}/evaluation/comparison.json')
 stages={}
 for stage in ['native','final']:
  lock=s['source_locks']['new_partial_'+stage];ev=Path(lock['path']);a=read(ev/'adapter_manifest.json')
  check(ev/'adapter_manifest.json',lock['manifest_sha256']);check(lock['source'],lock['surface_sha256'])
  baseline=s['source_locks']['baseline_'+stage];b=read(Path(baseline['path'])/'adapter_manifest.json')
  check(Path(baseline['path'])/'adapter_manifest.json',baseline['manifest_sha256'])
  check(baseline['source'],baseline['surface_sha256'])
  for key in KEYS:assert a[key]==b[key],(scene,stage,key)
  assert a['profile_config_sha256']==protocol['config_sha256']
  assert a['gt_scope_sha256']==protocol['canonical_GT'][scene]['gt_scope_sha256']
  for p in ev.iterdir():
   if p.is_file():
    check(p)
    if p.suffix=='.json':copy(p,f'evidence/{scene}/evaluation/{stage}/{p.name}')
  stages[stage]={'source':lock['source'],'source_sha256':lock['surface_sha256'],'evaluation':str(ev),'adapter_manifest_sha256':lock['manifest_sha256'],'alignment':{k:a[k] for k in KEYS},'metrics':next(x for x in s['rows'] if x['variant']=='new_partial' and x['stage']==stage)}
 gt=protocol['canonical_GT'][scene];check(gt['file'],gt['file_sha256'])
 scenes.append({'scene':scene,'selected_seeds':s['new_count'],'GT_count':s['GT_count'],'result_root':str(root),'resolved_result_root':str(root.resolve()),'freeze_checks':checks,'stages':stages})
 print('SCENE',scene,'PASS',checks,'unique_files',len(files),flush=True)
audit=read(R/'review/swap34/delivery_audit.json')
for rel,expected in audit['PLY_hashes'].items():
 if 'partial' in rel:
  path=R/'review'/rel
  check(path,expected)
receipt={'status':'PASS','created_unix':time.time(),'algorithm_frozen':True,'evaluation_frozen':False,'source_commit':'1103d8196a980874ef743f572d3b4c0748a4fdab','upstream_verification':upstream,'scenes':scenes,'summary':[x for x in comp['summary'] if x['variant']=='new_partial'],'files':files,'copies':copies,'seconds':time.time()-started}
(OUT/'server_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('DONE',json.dumps({'status':'PASS','files':len(files),'copies':len(copies),'seconds':receipt['seconds'],'receipt':str(OUT/'server_verification.json')}),flush=True)
