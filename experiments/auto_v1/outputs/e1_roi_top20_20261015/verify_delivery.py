from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent;W=H.parents[1]
D=W/'results/固定案例_三模型对比_20261006/room2_E1_prerepair_CUT_top20_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(D/'viewer_data.json');projection=read(D/'projection_annotations.json');a=read(D/'ranking_audit.json');s=read(D/'selection_setup.json')
assert len(v['cases'])==32 and len({c['uid'] for c in v['cases']})==32 and a['status']=='PASS'
assert len(v['crops'])==s['candidate_crop_count'] and sum(len(c['views']) for c in v['cases'])==180
assets=set()
for c in v['cases']:
 assert c['aliases']==[r['id'] for r in c['rankings']]
 assert len({w['number'] for w in c['views']})==len(c['views'])
 for view in c['views']:
  assets.update([view['full_rgb'],view['alarm']]);assert view['recommended'] in view['variants']
  for key in view['variants']:
   crop=v['crops'][key];assert projection['crops'][key]['group']==view['trigger_group']
   assert len(crop['models'])==3
for key,crop in v['crops'].items():
 assets.add(crop['RGB'])
 if 'comparison' in crop:assets.add(crop['comparison'])
 for model in crop['models']:
  assets.update([model[k] for k in ['color','boundary','ids_file']]);assert model['ids'].startswith('data:image/png;base64,')
for asset in assets:assert (D/asset).is_file(),asset
assert not s['map_modified'] and not s['tracking_started'] and s['human_records_created']==0
for model in ['cropformer','sam2_1']:assert read(D/'provenance'/model/'audit.json')['status']=='PASS'
digest={p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file() and p.name!='final_files_sha256.json'}
(D/'final_files_sha256.json').write_text(json.dumps(digest,ensure_ascii=False,indent=2),encoding='utf-8')
report=dict(status='READY_FOR_HUMAN_QUALITY_REVIEW',unique_ROIs=32,ranked_entries=60,groups={'CONFLICT':20,'U':20,'T':20},view_entries=180,crops=len(v['crops']),models_per_crop=3,referenced_assets_checked=len(assets),baseline_repaired=False,map_modified=False,GT_used=False,source_surface_sha256=a['source_surface_sha256'],download=read(H/'download_verification.json'),url='http://127.0.0.1:8785/room2_E1_prerepair_CUT_top20_20261015/index.html')
(H/'delivery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
