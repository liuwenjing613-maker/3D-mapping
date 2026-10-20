from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent;W=H.parents[1];WEB=W/'results/固定案例_三模型对比_20261006'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
old=read(H/'old_annotations_hashes.json')
for p,h in old.items():assert sha(Path(p))==h,p
summary=dict(status='READY_FOR_HUMAN_ANNOTATION',baseline='P1-A1-strict-vote unrepaired',evaluation_protocol='object_observed_repair / revision 2',GT_used=False,map_modified=False,old_annotations_unchanged=True,old_annotation_files_checked=len(old),ranking_audit=read(H/'catalog_audit.json'),scenes={})
for s in ['room0','room2']:
 d=WEB/(s+'_strict_vote_annotation_20261010');setup=read(d/'selection_setup.json');v=read(d/'viewer_data.json');projection=read(d/'projection_annotations.json')
 assert setup['human_records_created']==0 and setup['previous_reference_count']==0
 assert setup['baseline_method']=='P1-A1-strict-vote' and not setup['baseline_repaired']
 files=read(d/'files_sha256.json')
 for rel in ['index.html','plan.txt','selection_setup.json','files_sha256.json']:files[rel]=sha(d/rel)
 # Every referenced asset must be part of the verified archive, including inline-ID source PNGs.
 for case in v['cases']:
  for view in case['views']:
   for k in ['full_rgb','alarm']:assert view[k] in files
   assert view['recommended'] in v['crops']
 for key,crop in v['crops'].items():
  assert crop['RGB'] in files
  for m in crop['models']:
   for k in ['color','boundary','ids_file']:assert m[k] in files
  assert key in projection['crops']
 (d/'final_files_sha256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
 summary['scenes'][s]=dict(ROIs=30,views=90,crops=len(v['crops']),models_per_crop=3,human_records_created=0,URL='http://127.0.0.1:8785/'+d.name+'/index.html',page_sha256=sha(d/'index.html'),source_candidate_sha256=setup['source_candidate_sha256'],source_surface_sha256=setup['source_surface_sha256'],download_verification=read(H/s/'download_verification.json'))
(H/'delivery.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary['scenes'],ensure_ascii=False))
