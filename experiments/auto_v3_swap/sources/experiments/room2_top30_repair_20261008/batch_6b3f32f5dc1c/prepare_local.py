"""Validate the partial human export against its exact, frozen top30 assets."""
from pathlib import Path
import base64, hashlib, io, json, shutil
import numpy as np
from PIL import Image

WORK=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PAGE=WORK/'results/固定案例_三模型对比_20261006/room2_selection_top30_20261008'
SOURCE=WORK/'room2_前30_修复候选选择_20261008.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def decode(image):
 a=np.asarray(image,np.uint32)
 return a if a.ndim==2 else a[...,0]|(a[...,1]<<8)|(a[...,2]<<16)
def save(path,value):
 with Path(path).open('w',encoding='utf-8',newline='\n') as s:s.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
 selection=json.loads(SOURCE.read_text(encoding='utf-8-sig'))
 view=json.loads((PAGE/'viewer_data.json').read_text(encoding='utf-8-sig'))
 source=json.loads((WORK/'outputs/room2_selection_top30_20261008/ranked_source_manifest.json').read_text(encoding='utf-8-sig'))
 assert selection['schema']=='cvpr_roi_human_selection_v1'
 assert selection['scene_scope']=='room2' and selection['source_selection']==view['name']=='room2-top30-20261008'
 assert selection['is_qa'] is False and selection['total_fixed_cases']==len(view['cases'])==30
 assert selection['source_candidate_sha256']==source['candidate_manifest_sha256']
 assert selection['source_surface_sha256']==source['surface_sha256']
 cases={c['uid']:c for c in view['cases']}
 selections=selection['selections'];uids=[r['case_uid'] for r in selections]
 assert len(uids)==len(set(uids)) and set(uids)<=set(cases)
 assert set(selection['pilot_case_uids'])==set(cases)
 assert selection['annotation_complete']==(len(uids)==30)
 assert [(c['rank'],c['case_uid'],c['ROI']) for c in selection['case_order']]==[(c['rank'],c['uid'],c['ROI']) for c in view['cases']]
 target=OUT/'inputs';target.mkdir(exist_ok=True)
 shutil.copyfile(SOURCE,target/'human_selection.json')
 rows=[];oid=1
 for choice in selections:
  assert choice['scene']=='room2' and not choice.get('is_qa',False)
  case=cases[choice['case_uid']]
  assert case['ROI']==choice['ROI'] and case['rank']==choice['conflict_rank']
  selected=[v for v in case['views'] if v['frame']==choice['frame'] and v['number']==choice['view_number']]
  assert len(selected)==1 and choice['crop_key'] in selected[0]['variants'] and selected[0]['source_RGB']==choice['source_rgb']
  if choice['choice']=='unreliable':
   assert not choice['mask_ids'] and choice['label_asset'] is None
   rows.append({'status':'NO_RELIABLE_SEED','case_uid':choice['case_uid'],'source_choice':choice});continue
  crop=view['crops'][choice['crop_key']]
  assert crop['box']==choice['crop_xyxy'] and crop['wh']==choice['native_crop_wh']
  mi=['original','cropformer_local','sam2_1_local'].index(choice['choice']);assert mi==choice['model_index']
  model=crop['models'][mi]
  asset=(PAGE/model['ids_file']).resolve()
  assert asset==(PAGE/choice['label_asset']).resolve() and PAGE.resolve() in asset.parents
  labels=decode(Image.open(asset))
  np.testing.assert_array_equal(labels,decode(Image.open(io.BytesIO(base64.b64decode(model['ids'].split(',',1)[1])))))
  assert list(Image.open(asset).size)==crop['wh']
  mids=choice['mask_ids'];assert mids and len(mids)==len(set(mids))
  objects=[];seed=np.zeros((680,1200),np.uint16);x0,y0,x1,y1=choice['crop_xyxy']
  for mid in sorted(mids):
   mask=labels==mid;assert mask.any()
   seed[y0:y1,x0:x1][mask]=oid
   touch=bool((y0>0 and mask[0].any()) or (y1<680 and mask[-1].any()) or (x0>0 and mask[:,0].any()) or (x1<1200 and mask[:,-1].any()))
   objects.append({'track_id':oid,'native_mask_id':mid,'crop_pixels':int(mask.sum()),'touches_crop_border':touch})
   oid+=1
  assert sorted(o['native_mask_id'] for o in objects if o['touches_crop_border'])==sorted(choice.get('border_touching_mask_ids',[]))
  name=choice['case_uid']+'_selected_seed.png';Image.fromarray(seed).save(target/name)
  shutil.copyfile(asset,target/(choice['case_uid']+'_asset.png'))
  rows.append({'status':'READY','case_uid':choice['case_uid'],'scene':'room2','source_choice':choice,
   'objects':objects,'seed_file':name,'seed_sha256':sha(target/name),'source_label_sha256':sha(asset),
   'source_label_local_path':str(asset),'source_label_relative_path':model['ids_file'],'selection_viewer_sha256':sha(PAGE/'viewer_data.json')})
 missing=[{'case_uid':c['uid'],'ROI':c['ROI'],'rank':c['rank'],'status':'UNANNOTATED'} for c in view['cases'] if c['uid'] not in uids]
 report={'status':'PASS','annotation_sha256':sha(SOURCE),'annotation_exported_at':selection['exported_at'],
  'scene':'room2','total_ranked_cases':30,'total_cases':len(rows),'repair_cases':sum(r['status']=='READY' for r in rows),
  'selected_masks':oid-1,'no_reliable_seed_cases':sum(r['status']=='NO_RELIABLE_SEED' for r in rows),'unannotated_cases':missing,
  'source_candidate_sha256':selection['source_candidate_sha256'],'source_surface_sha256':selection['source_surface_sha256'],
  'annotation_complete':selection['annotation_complete'],'seeds':rows,'GT_used':False,
  'original_full_masks_to_be_restored_and_verified_against_raw_server_source':True}
 save(target/'local_ingest.json',report)
 save(OUT/'selection_views.json',{'scene':'room2','cases':[c for c in view['cases'] if c['uid'] in uids],'unannotated_cases':missing})
 print(json.dumps({k:v for k,v in report.items() if k!='seeds'},ensure_ascii=False))
if __name__=='__main__':main()
