from pathlib import Path
import json,numpy as np
from PIL import Image
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
OLD=ROOT/'room2_E1_prerepair_CUT_top20_20261015';NEW=ROOT/'room2_E1_retained_ROI_top20_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def decode(p):
 a=np.asarray(Image.open(p),np.uint16);return a[:,:,0]+256*a[:,:,1]
old=read(OLD/'viewer_data.json');new=read(NEW/'viewer_data.json');pro=read(NEW/'projection_annotations.json');assets=set();removed=0;changed=0
assert len(new['cases'])==32 and len(new['crops'])==len(old['crops']) and sum(len(c['views']) for c in new['cases'])==180
for c in new['cases']:
 for v in c['views']:assets.update([v['full_rgb'],v['alarm']])
for key,crop in new['crops'].items():
 assets.add(crop['RGB'])
 if 'comparison' in crop:assets.add(crop['comparison'])
 for mi,m in enumerate(crop['models']):
  assets.update([m['color'],m['boundary'],m['ids_file']]);assert m['ids']==m['ids_file'];rm=set(m['removed_ids']);removed+=len(rm)
  assert not (rm&set(map(int,m['colors']))) and not (rm&{p['id'] for p in m['positions']})
  for state in ['U','T','C']:assert not(rm&{r['id'] for r in pro['crops'][key]['models'][mi][state]})
  assert m['count']==m['count_before_filter']-len(rm)==len(m['colors'])
  if not rm:continue
  changed+=1;before=decode(OLD/old['crops'][key]['models'][mi]['ids_file']);after=decode(NEW/m['ids_file']);discard=np.isin(before,list(rm))
  assert np.all(after[discard]==0) and np.array_equal(after[~discard],before[~discard])
  assert len(np.unique(after[after>0]))==m['count']
  for typ in ['color','boundary']:
   before_img=np.asarray(Image.open(OLD/old['crops'][key]['models'][mi][typ]).convert('RGBA'));after_img=np.asarray(Image.open(NEW/m[typ]).convert('RGBA'))
   assert np.all(after_img[discard]==0) and np.array_equal(after_img[~discard],before_img[~discard])
for path in assets:assert (NEW/path).resolve().is_file(),path
html=(NEW/'index.html').read_text(encoding='utf-8')
assert 'cvpr-room2-e1-retained-roi-top20-20261015-v1' in html and 'cvpr-room2-e1-prerepair-cut-top20-20261015-v1' not in html
assert 'review_label_asset:' in html and "label_asset:chosenModel<0?null:'assets/'" in html
r=dict(status='PASS',ROIs=32,views=180,crops=len(new['crops']),referenced_assets=len(assets),removed_occurrences=removed,changed_model_crops=changed,
 removed_IDs_absent_from_images_legends_targets_and_projections=True,remaining_pixels_and_colors_exactly_preserved=True,export_uses_original_source_IDs_and_filtered_review_assets=True)
(H/'retained_verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r,ensure_ascii=False))
