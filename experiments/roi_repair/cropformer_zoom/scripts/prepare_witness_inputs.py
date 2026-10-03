from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
R=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003')
OLD=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002/conflict_diagnosis')
m=json.loads((R/'input_manifest.json').read_text());existing={j['key'] for j in m['jobs']};cases=json.loads((OLD/'conflict_cases.json').read_text())
cfg=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/configs/raw.json').read_text());scene=Path(cfg['source']['scene_root']);maskroot=Path(cfg['source']['mask_root'])
source_meta={r['frame']:r for r in map(json.loads,(maskroot/'frames.jsonl').read_text().splitlines())}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def add(key,rgb,baseline,info):
 assert key not in existing
 inp=R/'inputs'/f'{key}_rgb.png';base=R/'inputs'/f'{key}_baseline.png';assert not inp.exists()
 Image.fromarray(rgb).save(inp);Image.fromarray(baseline).save(base)
 job={'key':key,'input_file':str(inp),'baseline_file':str(base),'shape':list(rgb.shape[:2]),'RGB_file_sha256':sha(inp),'baseline_file_sha256':sha(base),'same_RGB_as':None,**info}
 m['jobs'].append(job);existing.add(key);return job
new=0;controls=0
for rec in cases:
 for side,v in enumerate(rec['source_views']):
  fid=v['frame_id'];i=v['id'];key=f'{rec["roi_id"]}_witness_ID{i}';assert key not in existing
  im=scene/f'results/frame{fid:06d}.jpg';mask=maskroot/f'frame{fid:06d}.png';assert sha(im)==source_meta[fid]['input_sha256'] and sha(mask)==source_meta[fid]['mask_sha256']
  rgb=np.asarray(Image.open(im).convert('RGB'));baseline=np.asarray(Image.open(mask));ux,vy=v['source_pixel_uv'];crop=[max(0,ux-220),max(0,vy-160),min(1200,ux+220),min(680,vy+160)];a,b,c,d=crop
  with np.load(OLD/f'{rec["roi_id"]}_source_ID{i}.npz') as z:assert np.array_equal(rgb[b:d,a:c],z['rgb'])
  vn=4+side;job=add(key,rgb[b:d,a:c],baseline[b:d,a:c],{'kind':'ROI_crop','strategy':'witness','ROI':rec['roi_id'],'rank':rec['rank'],'view':vn,'frame':fid,'crop_xyxy':crop,'source_instance_id':i,'source_mask_local_id':v['mask_local_id'],'alarm_projection_anchor_uv':[ux-a,vy-b]})
  m['cases'].append({'roi_id':rec['roi_id'],'rank':rec['rank'],'view':vn,'view_label':f'竞争来源 ID{i}','frame':fid,'representative':True,'core_counts':rec['counts'],'crops':{'witness':key}});new+=1
  fullkey=f'full_f{fid:06d}'
  if fullkey not in existing:add(fullkey,rgb,baseline,{'kind':'full_frame_control','frame':fid,'source_RGB_sha256':source_meta[fid]['input_sha256']});controls+=1
m['competition_source_RGB_crops']=new;m['original_selected_ROI_views']=m['selected_ROI_views'];m['selected_ROI_views']+=new;m['full_frame_controls']+=controls
(R/'input_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('ADDED_WITNESSES',new,'EXTRA_CONTROLS',controls,flush=True)
