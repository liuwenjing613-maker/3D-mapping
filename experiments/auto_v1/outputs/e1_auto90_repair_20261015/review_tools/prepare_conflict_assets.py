"""Read-only reconstruction of saved Gate overlap conflicts for visual review."""
from pathlib import Path
import json,hashlib,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
C=Path(__file__).resolve().parent.parent
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
O=R/'review'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def frame_assets(fid):
 d=decisions[fid];initial={int(k) for k,v in d['initial_objects'].items() if v['accepted']}
 count=np.zeros((680,1200),np.uint8);masks={}
 for case in cases:
  ids=initial.intersection(x['track_id'] for x in case['objects'])
  if not ids:continue
  labels=np.asarray(Image.open(R/'cases'/case['case_uid']/'tracking'/f'f{fid:06d}.png'))
  count+=np.isin(labels,list(ids)).astype(np.uint8)
  if fid in seed_frames:
   for tid in ids:masks[tid]=labels==tid
 conflict=count>1
 assert int(conflict.sum())==d.get('cross_case_conflict_pixels',0),(fid,int(conflict.sum()),d.get('cross_case_conflict_pixels'))
 accepted=np.asarray(Image.open(R/'validated/full_track/accepted_masks'/f'f{fid:06d}.png'))
 assert set(np.unique(accepted))<={0,*d['accepted_track_ids']}
 assert not np.any(accepted[conflict])
 outputs=[]
 for name,a in [('conflict',conflict.astype(np.uint8)*255),('repair_masks',accepted.astype(np.uint8))]:
  path=O/name/f'f{fid:06d}.png';Image.fromarray(a).save(path)
  np.testing.assert_array_equal(np.asarray(Image.open(path)),a)
  if name=='repair_masks':np.testing.assert_array_equal(a,accepted)
  outputs.append({'file':str(path.relative_to(O)),'sha256':sha(path)})
 pairs=[]
 for tid in seed_frames.get(fid,[]):
  if tid not in masks:continue
  area=int(masks[tid].sum())
  for other,b in masks.items():
   if tid==other:continue
   n=int(np.count_nonzero(masks[tid]&b))
   if n:pairs.append({'track_id':tid,'other_track_id':other,'pixels':n,'fraction_of_this_mask':n/max(area,1)})
 return {'frame':fid,'conflict_pixels':int(conflict.sum()),'outputs':outputs,'seed_conflicts':pairs}

def main():
 global decisions,cases,seed_frames
 assert read(R/'validated/full_track/complete.json')['status']=='PASS'
 assert read(R/'alias_review.json')['aliases']=={}
 decisions={x['frame']:x for x in read(R/'validated/full_track/frame_decisions.json')['frames']}
 cases=read(R/'seed_diagnostics.json')['cases'];seed_frames={}
 for x in read(R/'seed_id_map.json')['objects']:seed_frames.setdefault(x['frame'],[]).append(x['track_id'])
 for name in ['conflict','repair_masks']:(O/name).mkdir(exist_ok=True)
 start=time.time()
 with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(frame_assets,sorted(decisions)))
 result={'status':'PASS','frames':len(rows),'conflicts_exactly_match_frozen_Gate':True,'accepted_masks_exact':True,'source_decisions_sha256':sha(R/'validated/full_track/frame_decisions.json'),'code_sha256':sha(Path(__file__)),'seconds':time.time()-start,'rows':rows}
 (R/'conflict_assets_complete.json').write_text(json.dumps(result))
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
