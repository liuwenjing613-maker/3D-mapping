from pathlib import Path
import json,collections
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015')
read=lambda p:json.loads(p.read_text())
done=read(R/'validated/full_track/complete.json');counts=done['counts']['accepted_frames_by_track']
frames=read(R/'validated/full_track/frame_decisions.json')['frames']
display={x['track_id']:x for x in read(C/'display_metadata.json')['objects']}
seeds={x['track_id']:x for x in read(R/'seed_id_map.json')['objects']}
rows=[]
for key,count in counts.items():
 if count:continue
 tid=int(key);reasons=collections.Counter();seed=None;visible=0;initial_pass=0;overlap_rejections=0
 for frame in frames:
  check=frame['objects'][key];reasons.update(check['reasons']);visible+=check['visible_seed_samples']>=20
  initial=frame['initial_objects'][key];initial_pass+=initial['accepted'];overlap_rejections+=initial['accepted'] and not check['accepted'] and check.get('rechecked_after_overlap_abstention',False)
  if frame['frame']==seeds[tid]['frame']:seed={**{k:check[k] for k in ('visible_seed_samples','visible_seed_fraction','tracked_coverage','new_surface_bbox_fraction','reasons')},'initial_coverage':initial['tracked_coverage'],'initial_accepted':initial['accepted'],'overlap_rechecked':check.get('rechecked_after_overlap_abstention',False)}
 rows.append({'track_id':tid,'group':display[tid]['group_id'],'mask':display[tid]['mask_id'],'source':display[tid]['model'],'visible_frames_20samples':visible,'initial_pass_frames':initial_pass,'overlap_rejection_frames':overlap_rejections,'reasons':dict(reasons),'seed_frame':seed})
print(json.dumps({'accepted_seeds':sum(x>0 for x in counts.values()),'zero_accepted_seeds':rows},ensure_ascii=False))
