"""Freeze only RGB-reviewed duplicate chair identities, before any GT evaluation."""
from pathlib import Path
import hashlib,json
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert json.loads((ROOT/'tracking_queue_complete.json').read_text())['status']=='PASS'
 diagnostics=json.loads((ROOT/'cross_seed_diagnostics.json').read_text())
 assert diagnostics['status']=='PASS' and diagnostics['GT_used'] is False
 seeds=json.loads((ROOT/'seed_manifest.json').read_text())['seeds']
 objects={o['track_id']:(r,o) for r in seeds for o in r.get('objects',[])}
 decisions=[]
 for a,b in [(4,24),(5,25)]:
  pair=next(p for p in diagnostics['pairs'] if (p['track_a'],p['track_b'])==(a,b))
  assert pair['same_old_family']
  for key in ['a_in_b_seed','b_in_a_seed']:
   assert pair[key]['visible_samples']>=1000 and pair[key]['coverage']>.9
  decisions.append({'member_track_ids':[a,b],'canonical_track_id':a,
   'physical_identity':'Same dining chair next to the window, viewed at f1065 and f1605',
   'sources':[{'ROI':objects[x][0]['source_choice']['ROI'],'frame':objects[x][0]['source_choice']['frame'],'mask_id':objects[x][1]['native_mask_id']} for x in [a,b]],
   'visual_review':'RGB seed contacts inspected before GT. Corresponding chair backs and legs match the reciprocal depth-visible seed supports; the two chairs stay separate.',
   'evidence':pair,'action':'Alias persistent identity only; preserve both seeds, all raw bidirectional masks, and unchanged per-frame reliability checks.'})
 review={'status':'FROZEN','GT_used':False,'aliases':{'24':4,'25':5},'unique_objects':23,
  'annotation_sha256':sha(ROOT/'inputs/human_selection.json'),'cross_seed_diagnostics_sha256':sha(ROOT/'cross_seed_diagnostics.json'),
  'review_code_sha256':sha(__file__),'seed_contact_hashes':{str(ROOT/name):sha(ROOT/name) for name in ['seed_contact_01.jpg','seed_contact_04.jpg','seed_contact_05.jpg']},
  'decisions':decisions,'other_tracks_remain_distinct':sorted(set(objects)-{4,5,24,25}),
  'no_GT_regrouping':True,'all_raw_tracks_and_numeric_thresholds_unchanged':True,
  'uncertain_same_family_or_disjoint_parts_are_not_automatically_aliased':True}
 path=ROOT/'alias_review.json'
 if path.exists():assert json.loads(path.read_text())==review
 else:path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'FROZEN','aliases':review['aliases'],'unique_objects':23}))
if __name__=='__main__':main()
