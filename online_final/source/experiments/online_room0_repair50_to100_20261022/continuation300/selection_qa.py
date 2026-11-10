"""Bounded post-selection acceptance on the actual current-branch map and masks."""
from common_continue import *
import numpy as np
from PIL import Image
def main():
 started=time.perf_counter();lo=0 if GROUP=='A' else OLD_COUNT*5
 allowed=list(range(lo,RAW_COUNT,5));O=CANDIDATE_ROOT/'all_edge_20261021';V=O/'v1';S2=O/'v2'
 mapping=read(B/'mapping_complete.json');assert mapping['status']=='PASS' and mapping['frame_ids']==FIDS
 assert (CANDIDATE_ROOT/'baseline').resolve()==(B/'baseline').resolve()
 for path,h in read(CANDIDATE_ROOT/'mapping_input_freeze.json')['source_hashes'].items():assert digest(path)==h,path
 cat=read(CANDIDATE_ROOT/'catalog/room0/candidate_manifest.json')
 assert cat['candidate_search_frame_ids']==allowed and cat['scoring_history_frame_ids']==FIDS
 assert cat['history_frames']==CHECKPOINT and cat['previous_committed_checkpoint']==OLD_COUNT
 assert cat['repair100_applied'] and not cat['current_checkpoint_repaired']
 views=[v['frame'] for e in cat['entries'] for v in e['views']]
 assert views and all(fid in allowed for fid in views)
 jobs=read(CANDIDATE_ROOT/'frontend/input_manifest.json')['jobs']
 assert all(j['frame'] in allowed for j in jobs)
 grouping=read(V/'grouping.json');chosen=read(S2/'seed_manifest.json');selection=read(S2/'selection.json')
 assert set(g['id'] for g in chosen['groups'])==set(selection['retained_group_ids'])
 checked=0;sam=0
 for row in grouping['seeds']:
  assert row['frame'] in allowed
  assert not row['touches_full_frame'] and not row['artificial_crop_edges'] and not row['touches_any_edge']
  path=V/row['mask'];assert digest(path)==row['mask_sha256'];a=np.asarray(Image.open(path))>0
  assert int(a.sum())==row['area'] and a.shape==(680,1200)
  assert not (a[0].any() or a[-1].any() or a[:,0].any() or a[:,-1].any())
  for alias in row['aliases']:
   if alias['model']==0:continue
   x0,y0,x1,y1=alias['source_box']
   assert not (x0>0 and a[y0:y1,x0].any())
   assert not (y0>0 and a[y0,x0:x1].any())
   assert not (x1<1200 and a[y0:y1,x1-1].any())
   assert not (y1<680 and a[y1-1,x0:x1].any())
  if any(alias.get('semantic_hit') for alias in row['aliases']):assert row['representative']['model']==2;sam+=1
  checked+=1
 # Original rules are frozen byte-for-byte. Newer95/90 code is not loaded.
 sc=CONT/'selection_code'/str(CHECKPOINT)/GROUP
 for rel in ['v2/filter_rules.py','v2/semantic_exemption.py','v1/noise_filter.py','v1/crop_filter.py','v1/source_policy.py']:
  assert digest(sc/rel)==digest(BASEC/'prefix100'/rel),rel
 for path,h in grouping['source_sha256'].items():assert digest(path)==h,path
 complete=read(O/'complete.json');assert complete['status']=='PASS' and complete['retained']==len(chosen['groups'])
 original_filter=read(TOP/'prefix100/all_edge_20261021/complete.json')['filter_code_sha256']
 assert complete['filter_code_sha256']==original_filter,'Actual direct-containment/dual95 source differs from accepted100'
 assert digest('/home/chenkejun/CVPR/experiments/online_room0_prefix50_20261021/v2/filter_rules.py')==original_filter
 blocked=[];events=0
 for p in (ROOT/'io_audit').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   row=json.loads(line)
   if row.get('probe'):continue
   events+=1
   if row.get('blocked'):blocked.append(row)
 assert not blocked,blocked[:3]
 report=dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,previous_checkpoint=OLD_COUNT,raw_cutoff=CURRENT_RAW,
  candidate_processed_frame_range=[lo//5+1,CHECKPOINT],candidate_raw_frame_range=[lo,CURRENT_RAW],scoring_history=[1,CHECKPOINT],
  candidate_view_count=len(views),candidate_distinct_frames=sorted(set(views)),frontend_jobs=len(jobs),
  retained=len(chosen['groups']),retained_group_ids=selection['retained_group_ids'],all_active_and_aux_masks_edge_free=True,
  checked_active_and_aux_masks=checked,semantic_routed_SAM_masks=sam,original_dual95_rules_unchanged=True,
  stage5_semantic_exemptions_unchanged=True,source_masks_unchanged=True,current_branch_mapping_unchanged=True,
  mapping_source_freeze=str(CANDIDATE_ROOT/'mapping_input_freeze.json'),source_manifest=str(S2/'seed_manifest.json'),
  source_manifest_sha256=digest(S2/'seed_manifest.json'),recorded_IO_events=events,forbidden_nonprobe_accesses=0,
  model_cache=read(CANDIDATE_ROOT/'exact_model_cache_audit.json')['counts'],current_checkpoint_repaired=False,GT_used=False,
  seconds=time.perf_counter()-started)
 dump(CANDIDATE_ROOT/'online_qa.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
