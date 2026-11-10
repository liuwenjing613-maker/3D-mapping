"""Independent final checks of candidate prefix manifests and unchanged dual95 decisions."""
from pathlib import Path
import json,hashlib,re,time
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
CONT=BASE/'continuation300';TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022');OUT=TOP/'continuation300'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def boundary_check(candidate,grouping):
 import numpy as np
 from PIL import Image
 prior=candidate/'grouping_r2'
 source=read(prior/'grouping.json')['seeds']
 audit=read(candidate/'all_edge_20261021/boundary_audit.json')
 rows={x['id']:x for x in audit['masks']}
 source_ids={x['id'] for x in source}
 assert len(rows)==len(source_ids)==len(source) and set(rows)==source_ids
 edges=set();full=set();crop=set()
 for s in source:
  p=prior/s['mask'];assert sha(p)==s['mask_sha256']
  a=np.asarray(Image.open(p))>0
  assert a.shape==(680,1200) and int(a.sum())==s['area']
  contacts=dict(left=int(a[:,0].sum()),top=int(a[0,:].sum()),right=int(a[:,-1].sum()),bottom=int(a[-1,:].sum()))
  names=set()
  for alias in s['aliases']:
   if alias['model']==0:continue
   x0,y0,x1,y1=alias['source_box']
   if x0>0 and a[y0:y1,x0].any():names.add('left')
   if y0>0 and a[y0,x0:x1].any():names.add('top')
   if x1<1200 and a[y0:y1,x1-1].any():names.add('right')
   if y1<680 and a[y1-1,x0:x1].any():names.add('bottom')
  record=rows[s['id']]
  assert record['frame_edge_pixels']==contacts and set(record['artificial_crop_edges'])==names
  assert record['touches_full_frame']==any(contacts.values())
  assert record['touches_any_edge']==bool(any(contacts.values()) or names)
  if any(contacts.values()):full.add(s['id'])
  if names:crop.add(s['id'])
 edges=full|crop;survivors=source_ids-edges
 assert {x['id'] for x in audit['edge_excluded']}==edges
 assert set(audit['boundary_survivor_mask_ids'])==survivors
 active={x['id'] for x in grouping['seeds']};noise={x['id'] for x in audit['noise_excluded']}
 assert active.isdisjoint(noise) and active|noise==survivors
 assert not (active|noise)&edges
 counts=dict(input_masks=len(source),full_frame_removed=len(full),crop_removed=len(crop),
  both_edge_removed=len(full&crop),total_edge_removed=len(edges),after_boundary_filter=len(survivors),
  noise_removed=len(noise),after_noise=len(active),groups=len(grouping['groups']))
 for k,v in counts.items():assert audit['stats'][k]==grouping['stats'][k]==v,(k,v)
 assert {j for x in grouping['groups'] for j in x['members']}==set(range(len(grouping['seeds'])))
 return dict(status='PASS',actual_input_mask_arrays_checked=len(source),source_grouping_sha256=sha(prior/'grouping.json'),
  boundary_audit_sha256=sha(candidate/'all_edge_20261021/boundary_audit.json'),
  full_frame_and_all_alias_crop_edges_recomputed=True,edge_masks_removed_before_noise_and_regrouping=True,
  survivor_partition_exact=True,counts=counts,
  generic_versions_warning_only_is_not_the_final_selection_gate=True)

def run(n,g):
 root=OUT/str(n)/g;candidate=root/'candidates';tracking=root/'tracking';lo=0 if g=='A' else (n-100)*5;cutoff=(n-1)*5
 dense=read(root/'inputs_complete.json');cfg=read(tracking/'track_config.json')
 assert dense['raw_frames']==cutoff+1 and dense['raw_cutoff']==cutoff and dense['mapping_frames']==n
 keys=list(dense['raw_rgb_hashes']);assert len(keys)==cutoff+1
 frame_ids=[]
 for value in keys:
  p=Path(value);m=re.fullmatch(r'frame(\d{6})\.jpg',p.name);assert m
  fid=int(m.group(1));assert 0<=fid<=cutoff
  actual=p.resolve();assert actual.name==p.name and actual.is_file()
  frame_ids.append(fid)
 assert sorted(frame_ids)==list(range(cutoff+1))
 trajectory=Path(cfg['source']['scene_root'])/cfg['source']['trajectory']
 assert len(trajectory.read_text().splitlines())==cutoff+1 and sha(trajectory)==dense['trajectory_sha256']
 manifest=read(candidate/'catalog/room0/candidate_manifest.json')
 assert manifest['candidate_search_frame_ids']==list(range(lo,cutoff+1,5))
 assert manifest['scoring_history_frame_ids']==list(range(0,cutoff+1,5))
 assert manifest['previous_committed_checkpoint']==n-100 and not manifest['current_checkpoint_repaired']
 assert Path(manifest['surface_file']).resolve()==(root/'mapping/baseline/final/instance_surface.npz').resolve()
 assert sha(manifest['surface_file'])==manifest['surface_sha256']
 assert sha(manifest['evidence_file'])==manifest['evidence_sha256']
 jobs=read(candidate/'frontend/input_manifest.json')['jobs']
 assert all(j['frame'] in range(lo,cutoff+1,5) for j in jobs)
 o=candidate/'all_edge_20261021';selected=read(o/'v2/selection.json');chosen=read(o/'v2/seed_manifest.json');grouping=read(o/'v1/grouping.json')
 boundary=boundary_check(candidate,grouping)
 assert selected['candidate_raw_frame_range']==[lo,cutoff]
 assert selected['candidate_processed_frame_range']==[lo//5+1,n]
 assert selected['scoring_history']==[1,n] and selected['repair100_applied'] and not selected['current_checkpoint_repaired']
 assert grouping['rule']['candidate_raw_frame_range']==[lo,cutoff]
 assert all(s['frame'] in range(lo,cutoff+1,5) and not s['touches_any_edge'] for s in grouping['seeds'])
 coverage={x['group_id']:x for x in read(o/'v2/measurements.json')['coverage']}
 stages={x['stage']:x for x in selected['stages']}
 stage3=set(stages['full_support_containment']['retained_group_ids'])
 expected=set()
 for gid in stage3:
  m=coverage[gid];c=m['intersection'];a=m['support_count'];b=m['instance_visible_count']
  if a>0 and b>0 and 20*c>=19*a and 20*c>=19*b:expected.add(gid)
 assert expected==set(stages['skip_repair_B']['removed_group_ids'])
 stage4=set(stages['skip_repair_B']['retained_group_ids'])
 exempt=set(selected['semantic_pr_exempt_group_ids'])
 pr={gid for gid in stage4 if 10*coverage[gid]['intersection']>coverage[gid]['support_count'] and 5*coverage[gid]['intersection']<coverage[gid]['instance_visible_count']} - exempt
 assert pr==set(stages['pr_filter']['removed_group_ids'])
 kept=set(selected['retained_group_ids']);assert kept==stage4-pr=={x['id'] for x in chosen['groups']}
 assert kept.isdisjoint(expected),'Stage5 may not rescue original dual95 exclusions'
 for x in chosen['groups']:
  assert sha(o/'v1'/x['primary']['mask'])==x['primary']['mask_sha256']
 seeds=read(tracking/'seed_id_map.json')['objects'];assert len(seeds)==len(kept)
 primaries={x['id']:x['primary'] for x in chosen['groups']}
 assert {x['group_id'] for x in seeds}==kept
 for x in seeds:
  assert x['source']['mask_sha256']==primaries[x['group_id']]['mask_sha256'] and x['frame']==primaries[x['group_id']]['frame']
 prepare=read(tracking/'prepare_complete.json');queue=read(tracking/'tracking_queue_complete.json');gate=read(tracking/'gate_complete.json')
 assert prepare['raw_prefix']==[0,cutoff] and prepare['mapping_frames']==n
 assert queue['raw_frames']==cutoff+1 and queue['raw_cutoff']==cutoff and queue['mapping_frames']==n
 assert gate['checkpoint']==n and gate['objects']==len(kept)
 for case in read(tracking/'seed_diagnostics.json')['cases']:
  report=read(tracking/'cases'/case['case_uid']/'tracking/complete.json')
  assert report['total_frames']==report['completed_frames']==cutoff+1
  assert [row['frame'] for row in report['frames']]==list(range(cutoff+1))
  assert report['directions_use_independent_seed_only_states'] and not report['GT_used']
  for row in report['frames']:
   if row['frame']!=report['seed_frame']:assert row['direction']==('reverse' if row['frame']<report['seed_frame'] else 'forward')
 original_filter=read(TOP/'prefix100/all_edge_20261021/complete.json')['filter_code_sha256']
 assert read(o/'complete.json')['filter_code_sha256']==original_filter
 qa=read(candidate/'online_qa.json');assert qa['status']=='PASS' and qa['retained']==len(kept)
 return dict(status='PASS',checkpoint=n,group=g,candidate_processed_range=[lo//5+1,n],candidate_raw_range=[lo,cutoff],
  scoring_history=[1,n],dense_tracking_range=[0,cutoff],dense_RGB_count=cutoff+1,retained=len(kept),retained_group_ids=sorted(kept),
  stage3_candidates=len(stage3),original_dual95_excluded=len(expected),stage5_excluded=len(pr),semantic_exempt_survivors=sorted(exempt&kept),
  boundary_entry_audit=boundary,all_primary_and_aux_mask_source_frames_valid=True,edge_free_QA_sha256=sha(candidate/'online_qa.json'),
  original_rule='stage4 p>=.95 AND r>=.95 excludes; stage5 p>.1 AND r<.2 except original semantic exemption',
  original_seed_mask_sha256_preserved_for_tracking=True,both_independent_directions_full_arrived_history=True,
  map_source_branch_correct=True,previous_checkpoint=n-100,model_cache=read(candidate/'exact_model_cache_audit.json')['counts'],
  inferred_tracking_cases=queue['inferred_cases'],reused_tracking_cases=queue['reused_cases'])
def main():
 start=time.perf_counter();rows=[];pending=[]
 for n in [200,300]:
  for g in ['A','B']:
   marker=OUT/str(n)/g/'tracking/complete.json'
   if not marker.exists() or read(marker).get('status')!='PASS':pending.append([n,g]);continue
   rows.append(run(n,g))
 result=dict(status='PASS' if not pending else 'PARTIAL_PENDING',rows=rows,pending=pending,seconds=time.perf_counter()-start,
  validates_only=True,models_or_repair_reexecuted=False,GT_used=False)
 path=OUT/'selection_protocol_audit.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(path)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
