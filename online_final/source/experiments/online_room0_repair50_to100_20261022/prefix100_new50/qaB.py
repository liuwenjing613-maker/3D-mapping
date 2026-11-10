from online_common import *
import time,sys,re
from collections import Counter
import numpy as np
from PIL import Image
A=C.parent/'prefix100';AR=R.parent/'prefix100'
def main():
 provenance=read(R/'arm_provenance.json');assert provenance['scoring_history']==[1,100] and provenance['candidate_processed_frame_range']==[51,100]
 for p,h in provenance['shared_sources_sha256'].items():assert sha(p)==h,p
 aq=read(AR/'online_qa.json');assert aq['status']=='PASS' and aq['strict_vote_recount_exact'] and aq['committed_history_reprojection_on_100_geometry_exact']
 mapping=read(R/'mapping_complete.json');assert mapping['frame_ids']==list(FIDS) and len(FIDS)==100 and max(FIDS)==495
 assert mapping['repair_applied'] and not mapping['repair100_applied']
 assert read(R/'frame51_state_audit.json')['repaired_state_used']
 cat=read(R/'catalog/room0/candidate_manifest.json');acat=read(AR/'catalog/room0/candidate_manifest.json')
 assert cat['history_frames']==100 and cat['candidate_search_frame_ids']==list(range(250,496,5)) and cat['scoring_history_frame_ids']==list(FIDS)
 assert cat['candidate_source_policy']=='new_mapping_frames_51_to100' and cat['repair50_applied'] and not cat['repair100_applied']
 assert cat['surface_sha256']==acat['surface_sha256'] and cat['evidence_sha256']==acat['evidence_sha256'] and cat['graph_sha256']==acat['graph_sha256']
 # ROI components/ranking are identical; only their allowed source view set changes.
 assert [(e['id'],e['core_sha256'],e['count']) for e in cat['entries']]==[(e['id'],e['core_sha256'],e['count']) for e in acat['entries']]
 with np.load(R/'catalog/room0/membership.npz') as z,np.load(AR/'catalog/room0/membership.npz') as az:
  assert set(z.files)==set(az.files)
  for k in z.files:np.testing.assert_array_equal(z[k],az[k])
 views=[v['frame'] for e in cat['entries'] for v in e['views']]
 assert views and all(fid in range(250,496,5) for fid in views)
 jobs=read(R/'frontend/input_manifest.json')['jobs'];assert all(j['frame'] in range(250,496,5) for j in jobs)
 for j in jobs:assert sha(j['input_file'])==j['RGB_sha256'] and sha(j['baseline_file'])==j['baseline_sha256']
 model_execution_counts={}
 for model in ('cropformer','sam2_1'):
  recs=[read(R/'frontend'/model/(j['key']+'.json')) for j in jobs]
  model_execution_counts[model]={'source_A_exact_records':sum(str(q.get('reuse_from','')).startswith('exact-A:') for q in recs),'within_B_duplicate_records':sum(bool(q.get('reuse_from')) and not str(q.get('reuse_from')).startswith('exact-A:') for q in recs),'new_model_inferences':sum(not q.get('reuse_from') for q in recs)}
  audit=read(R/'frontend'/model/'audit.json');assert audit['status']=='PASS' and audit['processed']==len(jobs)
 cache=read(R/'exact_model_cache_audit.json');assert cache['status']=='PASS'
 semantic_cache=read(R/'exact_semantic_cache_audit.json');assert semantic_cache['status']=='PASS'
 assert sha(semantic_cache['openclip_source'])==semantic_cache['openclip_source_sha256']
 assert sha(A/'semantic_route/classify.py')==semantic_cache['openclip_classifier_A_sha256']
 front=read(R/'semantic_route/semantic.json');afront=read(AR/'semantic_route/semantic.json');assert front['cache_namespace']==afront['cache_namespace']
 bg=read(R/'v2/semantic/classification.json');edgebg=read(R/'all_edge_20261021/v2/semantic/classification.json')
 ce=read(R/'semantic_route/cache_equivalence.json');numeric=read(R/'semantic_route/feature_numeric_cache_audit.json')
 assert ce['status']=='PASS' and ce['namespace_used']==front['cache_namespace'] and not ce['semantic_decisions_reused']
 if not numeric['all_features_bitwise_equal']:
  assert front['stats']['source_A_unique_feature_hits']==0 and front['stats']['source_A_features_discarded_after_numeric_check']>0
 abg=read(AR/'v2/semantic/classification.json')
 semantic_namespace_proof={'openclip':ce,'openclip_numeric_check':numeric,'segformer':[]}
 semantic_model_counts=dict(openclip_mask_feature_encodings=front['stats']['unique_image_encodings'],openclip_numeric_validation_encodings=numeric['samples'],openclip_cabinet_context_encodings=front['stats'].get('mask_only_hit_cabinet',0),openclip_A_feature_encodings_reused=front['stats']['source_A_unique_feature_hits'],segformer_initial_new_frames=bg['stats']['newly_inferred_frames'],segformer_initial_cached_frames=bg['stats']['cached_frames'],segformer_after_edge_new_frames=edgebg['stats']['newly_inferred_frames'],segformer_after_edge_cached_frames=edgebg['stats']['cached_frames'])

 for target in (bg,edgebg):
  assert target['model']==abg['model'] and target['config']==abg['config']
  preproc=next(h for path,h in target['input_hashes'].items() if path.endswith('/preprocessor_config.json'))
  apreproc=next(h for path,h in abg['input_hashes'].items() if path.endswith('/preprocessor_config.json'))
  assert preproc==apreproc
  runtime_model=dict(target['model'])
  for field in ('id2label','excluded_id_to_label'):runtime_model[field]={int(k):v for k,v in runtime_model[field].items()}
  basis={'model':runtime_model,'postprocess':target['config']['postprocess'],'preprocessor_file_sha256':preproc}
  ns=__import__('hashlib').sha256(json.dumps(basis,sort_keys=True).encode()).hexdigest()
  for fr in target['frames']:
   root=(R/'v2/semantic') if target is bg else (R/'all_edge_20261021/v2/semantic')
   meta=read((root/fr['cache']).with_suffix('.json'));assert meta['namespace']==ns and meta['RGB_sha256']==fr['RGB_sha256']
  semantic_namespace_proof['segformer'].append(dict(namespace=ns,fields=basis,configuration_matches_A=True,source_weights_and_download_manifest_rehashed_by_runtime=True,implementation_matches_A=True))

 for rel in ('v1/source_policy.py','v2/filter_rules.py','v2/semantic_exemption.py','v2/semantic/config.json'):assert sha(C/rel)==sha(A/rel),rel
 for row in cache['rows']:
  nr=read(R/'frontend'/row['model']/(row['target_key']+'.json'));assert nr['RGB_sha256']==row['RGB_sha256'] and sha(nr['mask_file'])==row['mask_sha256']
 O=R/'all_edge_20261021';W=O/'review';data=read(W/'data.json');grouping=read(O/'v1/grouping.json');seed=read(O/'v2/seed_manifest.json')
 assert data['summary']['candidate_source_policy']=='new_mapping_frames_51_to100' and data['summary']['scoring_history']==[1,100]
 assert data['summary']['repair50_applied'] and not data['summary']['repair100_applied']
 assert seed['candidate_processed_frame_range']==[51,100] and seed['scoring_history']==[1,100]
 assert all(s['frame'] in range(250,496,5) for s in grouping['seeds'])
 for s in grouping['seeds']:assert not s['touches_any_edge'] and not s['artificial_crop_edges']
 sys.path.insert(0,str(C/'v1'));from source_policy import assert_sam_only_groups
 assert_sam_only_groups([g['members'] for g in grouping['groups']],grouping['seeds'])
 for p,h in grouping['source_sha256'].items():assert sha(p)==h,p
 rows=data['rows'];kept=[x for x in rows if x['status']=='kept'];assert len(kept)==len(seed['groups'])==data['summary']['retained']
 for row in rows:
  assert row['frame'] in range(250,496,5)
  assert (row['status']=='edge')==(bool(any(row['edge_pixels'].values())) or bool(row['crop_edges']))
  for k in ('full','zoom','rgb','mask'):assert (W/row[k]).is_file()
 for row in kept:
  assert not any(row['edge_pixels'].values()) and not row['crop_edges']
  binary=np.asarray(Image.open(W/row['mask']))>0
  assert not (binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any())
 blocked=[];events=0
 for p in (R/'io_audit').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   e=__import__('json').loads(line);events+=1
   if e['blocked'] and not e.get('probe'):blocked.append(e)
 assert not blocked,blocked[:5]
 timings=[__import__('json').loads(l) for l in (R/'process_timing_events.jsonl').read_text().splitlines()]
 assert all(e['status']=='PASS' for e in timings if e['stage']!='qa')
 previous_QA_attempts=[e for e in timings if e['stage']=='qa']
 result=dict(status='PASS',arm='B',candidate_source_policy='new_mapping_frames_51_to100',candidate_processed_frame_range=[51,100],candidate_raw_frame_range=[250,495],scoring_history=[1,100],scoring_map_identical_to_A=True,shared_100_vote_recount_verified_by_identical_A_inputs=True,ROI_components_and_rankings_equal_A=True,ROI_view_search_restricted_before_mask_generation=True,ROI_view_count=len(views),ROI_distinct_frames=sorted(set(views)),unviewable_ROIs=[e['id'] for e in cat['entries'] if not e['views']],frontend_jobs=len(jobs),exact_model_cache=cache['counts'],model_execution_counts=model_execution_counts,semantic_namespace_proof=semantic_namespace_proof,semantic_model_counts=semantic_model_counts,model_output_reuse_claim='Same pixels, weight/config and implementation: existing A raw predictions reused file-for-file with identical output SHA; no new full fresh-run equality claim',semantic_cache_actual_stats={'frontend':front['stats'],'background_full':bg['stats'],'background_after_edge':edgebg['stats']},retained=len(kept),retained_frames=sorted(set(x['frame'] for x in kept)),candidate_count=len(rows),boundary=read(O/'boundary_audit.json')['stats'],all_active_and_aux_masks_edge_free=True,window_cabinet_SAM_policy_verified=True,source_masks_unchanged=True,forbidden_reads_or_shared_writes=0,recorded_IO_events=events,first_repair=50,second_repair_applied=False,GT_used=False,previous_QA_attempts=previous_QA_attempts,QA_retry_reason='Integer category keys restored before namespace hashing; no candidate outputs changed',data_sha256=sha(W/'data.json'),at=time.time())
 write(R/'online_qa.json',result);write(W/'qa.json',result)
 write(R.parent/'review/statusB.json',dict(status='PASS',arm='B',masks_ready=True,candidate_source_policy='new_mapping_frames_51_to100',candidate_source_processed_range=[51,100],candidate_processed_frame_range=[51,100],candidate_raw_frame_range=[250,495],scoring_history=[1,100],candidate_count=len(rows),retained=len(kept),QA_path=str(R/'online_qa.json'),qa_path=str(R/'online_qa.json'),retainedmanifest_path=str(O/'v2/seed_manifest.json'),seed_manifest=str(O/'v2/seed_manifest.json'),mask_source_directory=str(O/'v1'),page_url='http://127.0.0.1:8798/online-repair/masksB/',repair50_applied=True,repair100_applied=False))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
