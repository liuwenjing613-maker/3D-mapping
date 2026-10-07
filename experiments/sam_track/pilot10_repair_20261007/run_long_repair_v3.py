"""Frozen room2 full-sequence trial: selected-pixel vote replacement and unified v3."""
from pathlib import Path
from dataclasses import replace
import gc, json, os, time, traceback
import numpy as np
from PIL import Image
import run_repairs as r
from evidence_replacement import EvidenceReplacement
from repair_association import compose_frame_mask

UID='room2-14047feb0df3aaa5'
LONG=r.ROOT/'long_tracking_20261007'
WORK=LONG/'repair_v3_20261007'/UID
CONDITION='seed_plus_full_track'


def batch_replace(keys,votes,removed,added):
    # Intersection votes cancel within each complete deduplicated frame.
    # Validate all original retractions before adding any replacement evidence.
    k,v=EvidenceReplacement.add_delta(keys,votes,removed,-1)
    return EvidenceReplacement.add_delta(k,v,added,1)


def check_batch_equivalence():
    old=[np.array(a,np.int64) for a in [[1,2,7],[1,3,7],[2,3,8]]]
    new=[np.array(a,np.int64) for a in [[1,4,7],[1,4,8],[2,3,8]]]
    keys,votes=np.unique(np.concatenate(old),return_counts=True)
    ledger=EvidenceReplacement(keys,votes)
    removed=np.concatenate([np.setdiff1d(a,b) for a,b in zip(old,new)])
    added=np.concatenate([np.setdiff1d(b,a) for a,b in zip(old,new)])
    for i,(a,b) in enumerate(zip(old,new)):
        assert ledger.replace(str(i),a,b)
        assert not ledger.replace(str(i),a,b)
    k,v=batch_replace(keys,votes,removed,added)
    np.testing.assert_array_equal(k,ledger.keys)
    np.testing.assert_array_equal(v,ledger.votes)
    back_k,back_v=batch_replace(k,v,added,removed)
    np.testing.assert_array_equal(back_k,keys)
    np.testing.assert_array_equal(back_v,votes)
    try:
        batch_replace(keys,votes,np.array([999]),np.array([999]))
    except ValueError:
        pass
    else:
        raise AssertionError('Missing old evidence must fail before insertion')


def progress(phase,**extra):
    status={'status':'RUNNING','case_uid':UID,'phase':phase,**extra}
    r.dump(WORK/'status.json',status)
    print(json.dumps(status,ensure_ascii=False),flush=True)


def repair_full(seed,tracking):
    destination=WORK/'repair'
    complete=destination/'complete.json'
    if complete.exists():
        report=json.loads(complete.read_text())
        assert report['driver_sha256']==r.sha(__file__)
        assert r.sha(destination/'final/instance_surface.npz')==report['final_map_sha256']
        return report
    start=time.monotonic()
    progress('load_and_verify_baseline',total_mapping_frames=400,completed_mapping_frames=0)
    scene=r.Scene('room2')
    control=scene.baseline_control()
    association=scene.prepare_case(seed)
    frozen_ids={o['track_id']:o['persistent_id'] for o in association['objects']}
    base=association['instance_base']
    all_keys=scene.pair['surface_point_index'].astype(np.int64)*base+scene.pair['instance_id']
    order=np.argsort(all_keys,kind='stable')
    baseline_keys=all_keys[order]
    baseline_votes=scene.pair['frame_votes'][order].astype(np.int64)
    del all_keys,order
    original_digest=EvidenceReplacement(baseline_keys,baseline_votes).digest()
    tracked={o['frame']:o for o in tracking['frames']}
    assert sorted(tracked)==list(range(2000))
    frames=sorted(scene.records)
    assert frames==list(range(0,2000,5))
    removals,additions,scopes,transactions=[],[],[],[]
    seen_transactions=set()
    destination.mkdir(parents=True,exist_ok=True)
    fid=seed['source_choice']['frame']
    native_seed=np.array(Image.open(r.SEEDS/seed['seed_file']),np.uint16)
    for index,frame_id in enumerate(frames):
        frame,old_table,old_keys,_=scene.old_frame(frame_id,base)
        row=tracked[frame_id]
        selected_file=Path(row['label_file'])
        assert r.sha(selected_file)==row['label_sha256']
        selected=np.array(Image.open(selected_file),np.uint16)
        if frame_id==fid:
            np.testing.assert_array_equal(selected,native_seed)
        combined,table=compose_frame_mask(frame.mask_local,selected,old_table,frozen_ids)
        points,local,stats=r.project_frame_regions(replace(frame,mask_local=combined),scene.tree,
                                                 pixel_stride=2,max_distance_m=.015,workers=8)
        new_keys=r.frame_instance_keys(points,local,table,base)
        selected_points=np.unique(points[local>int(frame.mask_local.max())])
        removed=np.setdiff1d(old_keys,new_keys)
        added=np.setdiff1d(new_keys,old_keys)
        changed=np.concatenate([removed,added])
        assert np.all(np.isin(changed//base,selected_points))
        tx=f'{UID}:{CONDITION}:f{frame_id:06d}'
        assert tx not in seen_transactions
        seen_transactions.add(tx)
        tx_file=destination/f'f{frame_id:06d}_transaction.npz'
        np.savez_compressed(tx_file,removed_keys=removed,added_keys=added,
                            selected_surface_point_index=selected_points,instance_base=base)
        transactions.append({'transaction_id':tx,'frame':frame_id,'old_votes':len(old_keys),
            'new_votes':len(new_keys),'removed_pairs':len(removed),'added_pairs':len(added),
            'scope_surface_points':len(selected_points),'selected_mask_sha256':row['label_sha256'],
            'transaction_sha256':r.sha(tx_file),'signature':EvidenceReplacement.signature(old_keys,new_keys),
            'unselected_pixels_and_outside_scope_votes_unchanged':True})
        removals.append(removed);additions.append(added);scopes.append(selected_points)
        if (index+1)%20==0 or index+1==len(frames):
            progress('project_full_sequence',completed_mapping_frames=index+1,total_mapping_frames=len(frames),
                     elapsed_seconds=round(time.monotonic()-start,2))
    removed_all=np.concatenate(removals)
    added_all=np.concatenate(additions)
    scope=np.unique(np.concatenate(scopes))
    del removals,additions,scopes
    progress('replace_votes_and_verify_rollback',completed_mapping_frames=400,total_mapping_frames=400)
    repaired_keys,repaired_votes=batch_replace(baseline_keys,baseline_votes,removed_all,added_all)
    back_keys,back_votes=batch_replace(repaired_keys,repaired_votes,added_all,removed_all)
    np.testing.assert_array_equal(back_keys,baseline_keys)
    np.testing.assert_array_equal(back_votes,baseline_votes)
    del back_keys,back_votes,removed_all,added_all
    repaired_digest=EvidenceReplacement(repaired_keys,repaired_votes).digest()
    evidence=r.reduce_surface_votes(repaired_keys,repaired_votes,len(scene.xyz),base,
        scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
    native_labels=np.where(evidence['state']==r.CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32)
    keep=np.ones(len(scene.xyz),bool);keep[scope]=False
    for field in r.FIELDS:
        np.testing.assert_array_equal(evidence[field][keep],scene.evidence[field][keep])
    pair={'surface_point_index':(repaired_keys//base).astype(np.int32),
          'instance_id':(repaired_keys%base).astype(np.int32),'frame_votes':repaired_votes.astype(np.int32)}
    progress('fixed_3D_postprocessing',completed_mapping_frames=400,total_mapping_frames=400,
             intervention_surface_points=len(scope),elapsed_seconds=round(time.monotonic()-start,2))
    variants,_,stats=r.assign_surface(scene.xyz,scene.normals,scene.rgb,evidence,pair,native_labels,scene.settings)
    final=variants['holes_geodesic']
    for folder in ['native','final']:
        (destination/folder).mkdir(exist_ok=True)
    np.savez_compressed(destination/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**evidence)
    np.savez_compressed(destination/'native/surface_instance_frame_votes.npz',**pair)
    np.savez_compressed(destination/'native/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=native_labels)
    np.savez_compressed(destination/'final/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=final)
    np.save(destination/'intervention_surface_points.npy',scope)
    support=r.load_npz(r.ROOT/'repair'/UID/'seed_projected_support.npz')
    object_stats=[]
    for obj in association['objects']:
        oid,pid=obj['track_id'],obj['persistent_id']
        seed_points=np.unique(support['surface_point_index'][support['track_id']==oid])
        desired_keys=seed_points.astype(np.int64)*base+pid
        pos=np.searchsorted(repaired_keys,desired_keys)
        present=(pos<len(repaired_keys)) & (repaired_keys[np.minimum(pos,len(repaired_keys)-1)]==desired_keys)
        desired_votes=np.zeros(len(pos),np.int64);desired_votes[present]=repaired_votes[pos[present]]
        object_stats.append({'track_id':oid,'native_mask_id':obj['native_mask_id'],'persistent_id':pid,
            'seed_surface_points':len(seed_points),'baseline_seed_identity_fraction':float(np.mean(scene.final[seed_points]==pid)),
            'full_track_seed_identity_fraction':float(np.mean(final[seed_points]==pid)),
            'final_full_scene_identity_points':int(np.sum(final==pid)),
            'baseline_top_vote_median_on_seed_support':float(np.median(scene.evidence['top1_votes'][seed_points])),
            'long_desired_identity_vote_median_on_seed_support':float(np.median(desired_votes))})
    scene.verify_unchanged()
    report={'status':'PASS','case_uid':UID,'scene':'room2','ROI':seed['ROI'],'condition':CONDITION,
        'driver_sha256':r.sha(__file__),'unchanged_repair_helpers_sha256':r.code_hashes(),
        'seed_sha256':seed['seed_sha256'],'tracking_complete_sha256':r.sha(LONG/UID/'complete.json'),
        'association_sha256':r.sha(r.ROOT/'repair'/UID/'frozen_association.json'),
        'postprocessing_settings_sha256':scene.settings_sha,'base_source_hashes':scene.source_hashes,
        'baseline_control_reused_and_verified':control['final_map_bit_identical'],
        'tracking_frames':2000,'mapping_frames':frames,'mapping_frame_count':len(frames),
        'frames_selected_by_GT_or_quality':False,'tracking_quality_filter_applied':False,
        'scope':'replace selected pixels; inherit every unselected original pixel; deduplicate complete frame votes',
        'vote_weight_per_frame_surface_identity':1,'instance_base':base,'transactions':transactions,
        'batch_equivalent_to_complete_frame_replacement':True,'rollback_restores_original_votes_exactly':True,
        'duplicate_transaction_ids_rejected':True,'baseline_ledger_sha256':original_digest,
        'repaired_ledger_sha256':repaired_digest,'intervention_surface_points':len(scope),
        'native_label_changes':int(np.sum(native_labels!=scene.original)),
        'final_label_changes':int(np.sum(final!=scene.final)),
        'final_changes_outside_intervention':int(np.sum((final!=scene.final)&keep)),
        'unselected_original_pixels_and_outside_scope_raw_votes_unchanged':True,
        'original_inputs_unchanged':True,'GT_used_for_repair':False,
        'final_map_sha256':r.sha(destination/'final/instance_surface.npz'),
        'objects':object_stats,'postprocessing_statistics':stats,'seconds':round(time.monotonic()-start,2)}
    r.dump(complete,report)
    print(json.dumps({k:report[k] for k in ['status','condition','mapping_frame_count','final_label_changes','seconds']},ensure_ascii=False),flush=True)
    return report


def evaluate_full(seed,repair):
    # No GT is loaded before the repaired prediction and its inputs are frozen.
    import evaluate_pilot_v3 as e
    surface=WORK/'repair/final/instance_surface.npz'
    old_freeze=json.loads((r.ROOT/'v3/evaluation_freeze.json').read_text())
    assert old_freeze['code_sha256']==e.sha(e.__file__)
    assert old_freeze['flags']==e.FLAGS
    assert e.sha(e.PROTOCOL)=='0863fc50a69b8773988bc588a4b71253fdd6caf09e65ab2217195894fcf79ede'
    for key in ['baseline:room2',UID+':seed_only',UID+':seed_plus_short_track']:
        row=old_freeze['predictions'][key]
        assert r.sha(row['path'])==row['sha256']
    frozen={'status':'FROZEN_BEFORE_GT','driver_sha256':r.sha(__file__),
        'unchanged_evaluator_sha256':r.sha(e.__file__),'source_map_sha256':r.sha(surface),
        'long_tracking_sha256':r.sha(LONG/UID/'complete.json'),'human_manifest_sha256':r.sha(r.SEEDS/'seed_manifest.json'),
        'association_sha256':repair['association_sha256'],'previous_evaluation_freeze_sha256':r.sha(r.ROOT/'v3/evaluation_freeze.json'),
        'protocol_sha256':r.sha(e.PROTOCOL),'flags':e.FLAGS,'effective_protocol':old_freeze['effective_protocol'],
        'GT_used_for_repair':False,'frames_filtered_using_GT':False}
    r.dump(WORK/'evaluation_freeze.json',frozen)
    parser=e.ArgumentParser();e.add_protocol_args(parser)
    protocol,raw,debug=e.load_protocol(e.PROTOCOL,parser.parse_args(['--config',str(e.PROTOCOL),*e.FLAGS]))
    assert raw==old_freeze['effective_protocol']
    progress('unified_v3_evaluation',prediction_frozen_before_GT=True,metric_status='DEBUG_ONLY / NON_OFFICIAL')
    full=e.evaluate('room2',surface,WORK/'v3/full_track',protocol)
    baseline=json.loads((r.ROOT/'v3/baseline/room2/complete.json').read_text())
    for field in ['flags','protocol_sha256','evaluation_code_sha256','GT_sha256']:
        assert baseline[field]==full[field]
    before=e.per_gt(e.load_overlap(r.ROOT/'v3/baseline/room2/overlap.npz'))
    after=e.per_gt(e.load_overlap(WORK/'v3/full_track/overlap.npz'))
    prior=json.loads((r.ROOT/'v3'/UID/'case_comparison.json').read_text())
    targets=prior['targets']
    assert before.keys()==after.keys()
    iou_delta=float(np.mean([after[g]['best_IoU']-before[g]['best_IoU'] for g in targets]))
    lost=[g for g in before if g not in targets and before[g]['matched_IoU_gt_0_5'] and not after[g]['matched_IoU_gt_0_5']]
    degraded=[g for g in before if g not in targets and after[g]['best_IoU']<before[g]['best_IoU']-.01]
    structure_delta=sum(int(after[g]['split'])+after[g]['merged_predictions_touching_GT']+after[g]['duplicate_predictions']
        -int(before[g]['split'])-before[g]['merged_predictions_touching_GT']-before[g]['duplicate_predictions'] for g in targets)
    m,b=full['metrics'],baseline['metrics']
    f1_delta=m['CA_PRF1_0_5']['F1']-b['CA_PRF1_0_5']['F1']
    pq_delta=m['CA_PQ']['PQ']-b['CA_PQ']['PQ']
    improvement=(iou_delta>=.01 or structure_delta<0) and iou_delta>=-1e-8
    regression=bool(lost) or iou_delta<-.01 or structure_delta>0
    success=improvement and not regression and not degraded and f1_delta>=-1e-8 and pq_delta>=-1e-8
    conditions={'baseline':baseline['metrics'],
        'seed_only':prior['conditions']['seed_only']['scene_metrics'],
        'seed_plus_short_track':prior['conditions']['seed_plus_short_track']['scene_metrics'],
        CONDITION:full['metrics']}
    deltas={k:{'AP50':v['CA_AP50_uniform']-b['CA_AP50_uniform'],
        'F1':v['CA_PRF1_0_5']['F1']-b['CA_PRF1_0_5']['F1'],'PQ':v['CA_PQ']['PQ']-b['CA_PQ']['PQ']}
        for k,v in conditions.items()}
    summary={'status':'PASS','scope':'one fixed room2 case; 8 selected masks; all 2000 raw frames; 400 unchanged mapping frames',
        'case_uid':UID,'scene':'room2','ROI':seed['ROI'],'metric_status':m['status'],
        'formal_benchmark_result':False,'flags':e.FLAGS,'protocol_sha256':r.sha(e.PROTOCOL),
        'same_unified_v3_as_baseline_and_short_track':True,'predictions_frozen_before_GT':True,
        'no_GT_used_in_repair_or_frame_selection':True,'conditions':conditions,'deltas_from_baseline':deltas,
        'full_vs_short_deltas':{k:deltas[CONDITION][k]-deltas['seed_plus_short_track'][k] for k in ['AP50','F1','PQ']},
        'targets':targets,'target_source':prior['target_source'],'target_definition':prior['target_definition'],
        'target_mean_IoU_delta':iou_delta,'target_structure_count_delta':structure_delta,
        'previously_correct_unselected_lost':len(lost),'lost_unselected_GT_ids':lost,
        'unselected_IoU_degraded_over_0_01':degraded,
        'target_objects':[{'before':before[g],'after':after[g]} for g in targets],
        'strict_success':bool(success),'strict_success_rule':'same pilot10 rule: target IoU +0.01 or fewer structural errors; no target/unselected/F1/PQ regression',
        'tracking_seconds':tracking_time(),'repair_seconds':repair['seconds'],'evaluation_seconds':full['seconds'],
        'repair_objects':repair['objects'],'repair_label_changes':repair['final_label_changes'],
        'source_map_sha256':repair['final_map_sha256'],'source_maps_and_original_annotations_unchanged':True}
    assert r.sha(surface)==frozen['source_map_sha256']
    for path,digest in repair['base_source_hashes'].items():
        assert r.sha(path)==digest
    assert r.sha(r.SEEDS/seed['seed_file'])==seed['seed_sha256']
    manifest=json.loads((r.SEEDS/'seed_manifest.json').read_text())
    assert r.sha(r.SEEDS/'source_human_export.json')==manifest['source_export_sha256']
    r.dump(WORK/'v3_summary.json',summary)
    r.dump(WORK/'status.json',{'status':'PASS','case_uid':UID,'phase':'complete','strict_success':bool(success),
        'full_track_AP50':m['CA_AP50_uniform'],'full_track_F1':m['CA_PRF1_0_5']['F1'],
        'deltas_from_baseline':deltas[CONDITION],'metric_status':m['status']})
    print(json.dumps({'status':'PASS','deltas':deltas,'strict_success':bool(success)},ensure_ascii=False),flush=True)


def tracking_time():
    return json.loads((LONG/UID/'status.json').read_text())['elapsed_seconds']


def main():
    check_batch_equivalence()
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'v3_summary.json').exists():
        print('Existing completed trial is preserved',flush=True)
        return
    tracking=json.loads((LONG/UID/'complete.json').read_text())
    assert tracking['status']=='PASS' and tracking['completed_frames']==tracking['total_frames']==2000
    assert tracking['seed_preserved_exactly'] and tracking['directions_use_independent_seed_only_states']
    manifest=json.loads((r.SEEDS/'seed_manifest.json').read_text())
    seed=next(s for s in manifest['seeds'] if s['case_uid']==UID)
    assert tracking['seed_sha256']==seed['seed_sha256']==r.sha(r.SEEDS/seed['seed_file'])
    report=repair_full(seed,tracking)
    gc.collect()
    evaluate_full(seed,report)


if __name__=='__main__':
    try:
        main()
    except Exception:
        r.dump(WORK/'failure.json',{'status':'FAIL','traceback':traceback.format_exc()})
        r.dump(WORK/'status.json',{'status':'FAIL','case_uid':UID,'phase':'failure'})
        raise
