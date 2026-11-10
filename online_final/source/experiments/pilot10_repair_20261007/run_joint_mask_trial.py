"""Paired full-observation trial; freezes GT-free gates before scoring."""
from pathlib import Path
from dataclasses import replace
import gc,json,time,traceback
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import run_repairs as r
from joint_vote_ops import keys_from_pairs,corrected_frame,batch_delta,self_test

UID='room2-14047feb0df3aaa5'
WORK=r.ROOT/'joint_mask_replacement_20261007/room2_mask14_mask24'
HERE=Path(__file__).parent
ARMS=['pixel_control','whole_mask']

def progress(phase,**extra):
    d={'status':'RUNNING','phase':phase,**extra};r.dump(WORK/'status.json',d)
    print(json.dumps(d,ensure_ascii=False),flush=True)

def aggregate_frames(arrays):
    unique,counts=np.unique(np.concatenate(arrays),return_counts=True)
    return unique,counts.astype(np.int64)

def visible_seed_samples(scene,frame,points):
    xyz=scene.xyz[points]
    pc=(xyz-frame.camera_to_world[:3,3])@frame.camera_to_world[:3,:3]
    good=(pc[:,2]>.05)&(pc[:,2]<10)
    pc=pc[good]
    if not len(pc):return np.empty(0,int),np.empty(0,int)
    u=np.rint(frame.camera.fx*pc[:,0]/pc[:,2]+frame.camera.cx).astype(int)
    v=np.rint(frame.camera.fy*pc[:,1]/pc[:,2]+frame.camera.cy).astype(int)
    h,w=frame.mask_local.shape
    inside=(u>=0)&(u<w)&(v>=0)&(v<h)
    u,v,pc=u[inside],v[inside],pc[inside]
    d=frame.depth_m[v,u]
    seen=(d>0)&np.isfinite(d)&(np.abs(d-pc[:,2])<=POLICY['seed_surface_visibility_depth_tolerance_m'])
    return u[seen],v[seen]

def bbox_inside(xyz,bounds):
    return np.any(np.stack([np.all((xyz>=lo)&(xyz<=hi),axis=1) for lo,hi in bounds]),axis=0) if len(xyz) else np.ones(0,bool)

def build_repairs(seed,tracking):
    start=time.monotonic()
    scene=r.Scene('room2');scene.baseline_control()
    association=scene.prepare_case(seed)
    fixed={o['track_id']:o['persistent_id'] for o in association['objects'] if o['track_id'] in [4,7]}
    assert fixed=={int(k):v for k,v in POLICY['persistent_ids'].items()}
    base=association['instance_base']
    report_path=WORK/'repair_summary.json'
    if report_path.exists():
        saved=json.loads(report_path.read_text())
        assert saved['driver_sha256']==r.sha(__file__) and saved['policy_sha256']==r.sha(HERE/'joint_mask_policy.json')
        for arm,d in saved['arms'].items():assert r.sha(WORK/arm/'final/instance_surface.npz')==d['final_map_sha256']
        return saved
    support=r.load_npz(r.ROOT/'repair'/UID/'seed_projected_support.npz')
    samples={}
    for oid in np.unique(support['track_id']):
        p=np.unique(support['surface_point_index'][support['track_id']==oid])
        samples[int(oid)]=p[np.linspace(0,len(p)-1,min(len(p),2048),dtype=int)]
    bounds={}
    for oid in fixed:
        p=np.unique(support['surface_point_index'][support['track_id']==oid]);xyz=scene.xyz[p]
        bounds[oid]=(xyz.min(0)-POLICY['seed_surface_bbox_padding_m'],xyz.max(0)+POLICY['seed_surface_bbox_padding_m'])
    original_keys=scene.pair['surface_point_index'].astype(np.int64)*base+scene.pair['instance_id']
    order=np.argsort(original_keys,kind='stable')
    original_keys,original_votes=original_keys[order],scene.pair['frame_votes'][order].astype(np.int64)
    arrays={arm:[] for arm in ARMS};old_arrays=[]
    removed={arm:[] for arm in ARMS};added={arm:[] for arm in ARMS};scopes={arm:[] for arm in ARMS}
    checks=[];accepted=[];retired=[];new_support=[]
    tracked={x['frame']:x for x in tracking['frames']}
    assert sorted(tracked)==list(range(2000))
    for index,fid in enumerate(sorted(scene.records)):
        frame,table,old_keys,_=scene.old_frame(fid,base)
        old_arrays.append(old_keys)
        cached=r.load_npz(Path(scene.records[fid]['support_file']))
        op,ol=cached['surface_point_index'],cached['mask_local_id']
        row=tracked[fid];label_file=Path(row['label_file'])
        assert r.sha(label_file)==row['label_sha256']
        labels=np.array(Image.open(label_file),np.uint16)
        assert set(np.unique(labels)).issubset({0,4,7})
        union=labels>0
        np_,nt,_=r.project_frame_regions(replace(frame,mask_local=labels),scene.tree,
                                      pixel_stride=2,max_distance_m=.015,workers=8)
        new_ids=np.array([fixed[int(o)] for o in nt],np.int32)
        new_keys=keys_from_pairs(np_,new_ids,base)
        reasons=[];target_checks={}
        visible={}
        for oid,pts in samples.items():visible[oid]=visible_seed_samples(scene,frame,pts)
        for oid in fixed:
            u,v=visible[oid]
            coverage=float(np.mean(labels[v,u]==oid)) if len(u) else 0.
            npoints=np_[nt==oid]
            spatial=float(np.mean(bbox_inside(scene.xyz[npoints],[bounds[oid]]))) if len(npoints) else 0.
            target_checks[oid]={'visible_seed_samples':len(u),'tracked_coverage':coverage,'new_surface_bbox_fraction':spatial}
            if len(u)<POLICY['minimum_visible_seed_surface_samples_each_target']:reasons.append(f'target{oid}_insufficient_visible_anchor')
            if coverage<POLICY['minimum_tracked_coverage_of_visible_seed_samples_each_target']:reasons.append(f'target{oid}_anchor_coverage')
            if spatial<POLICY['minimum_new_surface_support_fraction_inside_target_bbox']:reasons.append(f'target{oid}_spatial_extent')
        protected=[]
        for oid,(u,v) in visible.items():
            if oid in fixed:continue
            coverage=float(np.mean(union[v,u])) if len(u) else 0.
            if len(u)>=POLICY['minimum_visible_protected_seed_samples'] and coverage>POLICY['maximum_new_union_coverage_of_other_visible_seed_samples']:
                reasons.append(f'protected_seed{oid}_overlap');protected.append({'track_id':oid,'coverage':coverage})
        family_local=[int(mid) for mid in np.unique(ol) if int(table[mid]) in POLICY['old_observation_family_instance_ids']]
        family=np.isin(ol,family_local)
        family_spatial=float(np.mean(bbox_inside(scene.xyz[op[family]],list(bounds.values())))) if np.any(family) else 0.
        if not family_local:reasons.append('no_identified_old_family_mask')
        if family_spatial<POLICY['minimum_old_surface_support_fraction_inside_joint_bbox']:reasons.append('old_family_extends_outside_target_boxes')
        foreign=union & (frame.mask_local>0) & ~np.isin(frame.mask_local,family_local)
        foreign_fraction=float(np.sum(foreign)/max(1,np.sum(union)))
        if foreign_fraction>POLICY['maximum_new_mask_pixel_overlap_fraction_with_other_old_instance_families']:reasons.append('new_mask_overlaps_other_old_families')
        accept=not reasons
        check={'frame':fid,'accepted':accept,'reasons':reasons,'targets':target_checks,'old_local_masks':family_local,
            'old_family_spatial_fraction':family_spatial,'foreign_old_mask_pixel_fraction':foreign_fraction,'protected_seed_overlaps':protected,
            'tracking_label_sha256':row['label_sha256']}
        checks.append(check)
        if accept:
            accepted.append(fid)
            whole,untouched=corrected_frame(op,ol,table,family_local,np_,new_ids,base)
            expected_untouched=r.frame_instance_keys(op[~family],ol[~family],table,base)
            np.testing.assert_array_equal(untouched,expected_untouched)
            assert not np.any(np.isin(ol[~family],family_local))
            partial=frame.mask_local.copy()
            partial[np.isin(partial,family_local)&union]=0
            pp,pl,_=r.project_frame_regions(replace(frame,mask_local=partial),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
            partial_keys=r.frame_instance_keys(pp,pl,table,base)
            pixel=np.union1d(partial_keys,new_keys)
            effective={'pixel_control':pixel,'whole_mask':whole}
            affected_scope=np.unique(np.r_[op[family],np_])
            for arm,new in effective.items():
                assert np.all(np.isin(np.setxor1d(old_keys,new)//base,affected_scope))
                assert np.all(np.isin(untouched,new))
                arrays[arm].append(new);removed[arm].append(np.setdiff1d(old_keys,new));added[arm].append(np.setdiff1d(new,old_keys));scopes[arm].append(affected_scope)
                output=WORK/arm/'transactions'/f'f{fid:06d}.npz';output.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output,old_frame_keys=old_keys,new_frame_keys=new,retired_old_local_masks=family_local,
                    retained_old_frame_keys=untouched,instance_base=base)
            for mid in family_local:
                retired.append({'frame':fid,'old_local_mask':mid,'old_persistent_id':int(table[mid]),
                    'old_surface_region_pairs':int(np.sum(ol==mid)),'old_observation_version_retired':True})
            new_support.append({'frame':fid,'new_projected_region_pairs':len(np_),'new_frame_instance_votes':len(new_keys)})
        else:
            for arm in ARMS:arrays[arm].append(old_keys)
        if (index+1)%40==0:
            progress('match_and_verify_observations',completed_mapping_frames=index+1,accepted_mapping_frames=len(accepted),elapsed_seconds=round(time.monotonic()-start,2))
    # Independent replay validates the entire source ledger and both updated ledgers.
    baseline_replay_k,baseline_replay_v=aggregate_frames(old_arrays)
    np.testing.assert_array_equal(baseline_replay_k,original_keys);np.testing.assert_array_equal(baseline_replay_v,original_votes)
    del old_arrays,baseline_replay_k,baseline_replay_v
    r.dump(WORK/'frame_decisions.json',{'status':'PASS','policy_sha256':r.sha(HERE/'joint_mask_policy.json'),
        'accepted_frames':accepted,'rejected_frames':400-len(accepted),'frames':checks,'retired_observations':retired,'new_support':new_support})
    arm_reports={}
    for arm in ARMS:
        progress('vote_replay_and_postprocessing',arm=arm,accepted_mapping_frames=len(accepted))
        rem=np.concatenate(removed[arm]) if removed[arm] else np.empty(0,np.int64)
        add=np.concatenate(added[arm]) if added[arm] else np.empty(0,np.int64)
        k,v=batch_delta(original_keys,original_votes,rem,add)
        replay_k,replay_v=aggregate_frames(arrays[arm])
        np.testing.assert_array_equal(k,replay_k);np.testing.assert_array_equal(v,replay_v)
        back_k,back_v=batch_delta(k,v,add,rem)
        np.testing.assert_array_equal(back_k,original_keys);np.testing.assert_array_equal(back_v,original_votes)
        del replay_k,replay_v,back_k,back_v
        scope=np.unique(np.concatenate(scopes[arm])) if scopes[arm] else np.empty(0,np.int32)
        evidence=r.reduce_surface_votes(k,v,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
        native=np.where(evidence['state']==r.CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32)
        keep=np.ones(len(scene.xyz),bool);keep[scope]=False
        for field in r.FIELDS:np.testing.assert_array_equal(evidence[field][keep],scene.evidence[field][keep])
        pair={'surface_point_index':(k//base).astype(np.int32),'instance_id':(k%base).astype(np.int32),'frame_votes':v.astype(np.int32)}
        if np.array_equal(k,original_keys) and np.array_equal(v,original_votes):final=scene.final.copy();stats={'identical_baseline':True}
        else:
            variants,_,stats=r.assign_surface(scene.xyz,scene.normals,scene.rgb,evidence,pair,native,scene.settings);final=variants['holes_geodesic']
        out=WORK/arm
        for folder in ['native','final']:(out/folder).mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**evidence)
        np.savez_compressed(out/'native/surface_instance_frame_votes.npz',**pair)
        np.savez_compressed(out/'native/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=native)
        np.savez_compressed(out/'final/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=final)
        np.save(out/'intervention_surface_points.npy',scope)
        object_rows=[]
        for oid,pid in fixed.items():
            p=np.unique(support['surface_point_index'][support['track_id']==oid])
            object_rows.append({'track_id':oid,'native_mask_id':14 if oid==4 else 24,'persistent_id':pid,
                'seed_surface_identity_fraction':float(np.mean(final[p]==pid)),'full_scene_identity_points':int(np.sum(final==pid))})
        arm_reports[arm]={'status':'PASS','final_map_sha256':r.sha(out/'final/instance_surface.npz'),
            'complete_400_frame_replay_bit_identical_to_delta_ledger':True,'exact_rollback_pass':True,
            'unretired_old_observation_supports_preserved_exactly':True,'outside_scope_raw_evidence_bit_identical':True,
            'accepted_mapping_frames':len(accepted),'retired_old_observations':len(retired),'intervention_surface_points':len(scope),
            'native_label_changes':int(np.sum(native!=scene.original)),'final_label_changes':int(np.sum(final!=scene.final)),
            'postprocessing_changes_outside_raw_scope':int(np.sum((final!=scene.final)&keep)),
            'objects':object_rows,'postprocessing_statistics':stats}
        r.dump(out/'complete.json',arm_reports[arm])
    scene.verify_unchanged()
    report={'status':'PASS','driver_sha256':r.sha(__file__),'vote_ops_sha256':r.sha(HERE/'joint_vote_ops.py'),
        'policy_sha256':r.sha(HERE/'joint_mask_policy.json'),'tracking_sha256':r.sha(WORK/'tracking/complete.json'),
        'base_source_hashes':scene.source_hashes,'postprocessing_settings_sha256':scene.settings_sha,
        'association_sha256':r.sha(r.ROOT/'repair'/UID/'frozen_association.json'),'GT_used_for_repair':False,
        'original_inputs_unchanged':True,'accepted_mapping_frames':len(accepted),'total_mapping_frames':400,
        'arms':arm_reports,'seconds':round(time.monotonic()-start,2)}
    r.dump(report_path,report)
    return report

def score(seed,repair):
    import evaluate_pilot_v3 as e
    old_freeze=json.loads((r.ROOT/'v3/evaluation_freeze.json').read_text())
    assert old_freeze['flags']==e.FLAGS and old_freeze['code_sha256']==r.sha(e.__file__)
    freeze={'status':'FROZEN_BEFORE_GT','policy_sha256':repair['policy_sha256'],'driver_sha256':r.sha(__file__),
        'evaluator_sha256':r.sha(e.__file__),'protocol_sha256':r.sha(e.PROTOCOL),'flags':e.FLAGS,
        'effective_protocol':old_freeze['effective_protocol'],'GT_used_for_repair':False,
        'predictions':{a:{'path':str(WORK/a/'final/instance_surface.npz'),'sha256':repair['arms'][a]['final_map_sha256']} for a in ARMS}}
    r.dump(WORK/'evaluation_freeze.json',freeze)
    parser=e.ArgumentParser();e.add_protocol_args(parser)
    protocol,raw,debug=e.load_protocol(e.PROTOCOL,parser.parse_args(['--config',str(e.PROTOCOL),*e.FLAGS]))
    assert raw==old_freeze['effective_protocol']
    base=json.loads((r.ROOT/'v3/baseline/room2/complete.json').read_text())
    assert base['source_map_sha256']==r.sha(r.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz')
    before=e.per_gt(e.load_overlap(r.ROOT/'v3/baseline/room2/overlap.npz'))
    prior=json.loads((r.ROOT/'v3'/UID/'case_comparison.json').read_text())
    target_source=[x for x in prior['target_source'] if x['track_id'] in [4,7]]
    targets=sorted({x['GT_id'] for row in target_source for x in row['GT_overlap'] if x['included']})
    arms={}
    for arm in ARMS:
        progress('unified_v3_evaluation',arm=arm)
        current=e.evaluate('room2',WORK/arm/'final/instance_surface.npz',WORK/'v3'/arm,protocol)
        for field in ['flags','protocol_sha256','evaluation_code_sha256','GT_sha256']:assert current[field]==base[field]
        after=e.per_gt(e.load_overlap(WORK/'v3'/arm/'overlap.npz'))
        iou_delta=float(np.mean([after[g]['best_IoU']-before[g]['best_IoU'] for g in targets]))
        lost=[g for g in before if g not in targets and before[g]['matched_IoU_gt_0_5'] and not after[g]['matched_IoU_gt_0_5']]
        degraded=[g for g in before if g not in targets and after[g]['best_IoU']<before[g]['best_IoU']-.01]
        structural_delta=sum(int(after[g]['split'])+after[g]['merged_predictions_touching_GT']+after[g]['duplicate_predictions']
            -int(before[g]['split'])-before[g]['merged_predictions_touching_GT']-before[g]['duplicate_predictions'] for g in targets)
        m,b=current['metrics'],base['metrics']
        deltas={'AP':m['CA_AP_uniform']-b['CA_AP_uniform'],'AP50':m['CA_AP50_uniform']-b['CA_AP50_uniform'],
            'F1':m['CA_PRF1_0_5']['F1']-b['CA_PRF1_0_5']['F1'],'PQ':m['CA_PQ']['PQ']-b['CA_PQ']['PQ']}
        improvement=(iou_delta>=.01 or structural_delta<0) and iou_delta>=-1e-8
        regression=bool(lost) or iou_delta<-.01 or structural_delta>0
        success=improvement and not regression and not degraded and deltas['F1']>=-1e-8 and deltas['PQ']>=-1e-8
        arms[arm]={'metrics':m,'deltas_from_baseline':deltas,'target_mean_IoU_delta':iou_delta,
            'target_structure_count_delta':structural_delta,'previously_correct_unselected_lost':lost,
            'unselected_IoU_degraded_over_0_01':degraded,'strict_success':bool(success),
            'target_objects':[{'before':before[g],'after':after[g]} for g in targets],
            'repair':repair['arms'][arm],'evaluation_seconds':current['seconds']}
    # Geometry-only upper bound is post-hoc diagnostic; never used by matching/gates.
    gt=e.load_gt(r.INPUT/'room2/ground_truth/gt.npz')
    with np.load(r.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as z:xyz=z['xyz_m']
    distance,_=cKDTree(xyz).query(gt.xyz_ref,workers=8)
    upper={int(g):float(np.mean(distance[(gt.instance_id==g)&gt.valid_vertex_mask&~gt.ignore_vertex_mask]<.01)) for g in targets}
    for row in freeze['predictions'].values():assert r.sha(row['path'])==row['sha256']
    for path,digest in repair['base_source_hashes'].items():assert r.sha(path)==digest
    assert r.sha(r.SEEDS/seed['seed_file'])==seed['seed_sha256']
    summary={'status':'PASS','scope':'room2 mask14/mask24 only; standalone joint tracking; paired accepted frames; full old frame-mask observation retirement',
        'metric_status':base['metrics']['status'],'formal_benchmark_result':False,'flags':e.FLAGS,
        'policy_sha256':repair['policy_sha256'],'targets':targets,'target_source':target_source,
        'baseline_metrics':base['metrics'],'arms':arms,'accepted_mapping_frames':repair['accepted_mapping_frames'],
        'total_mapping_frames':400,'tracking_frames':2000,'tracking_seconds':json.loads((WORK/'tracking/status.json').read_text())['elapsed_seconds'],
        'repair_seconds':repair['seconds'],'surface_geometry_1cm_reachability_upper_bound_by_GT':upper,
        'GT_only_used_after_predictions_frozen':True,'baseline_and_original_annotations_unchanged':True,
        'paired_control_differs_only_in_old_family_retirement_extent':True}
    r.dump(WORK/'v3_summary.json',summary)
    r.dump(WORK/'status.json',{'status':'PASS','phase':'complete','accepted_mapping_frames':repair['accepted_mapping_frames'],
        'strict_success':arms['whole_mask']['strict_success'],'metric_status':summary['metric_status']})
    print(json.dumps({'status':'PASS','accepted_frames':repair['accepted_mapping_frames'],
        'whole_mask_deltas':arms['whole_mask']['deltas_from_baseline'],'strict_success':arms['whole_mask']['strict_success']},ensure_ascii=False),flush=True)

def main():
    global POLICY
    self_test();WORK.mkdir(parents=True,exist_ok=True)
    POLICY=json.loads((HERE/'joint_mask_policy.json').read_text())
    frozen=WORK/'repair_policy_freeze.json'
    if frozen.exists():assert json.loads(frozen.read_text())['policy_sha256']==r.sha(HERE/'joint_mask_policy.json')
    else:r.dump(frozen,{'status':'FROZEN_BEFORE_REPAIR_AND_GT','policy':POLICY,'policy_sha256':r.sha(HERE/'joint_mask_policy.json'),
        'driver_sha256':r.sha(__file__),'vote_ops_sha256':r.sha(HERE/'joint_vote_ops.py'),'GT_used':False})
    if (WORK/'v3_summary.json').exists():print('Completed trial preserved',flush=True);return
    while not (WORK/'tracking/complete.json').exists():
        if (WORK/'tracking/failure.json').exists():raise RuntimeError('Joint tracking failed')
        progress('wait_for_joint_tracking');time.sleep(10)
    tracking=json.loads((WORK/'tracking/complete.json').read_text())
    assert tracking['status']=='PASS' and tracking['completed_frames']==2000 and tracking['track_ids']==[4,7]
    assert tracking['seed_preserved_exactly_for_both_targets'] and tracking['directions_use_independent_seed_only_states']
    seed=next(s for s in json.loads((r.SEEDS/'seed_manifest.json').read_text())['seeds'] if s['case_uid']==UID)
    assert r.sha(r.SEEDS/seed['seed_file'])==seed['seed_sha256']==tracking['original_seed_sha256']
    progress('load_immutable_baseline')
    repair=build_repairs(seed,tracking);gc.collect();score(seed,repair)

if __name__=='__main__':
    try:main()
    except Exception:
        r.dump(WORK/'failure.json',{'status':'FAIL','traceback':traceback.format_exc()})
        r.dump(WORK/'status.json',{'status':'FAIL','phase':'failure'});raise
