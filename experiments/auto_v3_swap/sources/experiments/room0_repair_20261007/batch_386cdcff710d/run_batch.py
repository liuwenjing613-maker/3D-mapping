"""Joint offline replacement: rebuild each original frame once across all cases."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time,traceback
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d')
REPAIR_ROOT=ROOT/'validated'
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'));sys.path.insert(0,str(LEGACY/'objectwise_repair_20261007'))
import run_repairs as r
import fixed_surface_repair as f
import run_case as c
import objectwise_ops as o
from joint_vote_ops import keys_from_pairs,batch_delta
import joint_batch_ops as j

def check_hashes(hashes):
    for path,expected in hashes.items():
        if r.sha(path)!=expected: raise RuntimeError('Frozen input changed: '+path)

def status(out,stage,**fields):
    row={'status':'RUNNING','stage':stage,**fields};f.atomic_json(out/'status.json',row)
    print(json.dumps(row),flush=True)

def inputs():
    scene=r.Scene('room0');scene.baseline_control()
    cases=json.loads((ROOT/'seed_diagnostics.json').read_text())['cases']
    objects={};case_data={}
    policy=json.loads((LEGACY/'objectwise_repair_20261007/policy.json').read_text())
    review=json.loads((ROOT/'alias_review.json').read_text())
    assert review['status']=='FROZEN' and not review['GT_used']
    aliases={int(a):int(b) for a,b in review['aliases'].items()}
    canonical=j.canonical_map([obj['track_id'] for case in cases for obj in case['objects']],aliases)
    for case in cases:
        uid=case['case_uid'];cfg=json.loads(Path(case['config_path']).read_text())
        z=f.load_arrays(ROOT/'cases'/uid/'seed_projected_support.npz')
        case_data[uid]={'case':case,'config':cfg,'ids':[obj['track_id'] for obj in case['objects']],
                        'seed':np.array(Image.open(ROOT/'cases'/uid/'prepared_seed.png'),np.uint16)}
        for obj in case['objects']:
            oid=obj['track_id'];points=np.unique(z['surface_point_index'][z['track_id']==oid])
            samples=points[np.linspace(0,len(points)-1,min(len(points),2048),dtype=int)]
            bounds=(scene.xyz[points].min(0)-policy['seed_surface_bbox_padding_m'],scene.xyz[points].max(0)+policy['seed_surface_bbox_padding_m'])
            objects[oid]={**obj,'case_uid':uid,'canonical_track_id':canonical[oid],'points':points,'samples':samples,'bounds':bounds}
    return scene,case_data,objects,canonical,policy

def freeze_global_association(scene,cases,objects,canonical):
    path=ROOT/'global_association.json'
    # Same per-view visible-surface Dice and >0.5 acceptance as the existing code.
    # Across views of an alias use the best source Dice, then a single one-to-one solve.
    scores={cid:{} for cid in sorted(set(canonical.values()))};view_scores=[]
    for uid,data in cases.items():
        fid=data['case']['source_choice']['frame'];_,_,_,oldvisible=scene.old_frame(fid,scene.maximum_id+max(objects)+1)
        visible=np.unique(np.concatenate([oldvisible,*[objects[oid]['points'] for oid in data['ids']]]))
        oldids,oldn=np.unique(scene.final[visible][scene.final[visible]>0],return_counts=True)
        for oid in data['ids']:
            p=objects[oid]['points'];ids,n=np.unique(scene.final[p],return_counts=True);overlap=dict(zip(ids.tolist(),n.tolist()))
            values={int(pid):2*overlap.get(int(pid),0)/(len(p)+int(count)) for pid,count in zip(oldids,oldn)}
            cid=canonical[oid]
            for pid,value in values.items():scores[cid][pid]=max(scores[cid].get(pid,0),value)
            view_scores.append({'track_id':oid,'canonical_track_id':cid,'source_case':uid,'seed_points':len(p),
                                'visible_dice_candidates':sorted([{'persistent_id':pid,'dice':value} for pid,value in values.items()],key=lambda row:-row['dice'])[:8]})
    cids=sorted(scores);oldids=sorted({pid for values in scores.values() for pid in values})
    matrix=np.full((len(cids),len(oldids)+len(cids)),-1.,np.float64)
    for i,cid in enumerate(cids):
        for k,pid in enumerate(oldids):
            if scores[cid].get(pid,0)>.5:matrix[i,k]=scores[cid][pid]
        matrix[i,len(oldids)+i]=0.
    rr,cc=linear_sum_assignment(-matrix);rows=[]
    for i,k in zip(rr.tolist(),cc.tolist()):
        cid=cids[i];matched=k<len(oldids);pid=oldids[k] if matched else scene.maximum_id+cid
        rows.append({'canonical_track_id':cid,'member_track_ids':[oid for oid in canonical if canonical[oid]==cid],
                     'persistent_id':pid,'association':'visible_surface_dice' if matched else 'distinct_new_identity',
                     'accepted_dice':float(matrix[i,k]) if matched else None})
    assert len(set(row['persistent_id'] for row in rows))==len(rows)
    report={'status':'PASS','GT_used':False,'objects':rows,'source_view_scores':view_scores,
            'aliases':json.loads((ROOT/'alias_review.json').read_text())['aliases'],
            'alias_evidence':json.loads((ROOT/'alias_review.json').read_text()),'threshold_dice':.5,
            'new_identity_rule':'maximum original identity + canonical track ID',
            'instance_base':max([scene.maximum_id,*[row['persistent_id'] for row in rows]])+1}
    if path.exists():assert json.loads(path.read_text())==report
    else:f.atomic_json(path,report)
    return report

def validate_tracking(cases):
    initial=json.loads((ROOT/'input_freeze.json').read_text());check_hashes(initial['seed_and_config_hashes'])
    check_hashes(initial['original_baseline_source_hashes']);check_hashes(initial['raw_rgb_hashes'])
    reports={};proof=[]
    for uid,data in cases.items():
        cfg=data['config'];folder=ROOT/'cases'/uid/'tracking';report=json.loads((folder/'complete.json').read_text())
        assert report['status']=='PASS' and report['completed_frames']==report['total_frames']==2000
        assert report['case_uid']==uid and report['seed_frame']==cfg['seed_frame'] and report['track_ids']==data['ids']
        assert report['case_config_sha256']==r.sha(data['case']['config_path'])
        assert report['code_sha256']==initial['tracking_code_sha256']==r.sha(initial['tracking_code_path'])
        assert report['directions_use_independent_seed_only_states'] and not report['GT_used']
        assert report['checkpoint_sha256']==cfg['tracking']['checkpoint_sha256']==r.sha(cfg['tracking']['checkpoint'])
        frames={row['frame']:row for row in report['frames']};assert sorted(frames)==list(range(2000))
        seedpath=ROOT/'cases'/uid/'prepared_seed.png'
        assert report['seed_sha256']==r.sha(seedpath)
        for fid,row in frames.items():
            assert r.sha(row['label_file'])==row['label_sha256']
            if fid!=cfg['seed_frame']:assert row['direction']==('reverse' if fid<cfg['seed_frame'] else 'forward')
        np.testing.assert_array_equal(np.array(Image.open(frames[cfg['seed_frame']]['label_file'])),data['seed'])
        reports[uid]=frames
        proof.append({'case_uid':uid,'track_ids':data['ids'],'verified_raw_frames':2000,'tracking_seconds':report['elapsed_seconds'],
                      'independent_forward_and_reverse':True,'seed_exact':True,'report_sha256':r.sha(folder/'complete.json')})
    f.atomic_json(ROOT/'tracking_validation.json',{'status':'PASS','GT_used':False,'raw_RGB_files_checked':2000,'label_files_checked':2000*len(cases),'cases':proof})
    return reports

def metrics_for(oid,mask,points,visible,objects,data,frame,family_local,scene,policy):
    u,v=visible[oid];canonical=objects[oid]['canonical_track_id']
    foreign=mask & (frame.mask_local>0) & ~np.isin(frame.mask_local,family_local)
    return {'visible_seed_samples':len(u),'tracked_coverage':float(np.mean(mask[v,u])) if len(u) else 0.,
            'new_surface_bbox_fraction':float(np.mean(c.bbox_inside(scene.xyz[points],[objects[oid]['bounds']]))) if len(points) else 0.,
            'foreign_old_mask_pixel_fraction':float(foreign.sum()/max(1,mask.sum())),
            'other_seed_overlaps':[{'track_id':other,'visible_seed_samples':len(uv[0]),
                'coverage':float(np.mean(mask[uv[1],uv[0]])) if len(uv[0]) else 0.}
                for other,uv in visible.items() if objects[other]['canonical_track_id']!=canonical]}

def repair(condition,scene,cases,objects,canonical,policy,association,tracking):
    out=REPAIR_ROOT/condition;out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    fixed={row['canonical_track_id']:row['persistent_id'] for row in association['objects']};base=association['instance_base']
    initial=json.loads((ROOT/'input_freeze.json').read_text())
    paths=[Path(__file__),HERE/'joint_batch_ops.py',LEGACY/'run_repairs.py',LEGACY/'objectwise_repair_20261007/objectwise_ops.py',
           LEGACY/'objectwise_repair_20261007/run_case.py',LEGACY/'objectwise_repair_20261007/policy.json',
           LEGACY/'strict_local_repair_20261007/fixed_surface_repair.py',LEGACY/'joint_vote_ops.py',
           ROOT/'input_freeze.json',ROOT/'alias_review.json',ROOT/'global_association.json',ROOT/'tracking_validation.json',ROOT/'seed_manifest.json',
           *[ROOT/'cases'/uid/'tracking/complete.json' for uid in cases]]
    protected={**initial['seed_and_config_hashes'],**scene.source_hashes,**{str(p):r.sha(p) for p in paths}}
    freeze={'status':'FROZEN_BEFORE_REPAIR','condition':condition,'annotation_sha256':initial['annotation_sha256'],
            'source_hashes':protected,'GT_used_for_repair':False,'policy_sha256':r.sha(LEGACY/'objectwise_repair_20261007/policy.json'),
            'joint_changes':['GT-free cross-case alias review','one global one-to-one identity assignment',
               'same-object union and pixel overlap abstention','cross-case protected seed anchors',
               'whole retirement additionally requires every selected object sharing that original family',
               'each original frame/observation rebuilt once before persistent vote dedup'],
            'unchanged':['human masks','SAM2.1 independent forward/reverse','all numeric reliability thresholds',
               'vote units/weights','native confirmation','geometry/RGB','postprocessing','strict commit','unified v3']}
    if (out/'source_freeze.json').exists():assert json.loads((out/'source_freeze.json').read_text())==freeze
    else:f.atomic_json(out/'source_freeze.json',freeze)
    if (out/'complete.json').exists():
        done=json.loads((out/'complete.json').read_text());check_hashes(protected);check_hashes(done['output_sha256']);return done
    keys=scene.pair['surface_point_index'].astype(np.int64)*base+scene.pair['instance_id'];order=np.argsort(keys,kind='stable')
    original_keys=keys[order];original_votes=scene.pair['frame_votes'][order].astype(np.int64)
    old_frames=[];new_frames=[];removed=[];added=[];scopes=[];retired_points=[];decisions=[];observations=[]
    counts={'mapping_frames':400,'updated_frames':0,'whole_observations':0,'partial_observations':0,
            'cross_case_conflict_pixels':0,'accepted_frames_by_track':{str(oid):0 for oid in objects},
            'updated_frames_by_case':{uid:0 for uid in cases}}
    family_targets={pid:{canonical[oid] for oid,obj in objects.items() if obj['old_family_id']==pid}
                    for pid in {obj['old_family_id'] for obj in objects.values()}}
    for index,fid in enumerate(sorted(scene.records)):
        frame,table,old_keys,_=scene.old_frame(fid,base);old_frames.append(old_keys)
        cached=r.load_npz(Path(scene.records[fid]['support_file']));op,ol=cached['surface_point_index'],cached['mask_local_id']
        active={uid:data for uid,data in cases.items() if condition=='full_track' or fid==data['config']['seed_frame']}
        if not active:new_frames.append(old_keys);continue
        visible={oid:c.visible_seed(scene,frame,obj['samples'],policy) for oid,obj in objects.items()}
        family_local={uid:[int(mid) for mid in np.unique(ol) if int(table[mid]) in data['config']['old_family_ids']] for uid,data in active.items()}
        labels_by_case={};checks={};accepted=[];projections={}
        for uid,data in active.items():
            labels=np.array(Image.open(tracking[uid][fid]['label_file']),np.uint16) if condition=='full_track' else data['seed']
            assert labels.shape==frame.mask_local.shape and set(np.unique(labels)).issubset({0,*data['ids']})
            labels_by_case[uid]=labels
            p,t,_=r.project_frame_regions(replace(frame,mask_local=labels),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
            projections[uid]=(p,t)
            for oid in data['ids']:
                metric=metrics_for(oid,labels==oid,p[t==oid],visible,objects,data,frame,family_local[uid],scene,policy)
                ok,reasons=o.object_reliability(metric,policy)
                if not family_local[uid]:ok=False;reasons.append('no_identified_old_family_observation')
                checks[str(oid)]={**metric,'accepted':ok,'reasons':reasons}
                if ok:accepted.append(oid)
        if not accepted:new_frames.append(old_keys);decisions.append({'frame':fid,'objects':checks,'accepted_track_ids':[]});continue
        proposals=[(oid,labels_by_case[objects[oid]['case_uid']]==oid) for oid in accepted]
        merged,conflict=j.merge_foregrounds(proposals,canonical)
        counts['cross_case_conflict_pixels']+=int(conflict.sum())
        # Removing ambiguous pixels must still satisfy the unchanged reliability tests.
        if conflict.any():
            revised_accepted=[]
            for uid,data in active.items():
                trimmed=labels_by_case[uid].copy();trimmed[conflict]=0;labels_by_case[uid]=trimmed
                p,t,_=r.project_frame_regions(replace(frame,mask_local=trimmed),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
                for oid in [x for x in accepted if objects[x]['case_uid']==uid]:
                    metric=metrics_for(oid,trimmed==oid,p[t==oid],visible,objects,data,frame,family_local[uid],scene,policy)
                    ok,reasons=o.object_reliability(metric,policy)
                    checks[str(oid)]={**metric,'accepted':ok,'reasons':reasons,'rechecked_after_overlap_abstention':True}
                    if ok:revised_accepted.append(oid)
            accepted=sorted(revised_accepted)
            if accepted:merged,_=j.merge_foregrounds([(oid,labels_by_case[objects[oid]['case_uid']]==oid) for oid in accepted],canonical)
        if not accepted:new_frames.append(old_keys);decisions.append({'frame':fid,'objects':checks,'accepted_track_ids':[]});continue
        accepted_canonical=set(canonical[oid] for oid in accepted)
        case_actions={};case_accepted={};all_mids=set()
        for uid,data in active.items():
            ca=[oid for oid in data['ids'] if oid in accepted];case_accepted[uid]=ca
            if not ca:continue
            counts['updated_frames_by_case'][uid]+=1
            union=np.isin(merged,[canonical[oid] for oid in ca]);own=set(canonical[oid] for oid in data['ids'])
            protected_ok=all(len(uv[0])<policy['minimum_visible_protected_seed_samples'] or
                          float(np.mean(union[uv[1],uv[0]]))<=policy['maximum_new_union_coverage_of_other_visible_seed_samples']
                          for oid,uv in visible.items() if canonical[oid] not in own)
            case_actions[uid]={}
            for mid in family_local[uid]:
                spatial=float(np.mean(c.bbox_inside(scene.xyz[op[ol==mid]],[objects[oid]['bounds'] for oid in data['ids']])))
                case_actions[uid][mid]=o.observation_plan(ca,data['ids'],spatial,protected_ok,policy);all_mids.add(mid)
        plans={}
        for mid in sorted(all_mids):
            pid=int(table[mid]);acts=[actions[mid] for actions in case_actions.values() if mid in actions]
            plans[mid]=j.combined_plan(acts,family_targets.get(pid,set()),accepted_canonical,bool(np.any(conflict & (frame.mask_local==mid))))
        partial=[mid for mid,action in plans.items() if action=='partial']
        residual=np.where(np.isin(frame.mask_local,partial),frame.mask_local,0).astype(frame.mask_local.dtype)
        for mid in partial:
            pid=int(table[mid]);eligible={canonical[oid] for oid in accepted if pid in cases[objects[oid]['case_uid']]['config']['old_family_ids']}
            residual[(frame.mask_local==mid)&np.isin(merged,list(eligible))]=0
        pp,pl,_=r.project_frame_regions(replace(frame,mask_local=residual),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
        np_,nt,_=r.project_frame_regions(replace(frame,mask_local=merged),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
        newids=np.array([fixed[int(cid)] for cid in nt],np.int32)
        revised,retained,retired,scope=o.replace_observations(op,ol,table,plans,pp,pl,np_,newids,base)
        rem,add=np.setdiff1d(old_keys,revised),np.setdiff1d(revised,old_keys)
        assert np.all(np.isin(np.setxor1d(old_keys,revised)//base,scope))
        untouched=~np.isin(ol,list(plans));assert np.all(np.isin(keys_from_pairs(op[untouched],table[ol[untouched]],base),retained))
        counts['updated_frames']+=1
        for oid in accepted:counts['accepted_frames_by_track'][str(oid)]+=1
        new_frames.append(revised);removed.append(rem);added.append(add);scopes.append(scope);retired_points.append(retired[:,0].astype(np.int32))
        transaction=out/'transactions'/('f%06d.npz'%fid)
        f.atomic_npz(transaction,old_frame_keys=old_keys,new_frame_keys=revised,retained_old_frame_keys=retained,
                     retired_surface_local_pairs=retired,accepted_track_ids=np.array(accepted,np.int32),
                     allowed_surface_point_index=scope,instance_base=np.array([base],np.int64))
        for mid,action in plans.items():
            counts[action+'_observations']+=1
            observations.append({'frame':fid,'old_local_mask':mid,'old_persistent_id':int(table[mid]),'action':action,
                                 'original_surface_region_pairs':int(np.sum(ol==mid)),'retired_surface_region_pairs':int(np.sum(retired[:,1]==mid)),
                                 'retained_residual_region_pairs':int(np.sum(pl==mid))})
        decisions.append({'frame':fid,'objects':checks,'accepted_track_ids':accepted,'accepted_canonical_ids':sorted(accepted_canonical),
                          'old_observation_plans':plans,'cross_case_conflict_pixels':int(conflict.sum()),
                          'removed_votes':len(rem),'added_votes':len(add),'transaction_sha256':r.sha(transaction)})
        if (index+1)%40==0:status(out,'joint_frame_rebuild',completed_mapping_frames=index+1,updated_frames=counts['updated_frames'])
    bk,bv=j.count_vote_keys(old_frames);np.testing.assert_array_equal(bk,original_keys);np.testing.assert_array_equal(bv,original_votes)
    del old_frames,bk,bv
    rem=np.concatenate(removed) if removed else np.empty(0,np.int64);add=np.concatenate(added) if added else np.empty(0,np.int64)
    k,v=batch_delta(original_keys,original_votes,rem,add);nk,nv=j.count_vote_keys(new_frames)
    np.testing.assert_array_equal(k,nk);np.testing.assert_array_equal(v,nv)
    backk,backv=batch_delta(k,v,add,rem);np.testing.assert_array_equal(backk,original_keys);np.testing.assert_array_equal(backv,original_votes)
    del new_frames,nk,nv,backk,backv
    scope=np.unique(np.concatenate(scopes)).astype(np.int32) if scopes else np.empty(0,np.int32)
    evidence=r.reduce_surface_votes(k,v,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
    native=np.where(evidence['state']==r.CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32)
    pair={'surface_point_index':(k//base).astype(np.int32),'instance_id':(k%base).astype(np.int32),'frame_votes':v.astype(np.int32)}
    status(out,'postprocess_and_strict_commit',updated_frames=counts['updated_frames'])
    if np.array_equal(k,original_keys) and np.array_equal(v,original_votes):candidate=scene.final.copy();stats={'identical_baseline':True}
    else:
        variants,_,stats=r.assign_surface(scene.xyz,scene.normals,scene.rgb,evidence,pair,native,scene.settings);candidate=variants['holes_geodesic']
    baseline={'xyz_m':scene.xyz,'rgb':scene.rgb,'instance_id':scene.final};proposed={**baseline,'instance_id':candidate}
    labels,commit,blocked=f.bounded_final_labels(baseline,proposed,scope)
    raw={'xyz_m':scene.xyz,'rgb':scene.rgb,**evidence};proof=f.verify_raw_outside(scene.evidence,raw,scene.pair,pair,scope)
    f.atomic_npz(out/'native/surface_evidence.npz',**raw);f.atomic_npz(out/'native/surface_instance_frame_votes.npz',**pair)
    f.atomic_npz(out/'native/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=native)
    f.atomic_npz(out/'candidate/instance_surface.npz',**proposed);f.atomic_npz(out/'final/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=labels)
    f.atomic_npz(out/'allowed_surface_ids.npz',surface_point_index=scope)
    f.atomic_npz(out/'retired_surface_ids.npz',surface_point_index=np.unique(np.concatenate(retired_points)).astype(np.int32) if retired_points else np.empty(0,np.int32))
    f.atomic_npz(out/'blocked_outside_changes.npz',surface_point_index=blocked,unrestricted_label=candidate[blocked],restored_baseline_label=scene.final[blocked])
    f.atomic_json(out/'frame_decisions.json',{'status':'PASS','counts':counts,'frames':decisions,'observations':observations})
    hashes={str(path):r.sha(path) for path in [out/'final/instance_surface.npz',out/'candidate/instance_surface.npz',
            out/'native/surface_evidence.npz',out/'native/surface_instance_frame_votes.npz',out/'allowed_surface_ids.npz',
            out/'retired_surface_ids.npz',out/'blocked_outside_changes.npz',out/'frame_decisions.json']}
    retired=f.load_arrays(out/'retired_surface_ids.npz')['surface_point_index']
    report={'status':'PASS','condition':condition,'scene':'room0','GT_used_for_repair':False,'predictions_frozen_before_any_GT_evaluation':True,
            'complete_frame_replay_bit_identical_to_delta_ledger':True,'exact_rollback_pass':True,
            'unretired_old_observation_supports_preserved_exactly':True,'each_original_frame_rebuilt_once':True,
            'counts':counts,'strict_commit':{**commit,**proof},'scope_definition':'retired observation-region support union accepted new support; no geometric expansion',
            'scope_unknown_before':int(np.sum(scene.final[scope]<=0)),'scope_unknown_after':int(np.sum(labels[scope]<=0)),
            'whole_scene_unknown_before':int(np.sum(scene.final<=0)),'whole_scene_unknown_after':int(np.sum(labels<=0)),
            'assigned_to_unassigned_inside':int(np.sum((scene.final[scope]>0)&(labels[scope]<=0))),
            'unassigned_to_assigned_inside':int(np.sum((scene.final[scope]<=0)&(labels[scope]>0))),
            'retired_support_points':len(retired),'retired_support_final_unknown':int(np.sum(labels[retired]<=0)),
            'retired_support_native_state_counts':{str(state):int(np.sum(evidence['state'][retired]==state)) for state in [0,1,2,3]},
            'retired_support_final_unknown_by_native_state':{str(state):int(np.sum((evidence['state'][retired]==state)&(labels[retired]<=0))) for state in [0,1,2,3]},
            'removed_frame_surface_identity_votes':len(rem),'added_frame_surface_identity_votes':len(add),
            'confirmation_thresholds':{'minimum_votes':scene.report['min_confirmed_votes'],'minimum_ratio':scene.report['min_confirmed_ratio']},
            'postprocessing_settings_sha256':scene.settings_sha,'postprocessing_statistics':stats,'output_sha256':hashes,
            'original_inputs_unchanged':True,'seconds':round(time.monotonic()-started,2)}
    scene.verify_unchanged();check_hashes(protected);check_hashes(initial['raw_rgb_hashes'])
    f.atomic_json(out/'complete.json',report);f.atomic_json(out/'repair_summary.json',report)
    f.atomic_json(out/'status.json',{'status':'PASS','stage':'complete','outside_final_label_changes':commit['final_outside_changes']})
    print(json.dumps({k:report[k] for k in ['status','condition','counts','strict_commit','seconds']}),flush=True)
    return report

def main(identity_only=False):
    scene,cases,objects,canonical,policy=inputs()
    association=freeze_global_association(scene,cases,objects,canonical)
    if identity_only:
        print(json.dumps(association));return
    tracking=validate_tracking(cases)
    for condition in ['seed_only','full_track']:repair(condition,scene,cases,objects,canonical,policy,association,tracking)
    scene.verify_unchanged()
    f.atomic_json(ROOT/'repair_complete.json',{'status':'PASS','conditions':['seed_only','full_track'],
         'outputs_root':str(REPAIR_ROOT),'implementation':'run_batch.py','repair_cases':len(cases),'no_reliable_seed_cases':14-len(cases),'selected_masks':len(objects),'unique_objects':len(association['objects']),
         'GT_used_for_repair':False,'annotation_sha256':r.sha(ROOT/'inputs/human_selection.json')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--identity-only',action='store_true');args=parser.parse_args()
    try:main(args.identity_only)
    except Exception:
        f.atomic_json(ROOT/'failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
