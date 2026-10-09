"""Replay frozen raw mask edits, then strictly reduce complete frame candidates."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys,time,traceback,gc
import numpy as np
from PIL import Image
from strict_vote_ops import strict_frame_keys,frame_delta

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
PRIOR=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'))
import run_repairs as r
import fixed_surface_repair as f
from joint_vote_ops import batch_delta
ROOTS={
 'room0':Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d'),
 'room2':Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')}
CONDITIONS=('seed_only','full_track')
FIELDS=r.FIELDS
hashes={}

def record(path,expected=None):
    path=Path(path);digest=r.sha(path)
    if expected is not None:assert digest==expected,'Frozen input changed: '+str(path)
    if str(path) in hashes:assert hashes[str(path)]==digest
    hashes[str(path)]=digest
    return digest

def read(path):record(path);return json.loads(Path(path).read_text())
def load(path):record(path);return r.load_npz(path)
def dump(path,value):f.atomic_json(path,value)
def emit(stage,**kw):
    row={'stage':stage,'updated_unix':time.time(),**kw};dump(OUT/'progress.json',row);print(json.dumps(row),flush=True)
def aggregate(frames):return np.unique(np.concatenate(frames),return_counts=True)
def reencode(pair,base):
    keys=pair['surface_point_index'].astype(np.int64)*base+pair['instance_id']
    order=np.argsort(keys,kind='stable');return keys[order],pair['frame_votes'][order].astype(np.int64)
def pair_table(keys,votes,base):
    return {'surface_point_index':(keys//base).astype(np.int32),'instance_id':(keys%base).astype(np.int32),'frame_votes':votes.astype(np.int32)}
def reference_strict(keys,base):
    assert np.all(keys[1:]>keys[:-1])
    point=keys//base;unique=np.ones(len(keys),bool)
    unique[1:] &= point[1:]!=point[:-1];unique[:-1] &= point[1:]!=point[:-1]
    return keys[unique]
def compare_evidence(first,second):
    for name in FIELDS:np.testing.assert_array_equal(first[name],second[name],err_msg=name)
def state_matrix(before,after):return np.bincount(before.astype(np.int64)*4+after,minlength=16).reshape(4,4).astype(int).tolist()
def summary(evidence,final,before=None):
    s=evidence['state'];assert np.all(s[final<=0]!=2)
    data={'native_state_counts':np.bincount(s,minlength=4).astype(int).tolist(),
          'final_three_state_counts':{str(i):int(np.sum((s==i)&(final<=0))) for i in (0,1,3)},
          'final_unassigned':int(np.sum(final<=0))}
    if before is not None:
        data.update(assigned_to_unassigned=int(np.sum((before>0)&(final<=0))),
                    unassigned_to_assigned=int(np.sum((before<=0)&(final>0))),
                    changed_labels=int(np.sum(before!=final)))
    return data

def catalog(root,scene):
    initial=read(root/'input_freeze.json');assert initial['GT_used'] is False
    for group in ('original_baseline_source_hashes','seed_and_config_hashes','raw_rgb_hashes'):
        for p,d in initial[group].items():record(p,d)
    cases=read(root/'seed_diagnostics.json')['cases']
    association=read(root/'global_association.json');assert association['GT_used'] is False
    read(root/'alias_review.json');read(root/'repair_complete.json')
    obj={};info={};canonical={};fixed={}
    for row in association['objects']:
        fixed[row['canonical_track_id']]=row['persistent_id']
        for oid in row['member_track_ids']:canonical[oid]=row['canonical_track_id']
    for case in cases:
        uid=case['case_uid'];cfg=read(case['config_path'])
        track=read(root/'cases'/uid/'tracking/complete.json')
        assert track['status']=='PASS' and track['completed_frames']==track['total_frames']==2000
        assert track['GT_used'] is False and track['directions_use_independent_seed_only_states']
        assert track['case_config_sha256']==r.sha(case['config_path'])
        record(initial['tracking_code_path'],track['code_sha256'])
        record(cfg['tracking']['checkpoint'],track['checkpoint_sha256'])
        seedpath=root/'cases'/uid/'prepared_seed.png';record(seedpath,track['seed_sha256'])
        info[uid]={'config':cfg,'frames':{row['frame']:row for row in track['frames']},'seedpath':seedpath}
        for item in case['objects']:obj[item['track_id']]=uid
    assert set(obj)==set(canonical)
    metadata={cond:read(root/'validated'/cond/'frame_decisions.json') for cond in CONDITIONS}
    decisions={cond:{row['frame']:row for row in d['frames']} for cond,d in metadata.items()}
    prior_reports={}
    for cond in CONDITIONS:
        folder=root/'validated'/cond
        freeze=read(folder/'source_freeze.json')
        assert freeze['GT_used_for_repair'] is False
        for p,d in freeze['source_hashes'].items():record(p,d)
        done=read(folder/'complete.json');assert done['status']=='PASS'
        for p,d in done['output_sha256'].items():record(p,d)
        prior_reports[cond]=done
        assert sum(bool(row.get('accepted_track_ids')) for row in metadata[cond]['frames'])==done['counts']['updated_frames']
    return association,obj,info,canonical,fixed,decisions,prior_reports

def raw_replay(scene,fid,frame,table,op,ol,root,cond,decision,obj,info,canonical,fixed,merge):
    txpath=root/'validated'/cond/'transactions'/('f%06d.npz'%fid)
    record(txpath,decision['transaction_sha256']);tx=load(txpath);base=int(tx['instance_base'][0])
    accepted=list(map(int,decision['accepted_track_ids']))
    np.testing.assert_array_equal(tx['accepted_track_ids'],accepted)
    # Reconstruct the first overlap removal, including objects rejected only AFTER trimming.
    initial=set(accepted)|{int(oid) for oid,item in decision['objects'].items() if item.get('rechecked_after_overlap_abstention')}
    labels={}
    for uid in sorted({obj[oid] for oid in initial}):
        if cond=='seed_only':
            assert fid==info[uid]['config']['seed_frame'];path=info[uid]['seedpath']
        else:
            row=info[uid]['frames'][fid];path=Path(row['label_file']);record(path,row['label_sha256'])
        labels[uid]=np.array(Image.open(path),dtype=np.uint16,copy=True)
        assert labels[uid].shape==frame.mask_local.shape
    first,conflict=merge([(oid,labels[obj[oid]]==oid) for oid in sorted(initial)],canonical)
    assert int(conflict.sum())==decision['cross_case_conflict_pixels']
    for value in labels.values():value[conflict]=0
    merged,remaining=merge([(oid,labels[obj[oid]]==oid) for oid in accepted],canonical)
    assert not remaining.any()
    plans={int(mid):action for mid,action in decision['old_observation_plans'].items()}
    assert all(action in ('whole','partial') for action in plans.values())
    partial=[mid for mid,action in plans.items() if action=='partial']
    residual=np.where(np.isin(frame.mask_local,partial),frame.mask_local,0).astype(frame.mask_local.dtype)
    for mid in partial:
        pid=int(table[mid]);eligible={canonical[oid] for oid in accepted if pid in info[obj[oid]]['config']['old_family_ids']}
        residual[(frame.mask_local==mid)&np.isin(merged,list(eligible))]=0
    pp,pl,_=r.project_frame_regions(replace(frame,mask_local=residual),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
    np_,nc,_=r.project_frame_regions(replace(frame,mask_local=merged),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
    retained_points=np.r_[op[~np.isin(ol,list(plans))],pp]
    retained_local=np.r_[ol[~np.isin(ol,list(plans))],pl]
    rb=len(table)
    raw_region=np.unique(op.astype(np.int64)*rb+ol)
    residual_region=np.unique(retained_points.astype(np.int64)*rb+retained_local)
    assert np.all(np.isin(residual_region,raw_region))
    retired=np.setdiff1d(raw_region,residual_region)
    stored=tx['retired_surface_local_pairs'];np.testing.assert_array_equal(retired,np.unique(stored[:,0]*rb+stored[:,1]))
    retained=np.unique(retained_points.astype(np.int64)*base+table[retained_local])
    np.testing.assert_array_equal(retained,tx['retained_old_frame_keys'])
    newkeys=np.unique(np_.astype(np.int64)*base+np.asarray([fixed[int(cid)] for cid in nc],np.int64))
    rebuilt=np.union1d(retained,newkeys)
    np.testing.assert_array_equal(rebuilt,tx['new_frame_keys'])
    old=np.unique(op.astype(np.int64)*base+table[ol]);np.testing.assert_array_equal(old,tx['old_frame_keys'])
    scope=np.unique(np.r_[retired//rb,np_]).astype(np.int32)
    np.testing.assert_array_equal(scope,tx['allowed_surface_point_index'])
    oldstrict,newstrict,removed,added=frame_delta(old,rebuilt,base)
    np.testing.assert_array_equal(oldstrict,reference_strict(old,base));np.testing.assert_array_equal(newstrict,reference_strict(rebuilt,base))
    assert len(newstrict)==len(np.unique(newstrict//base))
    assert np.all(np.isin(np.r_[removed,added]//base,scope))
    retired_identity=np.unique((retired//rb)*base+table[retired%rb])
    shared=np.intersect1d(retired_identity,retained)
    assert np.all(np.isin(shared,rebuilt))
    shared_effective=np.intersect1d(shared,newstrict)
    cancelled_old=np.setdiff1d(old,oldstrict)
    release=np.intersect1d(added,np.intersect1d(retained,cancelled_old))
    return old,rebuilt,newstrict,removed,added,scope,{
        'shared_retired_and_retained_identity_keys':len(shared),'shared_effective_keys_after':len(shared_effective),
        'shared_keys_abstained_after':len(shared)-len(shared_effective),'released_previously_abstained_retained_votes':len(release),
        'ambiguous_point_frames_before':int(len(old)-len(oldstrict)),'ambiguous_point_frames_after':int(len(rebuilt)-len(newstrict)),
        'raw_projection_matches_frozen_transaction':True},tx

def scene_run(scene_id,priorfreeze):
    started=time.monotonic();root=ROOTS[scene_id];folder=OUT/scene_id
    assert not (folder/'complete.json').exists()
    emit('validate_frozen_repair_inputs',scene=scene_id)
    assoc,obj,info,canonical,fixed,decisions,prior_reports=catalog(root,scene_id)
    base=assoc['instance_base'];scene=r.Scene(scene_id)
    for p,d in scene.source_hashes.items():record(p,d)
    record(r.INPUT/scene_id/'geometry/surface.ply',scene.report['tsdf_surface_sha256'])
    record(r.INPUT/scene_id/'configs/raw.json')
    # Both frozen batches use exactly this identical overlap-abstention helper.
    helper_path=Path('/home/chenkejun/CVPR/experiments/room0_repair_20261007/batch_386cdcff710d/joint_batch_ops.py')
    record(helper_path)
    other_helper=Path('/home/chenkejun/CVPR/experiments/room2_top30_repair_20261008/batch_6b3f32f5dc1c/joint_batch_ops.py')
    record(other_helper,r.sha(helper_path))
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_joint_helper',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    before_frames=[];strict_before=[];raw_after={c:[] for c in CONDITIONS};strict_after={c:[] for c in CONDITIONS}
    removed={c:[] for c in CONDITIONS};added={c:[] for c in CONDITIONS};scopes={c:[] for c in CONDITIONS};receipts={c:[] for c in CONDITIONS}
    for index,fid in enumerate(sorted(scene.records)):
        frame,table,old,_=scene.old_frame(fid,base)
        support_path=Path(scene.records[fid]['support_file']);record(support_path,scene.records[fid]['support_sha256'])
        support=r.load_npz(support_path);op,ol=support['surface_point_index'],support['mask_local_id']
        for path in (scene.source.scene_root/scene.source.config['source']['trajectory'],
                     scene.source.mask_root/scene.source.config['source']['mask_pattern'].format(frame=fid)):
            if str(path) not in hashes:record(path)
        oldstrict=strict_frame_keys(old,base);np.testing.assert_array_equal(oldstrict,reference_strict(old,base))
        before_frames.append(old);strict_before.append(oldstrict)
        for cond in CONDITIONS:
            decision=decisions[cond].get(fid)
            if not decision or not decision.get('accepted_track_ids'):
                assert not (root/'validated'/cond/'transactions'/('f%06d.npz'%fid)).exists()
                raw_after[cond].append(old);strict_after[cond].append(oldstrict);continue
            before,after,eff,rem,add,scope,receipt,tx=raw_replay(scene,fid,frame,table,op,ol,root,cond,decision,obj,info,canonical,fixed,helper.merge_foregrounds)
            np.testing.assert_array_equal(before,old)
            raw_after[cond].append(after);strict_after[cond].append(eff);removed[cond].append(rem);added[cond].append(add);scopes[cond].append(scope)
            receipt.update(frame=fid,strict_removed_votes=len(rem),strict_added_votes=len(add),allowed_points=len(scope))
            receipts[cond].append(receipt)
            f.atomic_npz(folder/cond/'transactions'/('f%06d.npz'%fid),old_candidate_keys=before,new_candidate_keys=after,
                old_frame_keys=oldstrict,new_frame_keys=eff,retained_old_candidate_keys=tx['retained_old_frame_keys'],
                retired_surface_local_pairs=tx['retired_surface_local_pairs'],allowed_surface_point_index=scope,
                instance_base=np.asarray([base],np.int64))
        if (index+1)%40==0:emit('reproject_frozen_observations',scene=scene_id,frames_done=index+1,total_frames=400)
    original_keys,original_votes=reencode(scene.pair,base)
    k,v=aggregate(before_frames);np.testing.assert_array_equal(k,original_keys);np.testing.assert_array_equal(v,original_votes)
    bk,bv=aggregate(strict_before)
    strict_evidence=load(PRIOR/'ablation'/scene_id/'abstain/surface_evidence.npz')
    strict_pair=load(PRIOR/'ablation'/scene_id/'abstain/surface_instance_frame_votes.npz')
    ek,ev=reencode(strict_pair,base);np.testing.assert_array_equal(bk,ek);np.testing.assert_array_equal(bv,ev)
    baseline_recomputed=r.reduce_surface_votes(bk,bv,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
    compare_evidence(baseline_recomputed,strict_evidence)
    strict_final=load(priorfreeze['scenes'][scene_id]['predictions']['abstain_final']['path'])['instance_id']
    result={'status':'PASS','scene':scene_id,'base':base,'raw_frames_verified':400,'GT_used_for_mapping':False,
            'strict_baseline_matches_prior_exactly':True,'conditions':{},'predictions':{}}
    for semantic,freeze_name in (('original','original'),('strict','abstain')):
        for stage in ('native','final'):
            rec=priorfreeze['scenes'][scene_id]['predictions'][freeze_name+'_'+stage]
            record(rec['path'],rec['sha256']);result['predictions'][semantic+'_baseline_'+stage]=rec
    for cond in CONDITIONS:
        emit('verify_full_replay_and_rollback',scene=scene_id,condition=cond)
        source=root/'validated'/cond
        rawk,rawv=aggregate(raw_after[cond]);stored_pair=load(source/'native/surface_instance_frame_votes.npz')
        expectedk,expectedv=reencode(stored_pair,base);np.testing.assert_array_equal(rawk,expectedk);np.testing.assert_array_equal(rawv,expectedv)
        rawstate=r.reduce_surface_votes(rawk,rawv,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
        old_evidence=load(source/'native/surface_evidence.npz');compare_evidence(rawstate,old_evidence)
        rem=np.concatenate(removed[cond]) if removed[cond] else np.empty(0,np.int64)
        add=np.concatenate(added[cond]) if added[cond] else np.empty(0,np.int64)
        nk,nv=batch_delta(bk,bv,rem,add)
        fullk,fullv=aggregate(strict_after[cond]);np.testing.assert_array_equal(nk,fullk);np.testing.assert_array_equal(nv,fullv)
        backk,backv=batch_delta(nk,nv,add,rem);np.testing.assert_array_equal(backk,bk);np.testing.assert_array_equal(backv,bv)
        scope=np.unique(np.concatenate(scopes[cond])).astype(np.int32)
        np.testing.assert_array_equal(scope,load(source/'allowed_surface_ids.npz')['surface_point_index'])
        evidence=r.reduce_surface_votes(nk,nv,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
        native=np.where(evidence['state']==r.CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32)
        pair=pair_table(nk,nv,base)
        fullraw={'xyz_m':scene.xyz,'rgb':scene.rgb,**evidence}
        proof=f.verify_raw_outside(strict_evidence,fullraw,strict_pair,pair,scope)
        emit('frozen_postprocessing',scene=scene_id,condition=cond)
        variants,_,stats=r.assign_surface(scene.xyz,scene.normals,scene.rgb,evidence,pair,native,scene.settings)
        candidate=variants['holes_geodesic']
        final,commit,blocked=f.bounded_final_labels({'xyz_m':scene.xyz,'rgb':scene.rgb,'instance_id':strict_final},
            {'xyz_m':scene.xyz,'rgb':scene.rgb,'instance_id':candidate},scope)
        target=folder/cond
        f.atomic_npz(target/'native/surface_evidence.npz',**fullraw)
        f.atomic_npz(target/'native/surface_instance_frame_votes.npz',**pair)
        f.atomic_npz(target/'allowed_surface_ids.npz',surface_point_index=scope)
        f.atomic_npz(target/'candidate/instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=candidate)
        f.atomic_npz(target/'blocked_outside_changes.npz',surface_point_index=blocked)
        old_final=load(source/'final/instance_surface.npz')['instance_id']
        data={'raw_repaired_transaction_replay_exact':True,'strict_full_replay_equals_delta':True,'exact_rollback':True,
            'old_vote_rule_controls':summary(old_evidence,old_final,scene.final),
            'strict_vote_rule':summary(evidence,final,strict_final),'strict_commit':{**commit,**proof},
            'native_transition_vs_strict_baseline':state_matrix(strict_evidence['state'],evidence['state']),
            'frame_receipts':receipts[cond], 'projection_checked_transactions':len(receipts[cond]),
            'raw_shared_vote_keys_preserved':sum(x['shared_retired_and_retained_identity_keys'] for x in receipts[cond]),
            'released_previously_abstained_retained_votes':sum(x['released_previously_abstained_retained_votes'] for x in receipts[cond]),
            'same_fixed_repair_scope':True,'same_frozen_thresholds_and_postprocessing':True,
            'final_state_transition_vs_strict_baseline':state_matrix(np.where(strict_final>0,2,strict_evidence['state']),np.where(final>0,2,evidence['state'])),
            'final_changed_points_vs_original_semantics':int(np.sum(final!=old_final)),
            'strict_removed_votes':len(rem),'strict_added_votes':len(add),'postprocessing_statistics':stats}
        for stage,labels in (('native',native),('final',final)):
            oldpath=source/stage/'instance_surface.npz';oldmap=load(oldpath)
            old_inventory=oldmap.get('native_instance_ids',np.unique(oldmap['instance_id'][oldmap['instance_id']>0])).astype(np.int32)
            inventory=np.union1d(old_inventory,np.unique(labels[labels>0])).astype(np.int32)
            path=target/stage/'instance_surface.npz'
            f.atomic_npz(path,xyz_m=scene.xyz,rgb=scene.rgb,instance_id=labels,native_instance_ids=inventory)
            result['predictions']['original_'+cond+'_'+stage]={'path':str(oldpath),'sha256':r.sha(oldpath),'inventory':old_inventory.astype(int).tolist()}
            result['predictions']['strict_'+cond+'_'+stage]={'path':str(path),'sha256':r.sha(path),'inventory':inventory.astype(int).tolist()}
            data[stage+'_inventory_change']={'retained_original_inventory':len(old_inventory),'strict_inventory':len(inventory),
                'newly_activated_existing_ids':np.setdiff1d(inventory,old_inventory).astype(int).tolist()}
        dump(target/'complete.json',data);result['conditions'][cond]={k:v for k,v in data.items() if k not in ('frame_receipts','postprocessing_statistics')}
        emit('condition_complete',scene=scene_id,condition=cond,summary=result['conditions'][cond])
        del rawk,rawv,nk,nv,fullk,fullv,backk,backv,evidence,pair,fullraw,variants;gc.collect()
    for rec in result['predictions'].values():record(rec['path'],rec['sha256'])
    result['generated_output_sha256']={str(p):r.sha(p) for p in folder.rglob('*.npz')}
    scene.verify_unchanged();result['seconds']=time.monotonic()-started
    dump(folder/'complete.json',result);return result

def main():
    assert not (OUT/'predictions_freeze.json').exists(),'Predictions already frozen'
    priorfreeze=read(PRIOR/'predictions_freeze.json')
    assert priorfreeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    for path,digest in priorfreeze['code_sha256'].items():record(path,digest)
    for path in (HERE/'one_vote.py',HERE/'strict_vote_ops.py',Path(__file__),HERE/'test_strict_repair.py'):record(path)
    for path in (LEGACY/'run_repairs.py',LEGACY/'joint_vote_ops.py',LEGACY/'evidence_replacement.py',LEGACY/'strict_local_repair_20261007/fixed_surface_repair.py'):record(path)
    tests=read(OUT/'tests_complete.json');assert tests['status']=='PASS' and tests['tests']==10
    policy={'status':'FROZEN_BEFORE_NEW_PREDICTIONS_AND_GT','vote_rule':'one frame one point at most one ID; unresolved multi-ID abstention',
        'cases':{'room0':12,'room2':14},'conditions':list(CONDITIONS),'repair_source_roots':{s:str(p) for s,p in ROOTS.items()},
        'changed':'effective vote reduction after rebuilding the complete raw candidate frame',
        'fixed':['human masks','SAM tracking','identity association','accepted objects and observation plans','numeric thresholds','raw support projection','geometry/RGB','allowed repair scope','bounded final commit','postprocessing settings','v3 evaluation scope/config'],
        'inventory_policy':'Strict condition retains its original-semantic counterpart exported stage IDs, appends newly activated existing IDs; no disappeared ID is silently dropped',
        'decisions':['Evaluate latest room0 and room2 batches with both seed-only and full-track conditions.',
          'Replay frozen raw edits and verify every saved transaction using fresh residual/new mask projections.',
          'Reduce retained old and new candidate support together, so same-ID sharing remains once and different-ID support abstains.',
          'Use strict no-repair baseline for strict repair effect; original semantic controls use original no-repair baseline.',
          'Require 400-frame full sum, incremental equality, exact rollback, and unchanged raw/final outside frozen scope.',
          'Reuse frozen SAM inference and gates; do not reselect cases, identities or thresholds using GT.',
          'Freeze all native/final predictions before any GT read; score all rows with current same v3 development profile.'],
        'GT_used_for_mapping':False,'baseline_modified':False}
    dump(OUT/'policy.json',policy)
    results={}
    for scene in ROOTS:results[scene]=scene_run(scene,priorfreeze);gc.collect()
    emit('verify_all_input_hashes_after_mapping',input_files=len(hashes))
    for path,digest in hashes.items():assert r.sha(path)==digest,'Input modified: '+path
    freeze={'status':'PREDICTIONS_FROZEN_BEFORE_GT','GT_used_for_mapping':False,'baseline_modified':False,
        'scenes':{s:{'completion_sha256':r.sha(OUT/s/'complete.json'),'predictions':v['predictions']} for s,v in results.items()},
        'generated_output_sha256':{p:d for value in results.values() for p,d in value['generated_output_sha256'].items()},
        'source_sha256':hashes,'policy_sha256':r.sha(OUT/'policy.json'),'new_code_sha256':{str(p):r.sha(p) for p in HERE.glob('*.py')}}
    dump(OUT/'predictions_freeze.json',freeze)
    dump(OUT/'mapping_complete.json',{'status':'PASS','original_raw_frames_checked':800,
        'repair_conditions_replayed_frames':1600,'fresh_projection_transactions':sum(v['conditions'][c]['projection_checked_transactions'] for v in results.values() for c in CONDITIONS),
        'full_replay_delta_rollback_and_scope_pass':True,'predictions_frozen_before_GT':True,'input_files_verified':len(hashes)})
    emit('mapping_complete',status='PASS')

if __name__=='__main__':
    try:main()
    except Exception:dump(OUT/'failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
