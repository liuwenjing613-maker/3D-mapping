"""Fixed manual-core history reassociation on room0 and room2.

Adds only source-pixel reassociation to frozen strict full-track repair. Does
not read GT, change masks/tracking/gates/associations, or overwrite controls.
"""
from pathlib import Path
from dataclasses import replace
import gc,hashlib,json,sys,time,traceback
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from history_core_ops import core_claims,retarget_selection,strict_frame

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/p1a1_history_core_reassociation_20261009')
STRICT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
BATCH=Path('/home/chenkejun/CVPR/experiments/room0_repair_20261007/batch_386cdcff710d')
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'));sys.path.insert(0,str(BATCH))
import run_repairs as r
import fixed_surface_repair as f
from joint_vote_ops import batch_delta
from joint_batch_ops import merge_foregrounds
ROOTS={'room0':Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d'),
       'room2':Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')}
protected={}
def protect(path,expected=None):
    path=Path(path);d=r.sha(path)
    if expected is not None:assert d==expected,str(path)
    if str(path) in protected:assert protected[str(path)]==d
    protected[str(path)]=d;return d
def read(path):protect(path);return json.loads(Path(path).read_text())
def load(path,expected=None):protect(path,expected);return r.load_npz(path)
def dump(path,obj):f.atomic_json(path,obj)
def emit(stage,**data):
    v={'stage':stage,**data,'unix':time.time()};dump(OUT/'progress.json',v);print(json.dumps(v),flush=True)
def keys(p,ids,base):return np.unique(p.astype(np.int64)*base+ids)
def encode(pair,base):
    k=pair['surface_point_index'].astype(np.int64)*base+pair['instance_id'];order=np.argsort(k)
    return k[order],pair['frame_votes'][order].astype(np.int64)
def pairtable(k,v,base):return {'surface_point_index':(k//base).astype(np.int32),'instance_id':(k%base).astype(np.int32),'frame_votes':v.astype(np.int32)}
def project(frame,mask,scene):
    return r.project_frame_regions(replace(frame,mask_local=mask),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)[:2]
def sourcepixels(frame,scene):
    mid=frame.mask_local[::2,::2];dep=frame.depth_m[::2,::2]
    valid=(mid>0)&np.isfinite(dep)&(dep>0)&(dep<10);v,u=np.nonzero(valid);v=v*2;u=u*2;z=dep[valid].astype(np.float64)
    cam=np.column_stack(((u-frame.camera.cx)*z/frame.camera.fx,(v-frame.camera.cy)*z/frame.camera.fy,z))
    world=cam@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
    dist,p=scene.tree.query(world,k=1,workers=8);ok=np.isfinite(dist)&(dist<.015)
    return p[ok].astype(np.int64),mid[valid][ok].astype(np.int64),u[ok],v[ok]
def summary(e,labels):
    return {'native_state_U_T_CONFIRMED_C':np.bincount(e['state'],minlength=4).tolist(),
        'final_gray':int(np.sum(labels<=0)),
        'final_gray_three_states':{name:int(np.sum((labels<=0)&(e['state']==value))) for name,value in [('UNOBSERVED',0),('TENTATIVE',1),('CONFLICT',3)]}}
def region_summary(e,labels,p,pid):
    return {'points':len(p),'own_ID_points':int(np.sum(labels[p]==pid)),'own_ID_fraction':float(np.mean(labels[p]==pid)) if len(p) else None,
        'final_gray':int(np.sum(labels[p]<=0)),
        'native_state_U_T_CONFIRMED_C':np.bincount(e['state'][p],minlength=4).tolist()}
def catalog(root):
    cases=read(root/'seed_diagnostics.json')['cases'];association=read(root/'global_association.json')
    assert association['GT_used'] is False
    canonical={};fixed={};obj={};info={}
    for row in association['objects']:
        fixed[row['canonical_track_id']]=row['persistent_id']
        for oid in row['member_track_ids']:canonical[oid]=row['canonical_track_id']
    assert len(set(fixed.values()))==len(fixed),'Persistent IDs must identify canonical groups uniquely'
    initial=read(root/'input_freeze.json');assert initial['GT_used'] is False
    for case in cases:
        uid=case['case_uid'];cfg=read(case['config_path']);complete=read(root/'cases'/uid/'tracking/complete.json')
        assert complete['GT_used'] is False and complete['status']=='PASS'
        assert complete['completed_frames']==complete['total_frames']==2000
        protect(case['config_path'],complete['case_config_sha256'])
        seedpath=root/'cases'/uid/'prepared_seed.png';protect(seedpath,complete['seed_sha256'])
        info[uid]={'config':cfg,'frames':{row['frame']:row for row in complete['frames']},'seedpath':seedpath}
        for item in case['objects']:obj[item['track_id']]=uid
    assert set(obj)==set(canonical)
    return cases,association,obj,info,canonical,fixed
def reconstruct_current(frame,table,scene,decision,obj,info,canonical,fixed):
    accepted=list(map(int,decision['accepted_track_ids']))
    # The frozen batch returns the untouched original frame when every proposal
    # is rejected (also after overlap rechecking). These rows deliberately omit
    # both the conflict summary and the observation plans; no proposal survives.
    if not accepted:
        assert not decision.get('old_observation_plans'),decision['frame']
        residual=np.array(frame.mask_local,copy=True)
        rp,rl=project(frame,residual,scene)
        return residual,rp,rl,np.empty(0,np.int64),np.empty(0,np.int64)
    initial=set(accepted)|{int(oid) for oid,m in decision['objects'].items() if m.get('rechecked_after_overlap_abstention')}
    labels={}
    for uid in sorted({obj[oid] for oid in initial}):
        row=info[uid]['frames'][frame.frame_id];protect(row['label_file'],row['label_sha256'])
        labels[uid]=np.asarray(Image.open(row['label_file'])).copy()
    if initial:_,conflict=merge_foregrounds([(oid,labels[obj[oid]]==oid) for oid in sorted(initial)],canonical)
    else:conflict=np.zeros(frame.mask_local.shape,bool)
    assert int(conflict.sum())==decision['cross_case_conflict_pixels']
    for labels_one in labels.values():labels_one[conflict]=0
    if accepted:merged,remaining=merge_foregrounds([(oid,labels[obj[oid]]==oid) for oid in accepted],canonical)
    else:merged=np.zeros(frame.mask_local.shape,np.uint16);remaining=np.zeros(frame.mask_local.shape,bool)
    assert not remaining.any()
    residual=np.array(frame.mask_local,copy=True)
    for s,action in decision['old_observation_plans'].items():
        mid=int(s)
        if action=='whole':residual[frame.mask_local==mid]=0
        else:
            assert action=='partial'
            eligible={canonical[oid] for oid in accepted if int(table[mid]) in info[obj[oid]]['config']['old_family_ids']}
            residual[(frame.mask_local==mid)&np.isin(merged,list(eligible))]=0
    rp,rl=project(frame,residual,scene);np_,nl=project(frame,merged,scene)
    return residual,rp,rl,np_,nl

def scene_run(scene_id,freeze):
    started=time.time();root=ROOTS[scene_id];folder=OUT/scene_id;assert not folder.exists();folder.mkdir()
    emit('manual_core_preflight',scene=scene_id)
    cases,association,obj,info,canonical,fixed=catalog(root);base=association['instance_base']
    scene=r.Scene(scene_id);records=freeze['scenes'][scene_id]['predictions']
    control=load(records['strict_full_track_final']['path'],records['strict_full_track_final']['sha256'])
    evidence=load(STRICT/scene_id/'full_track/native/surface_evidence.npz')
    pair=load(STRICT/scene_id/'full_track/native/surface_instance_frame_votes.npz');bk,bv=encode(pair,base)
    np.testing.assert_array_equal(scene.xyz,control['xyz_m']);np.testing.assert_array_equal(scene.rgb,control['rgb'])
    decisions={row['frame']:row for row in read(root/'validated/full_track/frame_decisions.json')['frames']}
    core_arrays=[];target_ids=[];objects=[];allseed=[];regression_core=None
    for case in cases:
        uid=case['case_uid'];cfg=info[uid]['config'];seed=scene.source.load_frame(cfg['seed_frame']);label=np.asarray(Image.open(info[uid]['seedpath']))
        support=load(root/'cases'/uid/'seed_projected_support.npz')
        sp,sl=project(seed,label,scene);np.testing.assert_array_equal(sp,support['surface_point_index']);np.testing.assert_array_equal(sl,support['track_id'])
        corelabel=np.zeros(label.shape,np.uint16)
        for item in case['objects']:
            tid=item['track_id'];eroded=binary_erosion(label==tid,iterations=10);corelabel[eroded]=tid
        cp,cl=project(seed,corelabel,scene)
        for item in case['objects']:
            tid=item['track_id'];pid=fixed[canonical[tid]];p=np.unique(cp[cl==tid]);seedp=np.unique(sp[sl==tid])
            core_arrays.append(p);target_ids.append(pid);allseed.append(seedp)
            objects.append({'case_uid':uid,'ROI':case['source_choice']['ROI'],'track_id':tid,'canonical_track_id':canonical[tid],
                'native_mask_id':item['native_mask_id'],'persistent_id':pid,'seed_frame':cfg['seed_frame'],
                'original_seed_pixels':int(np.sum(label==tid)),'eroded_seed_pixels':int(np.sum(corelabel==tid)),
                'seed_points':len(seedp),'core_points':len(p),'empty_core':not len(p),
                'before_seed':region_summary(evidence,control['instance_id'],seedp,pid),
                'before_core':region_summary(evidence,control['instance_id'],p,pid)})
            if scene_id=='room0' and uid=='room0-971efdbf3ec0d694' and tid==1:
                regression_core=np.zeros(len(scene.xyz),bool);regression_core[p]=True;assert len(p)==20872 and pid==796
    owner,claimkeys=core_claims(core_arrays,target_ids,len(scene.xyz),base)
    unique_core=np.flatnonzero(owner>0);ambiguous=np.flatnonzero(owner==-2);core_union=np.flatnonzero(owner!=-1)
    for item,p in zip(objects,core_arrays):item.update(unique_owned_core_points=int(np.sum(owner[p]==item['persistent_id'])),ambiguous_core_points=int(np.sum(owner[p]==-2)))
    f.atomic_npz(folder/'core_claims.npz',surface_point_index=core_union,claimed_persistent_id=owner[core_union],raw_point_ID_claims=claimkeys,
        unique_owned_surface_point_index=unique_core,ambiguous_surface_point_index=ambiguous)
    dump(folder/'preflight.json',{'GT_used':False,'cases':len(cases),'seed_objects':len(objects),'canonical_objects':len(fixed),
        'all_seed_unique_points':len(np.unique(np.concatenate(allseed))),'core_union_points':len(core_union),
        'uniquely_owned_core_points':len(unique_core),'ambiguous_core_points':len(ambiguous),
        'empty_core_objects':sum(o['empty_core'] for o in objects),'objects':objects})
    emit('core_preflight_complete',scene=scene_id,unique_core_points=len(unique_core),ambiguous_core_points=len(ambiguous),empty_core_objects=sum(o['empty_core'] for o in objects))
    id_to_canonical=np.zeros(base,np.uint16)
    for cid,pid in fixed.items():id_to_canonical[pid]=cid
    remove_chunks=[];add_chunks=[];scopes=[];before_scope_frames=[];after_scope_frames=[];receipts=[];regression_count=0
    reassoc_sources={};reassoc_targets={};projection_checked=0
    for fi,fid in enumerate(sorted(scene.records)):
        record=scene.records[fid];z=load(record['support_file'],record['support_sha256'])
        op=z['surface_point_index'].astype(np.int64);ol=z['mask_local_id'].astype(np.int64)
        table=np.zeros(int(ol.max())+1,np.int64)
        for mid in np.unique(ol):table[mid]=scene.lookup[(fid,int(mid))]
        raw_original=keys(op,table[ol],base)
        txpath=STRICT/scene_id/'full_track/transactions'/f'f{fid:06d}.npz'
        tx=load(txpath,freeze['generated_output_sha256'][str(txpath)]) if txpath.exists() else None
        current_raw=tx['new_candidate_keys'] if tx is not None else raw_original
        before=strict_frame(current_raw,base)
        if tx is not None:np.testing.assert_array_equal(before,tx['new_frame_keys']);np.testing.assert_array_equal(raw_original,tx['old_candidate_keys'])
        regression_path=Path('/data/chenkejun/CVPR/results/p1a1_case_c0001_mask2_20261009/counterfactual/core_history_reassociation/transactions')/f'f{fid:06d}.npz'
        after=before;hascore=np.any(owner[op]>0) or (regression_core is not None and regression_path.exists())
        if hascore:
            frame=scene.source.load_frame(fid);maskpath=scene.source.mask_root/scene.source.config['source']['mask_pattern'].format(frame=fid)
            protect(maskpath,str(z['source_mask_sha256'][0]));fulltable=np.zeros(int(frame.mask_local.max())+1,np.int64)
            for mid in np.unique(frame.mask_local):
                if mid:fulltable[mid]=scene.lookup[(fid,int(mid))]
            table=fulltable;p,mid,u,v=sourcepixels(frame,scene);rb=len(table)
            np.testing.assert_array_equal(np.unique(p*rb+mid),np.unique(op*rb+ol))
            residual,rp,rl,np_,nl=reconstruct_current(frame,table,scene,decisions[fid],obj,info,canonical,fixed)
            retained=keys(rp,table[rl],base);newkeys=keys(np_,np.asarray([fixed[int(k)] for k in nl],np.int64),base)
            if tx is not None:
                oldtx=load(root/'validated/full_track/transactions'/f'f{fid:06d}.npz',decisions[fid]['transaction_sha256'])
                np.testing.assert_array_equal(retained,oldtx['retained_old_frame_keys'])
                np.testing.assert_array_equal(np.union1d(retained,newkeys),current_raw)
            else:np.testing.assert_array_equal(retained,raw_original);assert not len(newkeys)
            available=residual[v,u]==mid;oldids=table[mid]
            choose=retarget_selection(p,oldids,available,owner);projection_checked+=1
            # Regression: reproduce all 85 previous C0001/179 fragment transactions.
            if regression_core is not None:
                regpath=regression_path
                if regpath.exists():
                    reg=load(regpath);sel=available&(oldids==179)&regression_core[p]
                    regleft=keys(p[available&~sel],oldids[available&~sel],base)
                    regnew=keys(p[sel],np.full(int(np.sum(sel)),796,np.int64),base)
                    expect=np.union1d(regleft,np.union1d(newkeys,regnew))
                    np.testing.assert_array_equal(expect,reg['new_candidate_keys']);np.testing.assert_array_equal(strict_frame(expect,base),reg['new_frame_keys']);regression_count+=1
            if np.any(choose):
                changed_residual=np.array(residual,copy=True);fragment=np.zeros(frame.mask_local.shape,np.uint16)
                changed_residual[v[choose],u[choose]]=0;fragment[v[choose],u[choose]]=id_to_canonical[owner[p[choose]]]
                ap,al=project(frame,changed_residual,scene);hp,hl=project(frame,fragment,scene)
                left=keys(ap,table[al],base);history=keys(hp,np.asarray([fixed[int(k)] for k in hl],np.int64),base)
                # Pixel provenance oracle is independent of the fresh fragment projections.
                expected_left=keys(p[available&~choose],oldids[available&~choose],base)
                expected_history=keys(p[choose],owner[p[choose]],base)
                np.testing.assert_array_equal(left,expected_left);np.testing.assert_array_equal(history,expected_history)
                assert np.all(np.isin(left,retained))
                candidate=np.union1d(left,np.union1d(newkeys,history));after=strict_frame(candidate,base)
                rem=np.setdiff1d(before,after);add=np.setdiff1d(after,before);scope=np.unique(np.r_[rem//base,add//base]).astype(np.int32)
                assert np.all(owner[scope]>0)
                removed_raw=np.setdiff1d(retained,left);shared=np.intersect1d(removed_raw,np.union1d(newkeys,left))
                assert np.all(np.isin(shared,candidate));assert np.all(np.isin(newkeys,candidate))
                np.testing.assert_array_equal(candidate[owner[candidate//base]<=0],current_raw[owner[current_raw//base]<=0])
                remove_chunks.append(rem);add_chunks.append(add);scopes.append(scope)
                old_unique,old_n=np.unique(oldids[choose],return_counts=True)
                for k,n in zip(old_unique,old_n):reassoc_sources[str(int(k))]=reassoc_sources.get(str(int(k)),0)+int(n)
                new_unique,new_n=np.unique(owner[p[choose]],return_counts=True)
                for k,n in zip(new_unique,new_n):reassoc_targets[str(int(k))]=reassoc_targets.get(str(int(k)),0)+int(n)
                row={'frame':fid,'reassociated_source_pixels':int(choose.sum()),'history_candidate_point_ID_keys':len(history),
                    'retired_original_candidate_keys':len(removed_raw),'shared_candidate_keys_preserved':len(shared),
                    'before_ambiguous_points':len(np.unique(current_raw//base))-len(before),
                    'after_ambiguous_points':len(np.unique(candidate//base))-len(after),
                    'effective_removed':len(rem),'effective_added':len(add),'changed_scope_points':len(scope)}
                receipts.append(row)
                f.atomic_npz(folder/'transactions'/f'f{fid:06d}.npz',old_candidate_keys=current_raw,new_candidate_keys=candidate,
                    old_frame_keys=before,new_frame_keys=after,removed_effective_keys=rem,added_effective_keys=add,
                    allowed_surface_point_index=scope,reassociated_pixel_uv=np.column_stack((u[choose],v[choose])).astype(np.int32),
                    source_original_mask_local_id=mid[choose].astype(np.int32),source_old_persistent_id=oldids[choose].astype(np.int32),
                    target_persistent_id=owner[p[choose]].astype(np.int32),source_surface_point_index=p[choose].astype(np.int32))
        before_scope_frames.append(before[owner[before//base]>0]);after_scope_frames.append(after[owner[after//base]>0])
        if (fi+1)%50==0:emit('history_source_replay',scene=scene_id,frames=fi+1,changed_frames=len(receipts),fresh_projection_frames=projection_checked)
    if scene_id=='room0':assert regression_count==85,regression_count
    rem=np.concatenate(remove_chunks) if remove_chunks else np.empty(0,np.int64);add=np.concatenate(add_chunks) if add_chunks else np.empty(0,np.int64)
    scope=np.unique(np.concatenate(scopes)) if scopes else np.empty(0,np.int32);nk,nv=batch_delta(bk,bv,rem,add)
    backk,backv=batch_delta(nk,nv,add,rem);np.testing.assert_array_equal(backk,bk);np.testing.assert_array_equal(backv,bv)
    full_before_k,full_before_v=np.unique(np.concatenate(before_scope_frames),return_counts=True)
    full_after_k,full_after_v=np.unique(np.concatenate(after_scope_frames),return_counts=True)
    keep=owner[bk//base]>0;np.testing.assert_array_equal(bk[keep],full_before_k);np.testing.assert_array_equal(bv[keep],full_before_v)
    keep=owner[nk//base]>0;np.testing.assert_array_equal(nk[keep],full_after_k);np.testing.assert_array_equal(nv[keep],full_after_v)
    monitor=np.zeros(len(scene.xyz),bool);monitor[scope]=True
    bout=~monitor[bk//base];nout=~monitor[nk//base]
    np.testing.assert_array_equal(bk[bout],nk[nout]);np.testing.assert_array_equal(bv[bout],nv[nout])
    e=r.reduce_surface_votes(nk,nv,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
    for field in r.FIELDS:np.testing.assert_array_equal(e[field][~monitor],evidence[field][~monitor])
    native=np.where(e['state']==r.CONFIRMED,e['top1_instance_id'],-1).astype(np.int32);pt=pairtable(nk,nv,base)
    emit('history_votes_verified',scene=scene_id,changed_frames=len(receipts),scope_points=len(scope),single_case_regression_frames=regression_count)
    def progress(stage,values):emit('history_postprocessing',scene=scene_id,substage=stage,values=values)
    variants,_,poststats=r.assign_surface(scene.xyz,scene.normals,scene.rgb,e,pt,native,scene.settings,progress=progress)
    candidate=variants['holes_geodesic'];final=control['instance_id'].copy();final[scope]=candidate[scope]
    np.testing.assert_array_equal(final[~monitor],control['instance_id'][~monitor])
    oldnative=load(records['strict_full_track_native']['path'],records['strict_full_track_native']['sha256'])
    for stage,lab,old in [('native',native,oldnative),('final',final,control)]:
        inventory=np.union1d(old['native_instance_ids'],np.unique(lab[lab>0])).astype(np.int32)
        f.atomic_npz(folder/stage/'instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=lab,native_instance_ids=inventory)
    f.atomic_npz(folder/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**e)
    f.atomic_npz(folder/'native/surface_instance_frame_votes.npz',**pt)
    f.atomic_npz(folder/'allowed_surface_ids.npz',surface_point_index=scope)
    for item,p,seedp in zip(objects,core_arrays,allseed):
        item['after_seed']=region_summary(e,final,seedp,item['persistent_id']);item['after_core']=region_summary(e,final,p,item['persistent_id'])
    control_summary=summary(evidence,control['instance_id']);after_summary=summary(e,final)
    allS=np.unique(np.concatenate(allseed))
    complete={'status':'PASS','GT_read_by_mapper':False,'scene':scene_id,'cases':len(cases),'seed_objects':len(objects),'canonical_objects':len(fixed),
        'core_union_points':len(core_union),'unique_core_points':len(unique_core),'ambiguous_core_points':len(ambiguous),'empty_core_objects':sum(o['empty_core'] for o in objects),
        'mapping_frames':400,'fresh_original_and_current_repair_projection_frames':projection_checked,'reassociated_frames':len(receipts),
        'frozen_frames_without_accepted_new_masks':sum(not row['accepted_track_ids'] for row in decisions.values()),
        'reassociated_source_pixels':sum(row['reassociated_source_pixels'] for row in receipts),
        'raw_retired_candidate_keys':sum(row['retired_original_candidate_keys'] for row in receipts),
        'shared_candidates_preserved':sum(row['shared_candidate_keys_preserved'] for row in receipts),
        'source_old_ID_histogram':reassoc_sources,'target_ID_histogram':reassoc_targets,
        'allowed_scope_points':len(scope),'current_strict_full_track':control_summary,'history_reassociation':after_summary,
        'final_gray_delta':after_summary['final_gray']-control_summary['final_gray'],
        'final_changed_labels':int(np.sum(final!=control['instance_id'])),
        'assigned_to_gray':int(np.sum((control['instance_id']>0)&(final<=0))),
        'gray_to_assigned':int(np.sum((control['instance_id']<=0)&(final>0))),
        'native_state_transition_U_T_CONFIRMED_C':np.bincount(evidence['state'].astype(np.int64)*4+e['state'],minlength=16).reshape(4,4).tolist(),
        'single_C0001_mask2_old179_regression_transactions_exact':regression_count,
        'checks':{'fresh_original_projection_matches_frozen_support':True,'fresh_retained_and_new_mask_projections_match_frozen_repair':True,
            'source_pixel_oracle_equals_fresh_retained_and_history_projection':True,'strict_frame_point_uniqueness':True,
            'full_400_frame_replay_equals_delta':True,'exact_rollback':True,'shared_support_preserved':True,
            'existing_new_masks_preserved':True,'raw_and_final_outside_scope_bit_identical':True},
        'objects':objects,'frame_receipts':receipts,'postprocessing_statistics':poststats,'seconds':time.time()-started}
    dump(folder/'complete.json',complete)
    f.atomic_npz(folder/'selected_surface_comparison.npz',surface_point_index=allS,xyz_m=scene.xyz[allS],rgb=scene.rgb[allS],
        before_final_ID=control['instance_id'][allS],after_final_ID=final[allS],before_native_state=evidence['state'][allS],after_native_state=e['state'][allS],
        core_owner=owner[allS])
    scene.verify_unchanged()
    for p,d in scene.source_hashes.items():protect(p,d)
    emit('scene_mapping_complete',scene=scene_id,final_gray_delta=complete['final_gray_delta'],changed_labels=complete['final_changed_labels'],seconds=complete['seconds'])
    return {k:v for k,v in complete.items() if k not in ('postprocessing_statistics','frame_receipts')}

def main():
    assert not (OUT/'policy.json').exists() and not (OUT/'predictions_freeze.json').exists()
    tests=read(OUT/'tests_complete.json');assert tests['status']=='PASS' and tests['tests']==8
    freeze=read(STRICT/'predictions_freeze.json');assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    oldhashes={**freeze['source_sha256'],**freeze['new_code_sha256']}
    used=[LEGACY/'run_repairs.py',LEGACY/'joint_vote_ops.py',LEGACY/'strict_local_repair_20261007/fixed_surface_repair.py',BATCH/'joint_batch_ops.py']
    for p in used:protect(p,oldhashes[str(p)])
    for p in HERE.glob('*.py'):protect(p)
    dump(OUT/'policy.json',{'status':'FROZEN_BEFORE_TWO_SCENE_MAPPING_AND_NEW_R2_SCORING','method':'history_core_reassociation_v1',
        'scenes':['room0','room2'],'repair_cases':{'room0':12,'room2':14},'all_selected_seed_masks':{'room0':30,'room2':25},
        'control':'Frozen strict one-vote+abstention full-track repair',
        'manual_core_erosion_pixels':10,'empty_core_policy':'No fallback or parameter relaxation',
        'projection_stride':2,'projection_max_distance_m':.015,'maximum_depth_m':10,
        'old_identity_selection':'All retained original positive IDs different from the uniquely claimed target; no old_family_ids restriction',
        'multi_target_policy':'Same persistent ID unions; different persistent IDs mark ambiguous core and retain current observations',
        'already_accepted_new_masks':'Preserved exactly; may still cause strict per-frame abstention',
        'replacement_unit':'Frozen sampled source depth pixels; split residual old mask and fresh-project new fragment',
        'vote_rule':'Deduplicate same-ID supports; multi-ID candidates abstain jointly per frame/point',
        'final_commit':'Only effective changed source scope; candidate geodesic postprocessing settings frozen',
        'GT_used_for_masks_frames_IDs_or_thresholds':False,'GT_read_by_mapper':False,'diagnosis_informed_method':True,
        'pre_registered_benchmark':False,'baseline_modified':False,'SAM_tracking_reused':True})
    result={}
    for scene in ('room0','room2'):result[scene]=scene_run(scene,freeze);gc.collect()
    for p,d in protected.items():assert r.sha(p)==d,p
    outputs={str(p):r.sha(p) for p in OUT.rglob('*.npz')}
    dump(OUT/'predictions_freeze.json',{'status':'TWO_SCENE_PREDICTIONS_FROZEN_BEFORE_NEW_R2_SCORING','GT_read_by_mapper':False,
        'GT_used_for_masks_frames_IDs_or_thresholds':False,'diagnosis_informed_method':True,'pre_registered_benchmark':False,
        'source_sha256':protected,'output_sha256':outputs,'policy_sha256':r.sha(OUT/'policy.json'),
        'code_sha256':{str(p):r.sha(p) for p in HERE.glob('*.py')},'scenes':result,'baseline_modified':False})
    emit('mapping_complete',status='PASS',scenes=list(result))

if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
