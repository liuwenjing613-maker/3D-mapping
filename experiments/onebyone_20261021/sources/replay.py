"""Approved frame/TSDF-point replacement; unchanged frozen E1 history rules."""
from bootstrap import C,R,SOURCE,SOURCE_CODE,PARTIAL,COUNT,SCENE,GROUP,CFG,verify_sources,load_script
from pathlib import Path
import gc,json,time,traceback
import numpy as np
from PIL import Image
import run_history_core as h
from point_replace import replace_frame_points
obs=load_script('_point_original_observation_ops',SOURCE_CODE/'observation_ops.py')
STAGES=('direct','core_all','e1'); PARENT=dict(zip(STAGES,('baseline','direct','core_all')))
sha=h.r.sha
cat=lambda a:np.concatenate(a) if a else np.empty(0,np.int64)
aggregate=lambda a:np.unique(cat(a),return_counts=True)
dump=h.f.atomic_json

def emit(stage,**kw):
    v=dict(stage=stage,group=GROUP,scene=SCENE,unix=time.time(),**kw)
    dump(R/'progress.json',v); print(json.dumps(v),flush=True)

def adopted_mask(frame,decision,obj,info,canonical):
    accepted=list(map(int,decision['accepted_track_ids']))
    if not accepted: merged=np.zeros(frame.mask_local.shape,np.uint16)
    else:
        initial=set(accepted)|{int(o) for o,v in decision['objects'].items() if v.get('rechecked_after_overlap_abstention')}
        labels={}
        for uid in sorted({obj[o] for o in initial}):
            row=info[uid]['frames'][frame.frame_id]
            h.protect(row['label_file'],row['label_sha256']); labels[uid]=np.asarray(Image.open(row['label_file'])).copy()
        _,conflict=h.merge_foregrounds([(o,labels[obj[o]]==o) for o in sorted(initial)],canonical)
        assert int(conflict.sum())==decision['cross_case_conflict_pixels']
        for label in labels.values():label[conflict]=0
        merged,remaining=h.merge_foregrounds([(o,labels[obj[o]]==o) for o in accepted],canonical)
        assert not remaining.any()
    path=SOURCE/'validated/full_track/accepted_masks'/f'f{frame.frame_id:06d}.png'
    h.protect(path); np.testing.assert_array_equal(merged,np.asarray(Image.open(path)))
    return merged

def transition(a,b,base):
    ap,bp=a//base,b//base
    common,ai,bi=np.intersect1d(ap,bp,return_indices=True)
    return dict(valid_to_zero=len(np.setdiff1d(ap,bp)),zero_to_valid=len(np.setdiff1d(bp,ap)),
                valid_id_changed=int((a[ai]!=b[bi]).sum()))

def main():
    started=time.time(); assert not (R/'predictions_freeze.json').exists()
    R.mkdir(parents=True,exist_ok=True)
    codehash={str(p):sha(p) for p in C.glob('*.py')}
    selection=h.read(SOURCE/'confirmed_selection.json');assert len(selection['included'])==COUNT
    freeze=h.read(SOURCE/'e1/region_freeze.json')
    for kind in ('code_sha256','source_sha256'):
        for p,d in freeze[kind].items():h.protect(p,d)
    h.protect(SOURCE/'e1/region.npz',freeze['region_sha256'])
    h.protect(SOURCE/'e1/policy.json',freeze['policy_sha256'])
    pipeline=h.read(SOURCE/'pipeline_code_freeze.json')
    for p,d in pipeline['files'].items():h.protect(p,d)
    origfreeze=h.read(SOURCE/'original_repair/predictions_freeze.json')
    for p,d in origfreeze.get('new_code_sha256',{}).items():h.protect(p,d)
    cases,assoc,obj,info,canonical,fixed=h.catalog(SOURCE)
    assert len(obj)==COUNT; base=assoc['instance_base']; scene=h.r.Scene(SCENE)
    decisions={d['frame']:d for d in h.read(SOURCE/'validated/full_track/frame_decisions.json')['frames']}
    assert sorted(scene.records)==sorted(decisions)==list(range(0,2000,5))
    region=h.load(SOURCE/'e1/region.npz');prior,owner=region['prior_owner'],region['owner']
    apoints=np.flatnonzero((prior!=-1)|(owner!=-1))
    baseline_records=origfreeze['scenes'][SCENE]['predictions']
    baseline=h.load(baseline_records['strict_baseline_final']['path'],baseline_records['strict_baseline_final']['sha256'])
    baseroot=Path(baseline_records['strict_baseline_final']['path']).parent.parent
    e0=h.load(baseroot/'surface_evidence.npz'); bk,bv=h.encode(h.load(baseroot/'surface_instance_frame_votes.npz'),base)
    np.testing.assert_array_equal(scene.xyz,baseline['xyz_m']);np.testing.assert_array_equal(scene.rgb,baseline['rgb'])
    can_to_pid=np.zeros(max(fixed)+1,np.int64)
    for cid,pid in fixed.items():can_to_pid[cid]=pid
    dump(R/'policy.json',dict(GT_read_by_mapper=False,selected=COUNT,group=GROUP,source=str(SOURCE),
        replacement='Unique new persistent ID replaces all old candidates at this frame and projected TSDF point; different new IDs preserve original candidates',
        old_family_limits=False,history='Unchanged frozen E1 core then expanded-region relabeling of retained original observations; no invented observations',
        conflicts='New-new conflict preserves original evidence in direct stage; unchanged E1 history may subsequently revise it',
        fixed=['seeds','tracking','gates','2D overlap abstention','association','core','expanded region','visibility','TSDF','thresholds','postprocessing','R2'],
        stages=list(STAGES),control='Frozen baseline/whole/partial retained and checked; no original output overwritten'))
    framekeys={n:[] for n in ('baseline',*STAGES)}; removed={n:[] for n in STAGES};added={n:[] for n in STAGES};scopes={n:[] for n in STAGES}
    receipts=[]; partialtx=Path(CFG['partial_transactions'])
    for index,fid in enumerate(sorted(scene.records)):
        frame=scene.source.load_frame(fid);rec=scene.records[fid]
        support=h.load(rec['support_file'],rec['support_sha256']); p=support['surface_point_index'].astype(np.int64);mid=support['mask_local_id'].astype(np.int64)
        maskpath=scene.source.mask_root/scene.source.config['source']['mask_pattern'].format(frame=fid)
        h.protect(maskpath,str(support['source_mask_sha256'][0]))
        table=np.zeros(int(frame.mask_local.max())+1,np.int64)
        for m in np.unique(frame.mask_local):
            if m:table[m]=scene.lookup[(fid,int(m))]
        baseline_raw=h.keys(p,table[mid],base);baseline_eff=obs.strict_frame(baseline_raw,base)
        merged=adopted_mask(frame,decisions[fid],obj,info,canonical)
        maskp,maskcid=h.project(frame,merged,scene); newkeys=h.keys(maskp,can_to_pid[maskcid],base)
        # Every fresh new projection must reproduce the frozen original direct candidates.
        txpath=SOURCE/'original_repair'/SCENE/'full_track/transactions'/f'f{fid:06d}.npz'
        whole_eff=baseline_eff
        if txpath.exists():
            tx=h.load(txpath,origfreeze['generated_output_sha256'][str(txpath)])
            np.testing.assert_array_equal(baseline_raw,tx['old_candidate_keys'])
            np.testing.assert_array_equal(np.union1d(tx['retained_old_candidate_keys'],newkeys),tx['new_candidate_keys'])
            whole_eff=tx['new_frame_keys']
        else:assert not len(newkeys)
        saved=h.load(partialtx/f'f{fid:06d}.npz')
        np.testing.assert_array_equal(saved['baseline_candidate_keys'],baseline_raw)
        np.testing.assert_array_equal(saved['baseline_frame_keys'],baseline_eff)
        direct,retained,committed,conflicts=replace_frame_points(baseline_raw,newkeys,base)
        commitp=committed//base
        np.testing.assert_array_equal(direct[~np.isin(direct//base,commitp)],baseline_raw[~np.isin(baseline_raw//base,commitp)])
        np.testing.assert_array_equal(direct[np.isin(direct//base,conflicts)],baseline_raw[np.isin(baseline_raw//base,conflicts)])
        rp,old=retained//base,retained%base; available=np.ones(len(retained),bool)
        vis=np.zeros(len(scene.xyz),bool)
        vis[apoints],_,_=obs.visible_samples(scene.xyz[apoints],frame.camera_to_world,frame.camera,frame.depth_m,.02)
        coreids=old.copy();take=prior[rp]>0;coreids[take]=prior[rp[take]]
        expanded,_=obs.revise_all_history(rp,old,available,prior,owner,vis)
        raw={'direct':direct,'core_all':np.union1d(h.keys(rp,coreids,base),committed),'e1':np.union1d(h.keys(rp,expanded,base),committed)}
        eff={n:obs.strict_frame(raw[n],base) for n in STAGES};eff['baseline']=baseline_eff
        assert transition(baseline_eff,eff['direct'],base)['valid_to_zero']==0
        # Point-level history is equivalent to applying the frozen rule to all retained source supports.
        keep=~np.isin(p,commitp); sid,_=obs.revise_all_history(p,table[mid],keep,prior,owner,vis)
        np.testing.assert_array_equal(h.keys(p[keep],sid[keep],base),h.keys(rp,expanded,base))
        txout={'baseline_candidate_keys':baseline_raw,'new_mask_candidate_keys':newkeys,'committed_keys':committed,'conflict_points':conflicts}
        stats={}
        for name in STAGES:
            parent=PARENT[name];before=eff[parent];after=eff[name]
            assert len(after)==len(np.unique(after//base))
            rem=np.setdiff1d(before,after);add=np.setdiff1d(after,before)
            scope=np.unique(np.r_[rem//base,add//base])
            if name=='direct':assert np.all(np.isin(scope,commitp));scope=commitp
            elif name=='core_all':assert np.all(prior[scope]>0)
            else:assert np.all((prior[scope]!=-1)|(owner[scope]!=-1))
            framekeys[name].append(after);removed[name].append(rem);added[name].append(add);scopes[name].append(scope)
            txout[name+'_candidate_keys']=raw[name];txout[name+'_frame_keys']=after
            stats[name]=dict(transition(before,after,base),removed=len(rem),added=len(add))
        framekeys['baseline'].append(baseline_eff)
        h.f.atomic_npz(R/'transactions'/f'f{fid:06d}.npz',**txout)
        receipts.append(dict(frame=fid,committed_points=len(commitp),new_conflict_points=len(conflicts),
            whole=transition(baseline_eff,whole_eff,base),partial=transition(baseline_eff,saved['partial_direct_frame_keys'],base),**stats))
        if (index+1)%25==0:emit('replay',frames=index+1,total_frames=400,seconds=time.time()-started)
    pk,pv=aggregate(framekeys.pop('baseline'));np.testing.assert_array_equal(pk,bk);np.testing.assert_array_equal(pv,bv)
    reduced=h.r.reduce_surface_votes(bk,bv,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
    for field in h.r.FIELDS:np.testing.assert_array_equal(reduced[field],e0[field])
    results={'baseline':(bk,bv,e0,baseline['instance_id'])};reports={}
    for name in STAGES:
        emit('reduce',variant=name)
        pk,pv,pe,pf=results[PARENT[name]];rem=cat(removed.pop(name));add=cat(added.pop(name))
        nk,nv=h.batch_delta(pk,pv,rem,add);backk,backv=h.batch_delta(nk,nv,add,rem)
        np.testing.assert_array_equal(backk,pk);np.testing.assert_array_equal(backv,pv)
        fullk,fullv=aggregate(framekeys.pop(name));np.testing.assert_array_equal(nk,fullk);np.testing.assert_array_equal(nv,fullv)
        scope=np.unique(cat(scopes.pop(name))).astype(np.int32)
        if name=='e1':
            common,ni,pi=np.intersect1d(nk,pk,return_indices=True)
            scope=np.unique(np.r_[np.setxor1d(nk,pk)//base,common[nv[ni]!=pv[pi]]//base]).astype(np.int32)
        monitor=np.zeros(len(scene.xyz),bool);monitor[scope]=True
        np.testing.assert_array_equal(nk[~monitor[nk//base]],pk[~monitor[pk//base]])
        np.testing.assert_array_equal(nv[~monitor[nk//base]],pv[~monitor[pk//base]])
        e=h.r.reduce_surface_votes(nk,nv,len(scene.xyz),base,scene.report['min_confirmed_votes'],scene.report['min_confirmed_ratio'])
        for field in h.r.FIELDS:np.testing.assert_array_equal(e[field][~monitor],pe[field][~monitor])
        native=np.where(e['state']==h.r.CONFIRMED,e['top1_instance_id'],-1).astype(np.int32);pair=h.pairtable(nk,nv,base)
        if len(scope):
            variants,_,post=h.r.assign_surface(scene.xyz,scene.normals,scene.rgb,e,pair,native,scene.settings,
                progress=lambda s,v:emit('postprocess',variant=name,substage=s,values=v))
            final=pf.copy();final[scope]=variants['holes_geodesic'][scope];del variants
        else:final=pf.copy();post={'no_changed_scope':True}
        np.testing.assert_array_equal(final[~monitor],pf[~monitor])
        folder=R/name/SCENE
        reference={'direct':f'original_repair/{SCENE}/full_track','core_all':f'modified_repair/{SCENE}','e1':f'e1/E1_expanded_history/{SCENE}'}[name]
        for stage,labels in [('native',native),('final',final)]:
            oldmap=h.load(SOURCE/reference/stage/'instance_surface.npz')
            inventory=np.union1d(oldmap.get('native_instance_ids',np.unique(oldmap['instance_id'][oldmap['instance_id']>0])),np.unique(labels[labels>0])).astype(np.int32)
            h.f.atomic_npz(folder/stage/'instance_surface.npz',xyz_m=scene.xyz,rgb=scene.rgb,instance_id=labels,native_instance_ids=inventory)
        h.f.atomic_npz(folder/'native/surface_evidence.npz',xyz_m=scene.xyz,rgb=scene.rgb,**e)
        h.f.atomic_npz(folder/'native/surface_instance_frame_votes.npz',**pair)
        h.f.atomic_npz(folder/'allowed_surface_ids.npz',surface_point_index=scope)
        newly=(pe['state']==2)&(e['state']!=2)
        reports[name]=dict(status='PASS',parent=PARENT[name],scope_points=len(scope),removed_votes=len(rem),added_votes=len(add),
            valid_to_zero_frame_points=sum(x[name]['valid_to_zero'] for x in receipts),
            zero_to_valid_frame_points=sum(x[name]['zero_to_valid'] for x in receipts),
            confirmed_to_unassigned_points=int(newly.sum()),
            confirmed_to_unassigned_with_non_decreased_total_votes=int((newly&(e['total_frame_votes']>=pe['total_frame_votes'])).sum()),
            final_changed_vs_parent=int((final!=pf).sum()),summary=h.summary(e,final),
            native_state_transition=np.bincount(pe['state'].astype(np.int64)*4+e['state'],minlength=16).reshape(4,4).tolist(),postprocessing=post)
        dump(folder/'complete.json',reports[name]);results[name]=(nk,nv,e,final)
        del fullk,fullv,backk,backv,rem,add;gc.collect()
    dump(R/'frame_receipts.json',receipts);scene.verify_unchanged()
    for p,d in h.protected.items():assert sha(p)==d,p
    for p,d in codehash.items():assert sha(p)==d,p
    verify_sources()
    outputs={str(p):sha(p) for n in STAGES for p in (R/n).rglob('*.npz')}
    dump(R/'predictions_freeze.json',dict(status='PREDICTIONS_FROZEN_BEFORE_GT',GT_read_by_mapper=False,
        code_sha256=codehash,source_sha256=h.protected,output_sha256=outputs,policy_sha256=sha(R/'policy.json'),baseline_modified=False,
        checks=dict(baseline_full_replay_exact=True,fresh_new_projection_matches_frozen_whole_candidates=True,
            direct_no_valid_vote_lost=True,direct_outside_and_conflict_candidates_unchanged=True,
            history_original_rule_and_per_support_equivalence=True,all_stages_full_replay_equals_delta=True,
            all_stages_exact_rollback=True,outside_stage_scope_unchanged=True,strict_one_vote_per_frame_point=True),variants=STAGES))
    dump(R/'mapping_complete.json',dict(status='PASS',seconds=time.time()-started,stages=reports,
        direct_controls={n:{k:sum(x[n][k] for x in receipts) for k in ['valid_to_zero','zero_to_valid','valid_id_changed']} for n in ['whole','partial','direct']},
        committed_frame_points=sum(x['committed_points'] for x in receipts),new_conflict_frame_points=sum(x['new_conflict_points'] for x in receipts)))
    emit('mapping_complete',seconds=time.time()-started)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(R/'mapping_failure.json',dict(traceback=traceback.format_exc()));raise
