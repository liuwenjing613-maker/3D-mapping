def gate_online(scene,cases,objects,canonical,policy,association,tracking):
    condition='full_track'
    out=REPAIR_ROOT/'full_track';out.mkdir(parents=True,exist_ok=True)
    fixed={row['canonical_track_id']:row['persistent_id'] for row in association['objects']};base=association['instance_base']
    old_frames=[];new_frames=[];removed=[];added=[];scopes=[];retired_points=[];decisions=[];observations=[]
    counts={'mapping_frames':400,'updated_frames':0,'whole_observations':0,'partial_observations':0,
            'cross_case_conflict_pixels':0,'accepted_frames_by_track':{str(oid):0 for oid in objects},
            'updated_frames_by_case':{uid:0 for uid in cases}}
    family_targets={pid:{canonical[oid] for oid,obj in objects.items() if obj['old_family_id']==pid}
                    for pid in {obj['old_family_id'] for obj in objects.values()}}
    for index,fid in enumerate(sorted(scene.records)):
        frame,table,old_keys,_=scene.old_frame(fid,base);old_frames.append(old_keys)
        maskpath=out/'accepted_masks'/('f%06d.png'%fid);maskpath.parent.mkdir(exist_ok=True)
        Image.fromarray(np.zeros(frame.mask_local.shape,np.uint16)).save(maskpath)
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
                ok,reasons=object_reliability(metric,policy)
                if not family_local[uid]:ok=False;reasons.append('no_identified_old_family_observation')
                checks[str(oid)]={**metric,'accepted':ok,'reasons':reasons}
                if ok:accepted.append(oid)
        initial_checks=copy.deepcopy(checks)
        if not accepted:new_frames.append(old_keys);decisions.append({'frame':fid,'objects':checks,'initial_objects':initial_checks,'accepted_track_ids':[]});continue
        proposals=[(oid,labels_by_case[objects[oid]['case_uid']]==oid) for oid in accepted]
        merged,conflict=j.merge_foregrounds(proposals,canonical)
        counts['cross_case_conflict_pixels']+=int(conflict.sum())
        # Recheck trimmed masks with the same V1 gate, preserving the initial checks for diagnosis.
        if conflict.any():
            revised_accepted=[]
            for uid,data in active.items():
                trimmed=labels_by_case[uid].copy();trimmed[conflict]=0;labels_by_case[uid]=trimmed
                p,t,_=r.project_frame_regions(replace(frame,mask_local=trimmed),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
                for oid in [x for x in accepted if objects[x]['case_uid']==uid]:
                    metric=metrics_for(oid,trimmed==oid,p[t==oid],visible,objects,data,frame,family_local[uid],scene,policy)
                    ok,reasons=object_reliability(metric,policy)
                    checks[str(oid)]={**metric,'accepted':ok,'reasons':reasons,'rechecked_after_overlap_abstention':True}
                    if ok:revised_accepted.append(oid)
            accepted=sorted(revised_accepted)
            if accepted:merged,_=j.merge_foregrounds([(oid,labels_by_case[objects[oid]['case_uid']]==oid) for oid in accepted],canonical)
        if not accepted:new_frames.append(old_keys);decisions.append({'frame':fid,'objects':checks,'initial_objects':initial_checks,'accepted_track_ids':[]});continue
        Image.fromarray(merged.astype(np.uint16)).save(maskpath)
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
        original_plans=dict(plans)
        plans={mid:'partial' for mid in plans}
        decisions.append({'frame':fid,'objects':checks,'initial_objects':initial_checks,'accepted_track_ids':accepted,'accepted_canonical_ids':sorted(accepted_canonical),
                          'old_observation_plans':plans,'source_plan_types_before_partial':original_plans,'cross_case_conflict_pixels':int(conflict.sum())})
        counts['updated_frames']+=1
        for oid in accepted:counts['accepted_frames_by_track'][str(oid)]+=1
    counts['mapping_frames']=len(scene.records)
    f.atomic_json(out/'frame_decisions.json',{'status':'PASS','counts':counts,'frames':decisions,'whole_control_replayed':False})
    assert sorted(x['frame'] for x in decisions)==sorted(scene.records)
    return counts
