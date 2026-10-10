"""Score frozen two-scene predictions through main's current default R2 CLI.

GT is first read here, after both new map predictions have been frozen. All
before/after adapters use the original locked geometry correspondence.
"""
from pathlib import Path
import csv,gc,hashlib,json,subprocess,sys,time,traceback
import numpy as np

MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
ROOT=Path('/data/chenkejun/CVPR/results/p1a1_history_core_reassociation_20261009')
STRICT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
OUT=ROOT/'evaluation_r2'
sys.path.insert(0,str(MAIN))
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,REGISTRY_PATH,require_current_protocol,require_current_gt
from unified_eval.io import sha256_file,load_gt,load_prediction
from unified_eval.metrics import build_overlap,owner_coverage
from unified_eval.geometry import load_fixed_surface_correspondence
from unified_eval.schema import Protocol

protected={}
def protect(path,expected=None):
    p=Path(path);d=sha256_file(p)
    if expected is not None:assert d==expected,str(p)
    if str(p) in protected:assert protected[str(p)]==d
    protected[str(p)]=d;return d
def read(path):protect(path);return json.loads(Path(path).read_text())
def dump(path,value):
    p=Path(path);assert not p.exists(),str(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def emit(stage,**data):print(json.dumps({'stage':stage,**data,'unix':time.time()}),flush=True)
def cli(args):
    # Do not pass --config: use the verified current default entry in main.
    result=subprocess.run([sys.executable,'-m','unified_eval.cli',*map(str,args)],cwd=MAIN,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
    print(result.stdout.strip(),flush=True)
def fields(s):
    rows=s['per_scene'];n=sum(r['diagnostics']['gt_instance_count'] for r in rows)
    return {'F1':s['CA_PRF1_0_5']['F1'],'PQ':s['CA_PQ']['PQ'],'AP':s['CA_AP_uniform'],
        'AP50':s['CA_AP50_uniform'],'mCov':s['CA_mCov'],
        'macro_best_GT_recall':sum(r['diagnostics']['macro_best_gt_recall']*r['diagnostics']['gt_instance_count'] for r in rows)/n,
        **{k:s['CA_PRF1_0_5'][k] for k in ('TP','FP','FN')},
        **{k:s['owner_surface'][k] for k in ('correct_owner_vertices','wrong_owner_vertices','unassigned_with_geometry_target_vertices',
            'no_geometry_target_vertices','target_vertices','Correct_owner_Coverage','Wrong_owner_Coverage','Unassigned_Coverage',
            'No_geometry_Coverage','duplicate_coordinate_conflict_target_vertices')},
        'split_GT':s['structure']['split_gt_count'],'merge_predictions':s['structure']['merge_prediction_count']}
def independent_reference_owners(native,c):
    labels=np.where(native>0,native,-1);inverse=c.native_unique_index
    pairs=np.unique(np.column_stack((inverse,labels)),axis=0)
    groups,first,count=np.unique(pairs[:,0],return_index=True,return_counts=True)
    owners=np.full(len(c.unique_first_index),-1,np.int64)
    owners[groups[count==1]]=pairs[first[count==1],1]
    result=np.full(len(c.nearest_native_index),-1,np.int64);exists=c.nearest_native_index>=0
    result[exists]=owners[inverse[c.nearest_native_index[exists]]]
    return result
def quality(gt,pred,protocol,native,c):
    overlap=build_overlap(gt,pred,protocol);score=owner_coverage(overlap)
    alignment={str(r['instance_uid']):int(r['raw_gt_id']) for r in score['scoring_owner_alignment']}
    identity=np.full(gt.vertex_count,-1,np.int64);matched=np.full(gt.vertex_count,-1,np.int64)
    for instance in pred.instances:
        v=instance.vertex_indices;assert np.all(identity[v]<0)
        identity[v]=int(instance.metadata['native_instance_id']);matched[v]=alignment.get(str(instance.instance_uid),-1)
    np.testing.assert_array_equal(identity,independent_reference_owners(native,c))
    target=overlap.gt_object_mask;exists=c.nearest_native_index>=0
    q=np.zeros(gt.vertex_count,np.uint8);q[target&exists&(identity<=0)]=1;q[target&exists&(identity>0)]=3
    q[target&exists&(identity>0)&(matched==gt.instance_id)]=2
    assert int(np.sum(target&(q==2)))==score['correct_owner_vertices']
    assert int(np.sum(target&(q==3)))==score['wrong_owner_vertices']
    return {'overlap':overlap,'quality':q,'identity':identity,'target':target,'alignment':alignment}
def matrix(a,b,mask):return np.bincount(a[mask].astype(np.int64)*4+b[mask],minlength=16).reshape(4,4).astype(int)
def grouped(rows):
    result={'objects':len(rows)}
    for key in ('iou','best_recall'):
        old='before_best_iou' if key=='iou' else 'before_best_recall'
        new='after_best_iou' if key=='iou' else 'after_best_recall'
        delta='delta_iou' if key=='iou' else 'delta_best_recall'
        result['mean_before_'+key]=float(np.mean([r[old] for r in rows])) if rows else None
        result['mean_after_'+key]=float(np.mean([r[new] for r in rows])) if rows else None
        result['mean_delta_'+key]=float(np.mean([r[delta] for r in rows])) if rows else None
        result[key+'_improved']=sum(r[delta]>1e-12 for r in rows)
        result[key+'_degraded']=sum(r[delta]<-1e-12 for r in rows)
        result[key+'_unchanged']=sum(abs(r[delta])<=1e-12 for r in rows)
        result[key+'_improved_over_1pp']=sum(r[delta]>.01 for r in rows)
        result[key+'_degraded_over_1pp']=sum(r[delta]<-.01 for r in rows)
    result['degraded_over_1pp']=[r for r in rows if r['delta_iou']<-.01 or r['delta_best_recall']<-.01]
    result['rows']=rows
    return result
def enrich_pair(p,qa,qb,targets,scene,scope_objects):
    old,new=qa['overlap'],qb['overlap']
    np.testing.assert_array_equal(old.gt_ids,new.gt_ids);np.testing.assert_array_equal(old.gt_size,new.gt_size)
    assert len(p['per_gt'])==len(new.gt_ids)
    for row in p['per_gt']:
        idx=int(np.flatnonzero(new.gt_ids==row['raw_gt_id'])[0])
        row.update(scene_id=scene,semantic_class=scope_objects[row['raw_gt_id']]['semantic_class'],
            selected_target=row['raw_gt_id'] in targets,before_best_recall=float(old.recall[:,idx].max()),
            after_best_recall=float(new.recall[:,idx].max()))
        row['delta_best_recall']=row['after_best_recall']-row['before_best_recall']
    p['groups']={g:grouped([r for r in p['per_gt'] if r['selected_target']==choice])
        for g,choice in [('selected_targets',True),('other_targets',False)]}
    p['fixed_selected_physical_raw_GT_ids']=sorted(targets)
    p['target_definition']='Exact reference-vertex lineage of frozen human target list, intersect current evaluable GT; posthoc only'
    mask=qa['target'];np.testing.assert_array_equal(mask,qb['target'])
    change=mask&(qa['identity']!=qb['identity']);same=mask&~change
    allm=matrix(qa['quality'],qb['quality'],mask);cm=matrix(qa['quality'],qb['quality'],change);sm=matrix(qa['quality'],qb['quality'],same)
    np.testing.assert_array_equal(cm+sm,allm)
    assert int(allm[:,2].sum()-allm[2,:].sum())==p['correct_owner_vertex_gain']
    p.update(quality_order=['no_geometry','unassigned_with_geometry','correct_owner','wrong_owner'],
        quality_transition_matrix=allm.tolist(),quality_transition_on_changed_identity_vertices=cm.tolist(),
        quality_transition_on_unchanged_identity_vertices=sm.tolist(),changed_identity_target_reference_vertices=int(change.sum()),
        correct_owner_gain_at_changed_identity_vertices=int(cm[:,2].sum()-cm[2,:].sum()),
        correct_owner_gain_at_unchanged_identity_vertices=int(sm[:,2].sum()-sm[2,:].sum()),
        owner_alignment_changes={uid:[qa['alignment'].get(uid),qb['alignment'].get(uid)]
            for uid in sorted(set(qa['alignment'])|set(qb['alignment'])) if qa['alignment'].get(uid)!=qb['alignment'].get(uid)})
    return p

def main():
    start=time.time();assert not OUT.exists()
    frozen=read(ROOT/'predictions_freeze.json');freeze_sha=sha256_file(ROOT/'predictions_freeze.json')
    assert frozen['status']=='TWO_SCENE_PREDICTIONS_FROZEN_BEFORE_NEW_R2_SCORING'
    assert frozen['GT_read_by_mapper'] is False and frozen['GT_used_for_masks_frames_IDs_or_thresholds'] is False
    assert frozen['baseline_modified'] is False
    for p,d in frozen['output_sha256'].items():protect(p,d)
    for p,d in frozen['source_sha256'].items():protect(p,d)
    for p,d in frozen['code_sha256'].items():protect(p,d)
    protect(ROOT/'policy.json',frozen['policy_sha256'])
    tests=read(ROOT/'tests_complete.json');assert tests['status']=='PASS' and tests['tests']==8
    assert require_current_protocol(CURRENT_CONFIG)
    protect(CURRENT_CONFIG,CURRENT_PROTOCOL['config_sha256']);protect(REGISTRY_PATH);protect(__file__)
    protocol=Protocol.from_dict(read(CURRENT_CONFIG));assert protocol.profile_revision==2
    assert CURRENT_PROTOCOL['evaluable_GT_count']==350 and CURRENT_PROTOCOL['frozen'] is False
    commit=subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip()
    assert commit=='29304d8523f55f0f628cdf7dec904b6980f4a9ad'
    for name,d in CURRENT_PROTOCOL['evaluator_code_sha256'].items():protect(MAIN/'unified_eval'/name,d)
    parent=read(STRICT/'predictions_freeze.json');oldcomp=read(STRICT/'evaluation_r2/comparison.json')
    lineage=read(STRICT/'evaluation_r2/analysis/targets_and_attribution_complete.json')
    assert lineage['status']=='PASS' and lineage['profile_revision']==2
    assert lineage['main_metrics_sha256_unchanged']==sha256_file(STRICT/'evaluation_r2/comparison.json')
    # Protect the GT/reference lineage used to define selected vs other objects.
    for p,d in lineage['protected_sha256'].items():protect(p,d)
    dump(OUT/'evaluation_protocol_freeze.json',{'status':'CURRENT_DEFAULT_R2_LOCKED_BEFORE_SCORING',
        'predictions_both_scenes_frozen_before_new_scoring':True,'prediction_freeze_sha256':freeze_sha,
        'current_protocol':CURRENT_PROTOCOL,'entry_commit':commit,'default_main_CLI_used':True,
        'GT_scope_rebuilt':False,'diagnosis_informed_development_test':True,'pre_registered_benchmark':False})
    scores={};pairs={};canonical={};states={};rows=[];core_truth={}
    for scene in ('room0','room2'):
        done=read(ROOT/scene/'complete.json');assert done['status']=='PASS' and all(done['checks'].values())
        trimmed={k:v for k,v in done.items() if k not in ('postprocessing_statistics','frame_receipts')}
        assert trimmed==frozen['scenes'][scene]
        gtpath=Path(CURRENT_PROTOCOL['canonical_GT'][scene]['file']);assert require_current_gt(gtpath)==scene
        gt=load_gt(gtpath);assert len(np.unique(gt.instance_id[gt.valid_vertex_mask&~gt.ignore_vertex_mask]))==CURRENT_PROTOCOL['canonical_GT'][scene]['evaluable_GT_count']
        scope=read(gtpath.parent/'gt_scope.json');scope_objects={r['raw_gt_id']:r for r in scope['objects']}
        targets=set(lineage['scenes'][scene]['selected_physical_raw_GT_ids'])
        expected=25 if scene=='room0' else 20;assert len(targets)==expected
        cache=STRICT/'evaluation_r2/geometry_cache'/scene/'fixed_gt_to_tsdf_r2.npz';cache_sha=protect(cache)
        controlrecords=parent['scenes'][scene]['predictions']
        with np.load(controlrecords['strict_full_track_final']['path']) as z:xyz=z['xyz_m'].copy()
        c=load_fixed_surface_correspondence(cache,xyz,gt.xyz_ref)
        materialpath=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native')/scene/'P1-A1/surface_p0/materialization_report.json'
        material=read(materialpath);frames=[r['frame_id'] for r in material['frame_records']]
        assert frames==gt.metadata['observation_metadata']['frame_ids'] and len(frames)==400
        frame_sha=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest()
        assert frame_sha==gt.metadata['observation_metadata']['frame_list_sha256']
        scores[scene]={};pairs[scene]={};states[scene]={};canonical[scene]={}
        for stage in ('native','final'):
            controls={};qualities={}
            for cond in ('strict_baseline','strict_full_track'):
                name=cond+'_'+stage;path=STRICT/'evaluation_r2/evaluation'/scene/name
                manifest=read(path/'adapter_manifest.json');predpath=path/'canonical_prediction.npz'
                protect(controlrecords[name]['path'],controlrecords[name]['sha256'])
                assert manifest['source_surface_sha256']==controlrecords[name]['sha256']
                protect(predpath,manifest['canonical_prediction_sha256'])
                protect(path/'diagnostic_support_prediction.npz',manifest['diagnostic_prediction_sha256'])
                assert manifest['correspondence_file_sha256']==cache_sha and manifest['profile_revision']==2
                assert manifest['profile_config_sha256']==CURRENT_PROTOCOL['config_sha256']
                controls[cond]=predpath;pred=load_prediction(predpath)
                with np.load(controlrecords[name]['path']) as z:labels=z['instance_id'].copy();np.testing.assert_array_equal(xyz,z['xyz_m'])
                qualities[cond]=quality(gt,pred,protocol,labels,c)
                scores[scene][name]=fields(read(path/'metrics.json')['summary'])
                assert scores[scene][name]==oldcomp['per_scene'][scene][name]
                canonical[scene][name]=path
                states[scene][name]=oldcomp['states'][scene][name]
            name='history_core_'+stage;surface=ROOT/scene/stage/'instance_surface.npz';dest=OUT/'evaluation'/scene/name
            protect(surface,frozen['output_sha256'][str(surface)])
            with np.load(surface) as z:np.testing.assert_array_equal(xyz,z['xyz_m']);newlabels=z['instance_id'].copy()
            provenance=OUT/'provenance'/scene/(name+'.json')
            dump(provenance,{'input_scope_verified':True,'frame_count':400,'frame_list_sha256':frame_sha,
                'input_config_sha256':material['config_sha256'],'materialization_report_sha256':sha256_file(materialpath),
                'GT_read_by_mapper':False,'GT_used_for_masks_frames_IDs_or_thresholds':False,
                'prediction_freeze_sha256':freeze_sha,'diagnosis_informed_development_test':True,'pre_registered_benchmark':False,
                'full_400_frame_replay_equals_delta':True,'exact_rollback':True,'outside_scope_bit_identical':True,
                'fixed_R2_correspondence_reused':str(cache),'mapping_correctness_checks':done['checks']})
            emit('default_main_R2_adapt_and_score',scene=scene,prediction=name)
            cli(['adapt-surface','--gt',gtpath,'--surface',surface,'--fixed-correspondence',cache,
                '--source-provenance',provenance,'--method-name','P1-A1_history_core_'+scene+'_'+stage,
                '--method-commit',frozen['code_sha256'][str(Path(__file__).parent/'history_core_ops.py')],'--out',dest])
            am=read(dest/'adapter_manifest.json');assert am['profile_revision']==2 and am['profile_config_sha256']==CURRENT_PROTOCOL['config_sha256']
            assert am['correspondence_file_sha256']==cache_sha
            for cond in controls:
                cm=read(canonical[scene][cond+'_'+stage]/'adapter_manifest.json')
                for key in ('geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256'):
                    assert cm[key]==am[key],key
            newpred=load_prediction(dest/'canonical_prediction.npz');qn=quality(gt,newpred,protocol,newlabels,c)
            scores[scene][name]=fields(read(dest/'metrics.json')['summary']);canonical[scene][name]=dest
            if stage=='final':
                # Posthoc only: identify cores spanning several physical GT, and
                # separate persistent targets claiming parts of the same GT.
                with np.load(ROOT/scene/'core_claims.npz') as z:
                    owner=np.full(len(xyz),-1,np.int32)
                    owner[z['surface_point_index']]=z['claimed_persistent_id']
                nearest=c.nearest_native_index;exists=nearest>=0
                ref_owner=np.full(gt.vertex_count,-1,np.int32);ref_owner[exists]=owner[nearest[exists]]
                uidrows={str(uid):i for i,uid in enumerate(qn['overlap'].pred_uids)}
                pred_rows={int(inst.metadata['native_instance_id']):uidrows[str(inst.instance_uid)]
                    for inst in newpred.instances if str(inst.instance_uid) in uidrows}
                truthrows=[];dominant_groups={}
                for pid in sorted(set(o['persistent_id'] for o in done['objects'])):
                    select=qn['target']&(ref_owner==pid);ids,ns=np.unique(gt.instance_id[select],return_counts=True)
                    dominant=int(ids[np.argmax(ns)]) if len(ids) else None
                    part=float(ns.max()/ns.sum()) if len(ids) else None
                    row={'persistent_id':pid,'core_native_points':int(np.sum(owner==pid)),
                        'qualified_reference_vertices':int(select.sum()),'core_GT_histogram':{str(int(k)):int(v) for k,v in zip(ids,ns)},
                        'dominant_raw_GT_id':dominant,'dominant_GT_fraction':part,
                        'core_reference_assigned_to_own_ID':int(np.sum(select&(qn['identity']==pid))),
                        'core_reference_scored_correct':int(np.sum(select&(qn['quality']==2))),
                        'member_seed_masks':[{'case_uid':o['case_uid'],'native_mask_id':o['native_mask_id'],'track_id':o['track_id'],'ROI':o['ROI']} for o in done['objects'] if o['persistent_id']==pid]}
                    if dominant is not None and pid in pred_rows:
                        j=int(np.flatnonzero(qn['overlap'].gt_ids==dominant)[0]);i=pred_rows[pid]
                        row.update(own_prediction_IoU_on_dominant_GT=float(qn['overlap'].iou[i,j]),
                            own_prediction_recall_on_dominant_GT=float(qn['overlap'].recall[i,j]))
                    truthrows.append(row)
                    if dominant is not None:dominant_groups.setdefault(str(dominant),[]).append(pid)
                core_truth[scene]={'GT_used_to_change_predictions':False,'reference_based_posthoc_only':True,
                    'targets':truthrows,'distinct_target_IDs_with_same_dominant_physical_GT':{k:v for k,v in dominant_groups.items() if len(v)>1}}
                dump(OUT/'analysis'/scene/'core_identity_posthoc.json',core_truth[scene])
            with np.load(ROOT/scene/'native/surface_evidence.npz') as z:ns=z['state'].copy()
            if stage=='final':assert not np.any((newlabels<=0)&(ns==2))
            category=ns if stage=='native' else np.where(newlabels>0,2,ns).astype(np.uint8)
            states[scene][name]={'counts_U_T_assigned_C':np.bincount(category,minlength=4).astype(int).tolist(),
                'three_state_total':int(np.sum(newlabels<=0)),
                'r2_duplicate_coordinate_conflicts':am['adapter_statistics']['conflicting_exact_coordinate_group_count']}
            for cond,predpath in controls.items():
                pairdest=OUT/'paired'/scene/(name+'_vs_'+cond)
                cli(['eval-repair-pair','--gt',gtpath,'--before',predpath,'--after',dest/'canonical_prediction.npz','--out',pairdest])
                pm=read(pairdest/'paired_repair_metrics.json')
                pair=enrich_pair(pm,qualities[cond],qn,targets,scene,scope_objects)
                dump(pairdest/'paired_with_completeness_and_attribution.json',pair)
                pairs[scene][name+'_vs_'+cond]=pair
            emit('scene_stage_scored',scene=scene,stage_name=stage,scores=scores[scene][name])
            del qualities,qn,pred,newpred;gc.collect()
        for name,score in scores[scene].items():
            state=states[scene][name];counts=state['counts_U_T_assigned_C']
            rows.append({'scope':scene,'condition':name,**score,'U':counts[0],'T':counts[1],'C':counts[3],'gray_total':state['three_state_total']})
        del gt,c;gc.collect()
    pooled={};pooled_states={};pooled_pairs={}
    for stage in ('native','final'):
        for cond in ('strict_baseline','strict_full_track','history_core'):
            name=cond+'_'+stage;manifest=OUT/'pooled_inputs'/(name+'.json');dest=OUT/'pooled'/name
            entries=[{'gt':CURRENT_PROTOCOL['canonical_GT'][s]['file'],'prediction':str(canonical[s][name]/'canonical_prediction.npz'),
                'diagnostic_prediction':str(canonical[s][name]/'diagnostic_support_prediction.npz')} for s in ('room0','room2')]
            dump(manifest,{'scenes':entries});emit('default_main_R2_pooled_scoring',prediction=name)
            cli(['eval-batch','--scenes',manifest,'--out',dest]);s=read(dest/'summary.json')
            pooled[name]=fields(s)
            if cond!='history_core':assert pooled[name]==oldcomp['pooled'][name],name
            counts=np.sum([states[scene][name]['counts_U_T_assigned_C'] for scene in ('room0','room2')],axis=0).astype(int).tolist()
            pooled_states[name]={'counts_U_T_assigned_C':counts,'three_state_total':sum(counts[i] for i in (0,1,3))}
            rows.append({'scope':'pooled_room0_room2','condition':name,**pooled[name],'U':counts[0],'T':counts[1],'C':counts[3],'gray_total':pooled_states[name]['three_state_total']})
        for cond in ('strict_baseline','strict_full_track'):
            key='history_core_'+stage+'_vs_'+cond;pp=[pairs[scene][key] for scene in ('room0','room2')]
            pergt=[r for p in pp for r in p['per_gt']];assert len(pergt)==124
            pooled_pairs[key]={'per_gt':pergt,'groups':{g:grouped([r for r in pergt if r['selected_target']==choice])
                for g,choice in [('selected_targets',True),('other_targets',False)]},
                'correct_owner_vertex_gain':sum(p['correct_owner_vertex_gain'] for p in pp),
                'correct_owner_gain_at_changed_identity_vertices':sum(p['correct_owner_gain_at_changed_identity_vertices'] for p in pp),
                'correct_owner_gain_at_unchanged_identity_vertices':sum(p['correct_owner_gain_at_unchanged_identity_vertices'] for p in pp)}
            assert pooled_pairs[key]['groups']['selected_targets']['objects']==45
            assert pooled_pairs[key]['groups']['other_targets']['objects']==79
    for p,d in protected.items():assert sha256_file(p)==d,p
    for p,d in frozen['output_sha256'].items():assert sha256_file(p)==d,p
    assert require_current_protocol(CURRENT_CONFIG)
    for scene in ('room0','room2'):assert require_current_gt(CURRENT_PROTOCOL['canonical_GT'][scene]['file'])==scene
    result={'status':'PASS','evaluation_profile':CURRENT_PROTOCOL['evaluation_profile'],'profile_revision':2,
        'protocol_status':'CURRENT_DEVELOPMENT_NOT_FROZEN','scoring_commit':CURRENT_PROTOCOL['scoring_commit'],
        'audit_archive_commit':CURRENT_PROTOCOL['audit_archive_commit'],'entry_commit':commit,'default_main_CLI_used':True,
        'profile_config_sha256':CURRENT_PROTOCOL['config_sha256'],'current_total_evaluable_GT':350,'two_scene_evaluable_GT':124,
        'selected_physical_GT_count':45,'other_GT_count':79,'diagnosis_informed_development_test':True,'pre_registered_benchmark':False,
        'GT_used_for_masks_frames_IDs_or_thresholds':False,'fixed_geometry_GT_and_correspondence_verified':True,
        'all_frozen_controls_exactly_reproduced':True,'baseline_modified':False,'per_scene':scores,'pooled':pooled,
        'states':states,'pooled_states':pooled_states,'paired':pairs,'pooled_paired':pooled_pairs,
        'mapping':frozen['scenes'],'core_identity_posthoc':core_truth,'protected_sha256':protected,'seconds':time.time()-start}
    dump(OUT/'comparison.json',result)
    with (OUT/'comparison.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pergt=pooled_pairs['history_core_final_vs_strict_full_track']['per_gt']
    with (OUT/'per_gt_vs_current_repair.csv').open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(pergt[0]));writer.writeheader();writer.writerows(pergt)
    dump(OUT/'evaluation_complete.json',{'status':'PASS','profile_revision':2,'default_main_CLI_used':True,
        'comparison_sha256':sha256_file(OUT/'comparison.json'),'prediction_freeze_sha256':freeze_sha,
        'all_mapping_checks_passed':True,'all_inputs_and_original_predictions_unchanged':True,
        'output_sha256':{str(p):sha256_file(p) for p in OUT.rglob('*') if p.is_file()},'seconds':time.time()-start})
    emit('two_scene_R2_complete',status='PASS',pooled=pooled,seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.json').write_text(json.dumps({'status':'FAIL','traceback':traceback.format_exc()},indent=2)+'\n')
        raise
