"""Score all frozen original/strict repaired controls under one unchanged v3."""
from pathlib import Path
import csv,json,sys,subprocess,time
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
PRIOR=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
OLD=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009')
CODE=Path('/home/chenkejun/CVPR/worktrees/v3-object-observed-repair-20261009')
CONFIG=CODE/'unified_eval/configs/replica_ca_v3.object_observed_repair.development.json'
sys.path.insert(0,str(CODE))
from unified_eval.evaluate import evaluate_scenes
from unified_eval.io import load_gt,load_prediction,sha256_file
from unified_eval.metrics import build_overlap,owner_coverage
from unified_eval.repair_profile import adapt_surface,paired_revision_metrics
from unified_eval.schema import Protocol

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def emit(stage,**values):
    row={'stage':stage,'updated_unix':time.time(),**values};dump(OUT/'evaluation_progress.json',row);print(json.dumps(row),flush=True)
def verify(path,digest):assert sha256_file(path)==digest,'Frozen file changed: '+str(path)
def fields(summary):
    rows=summary['per_scene'];n=sum(r['diagnostics']['gt_instance_count'] for r in rows)
    return {'F1':summary['CA_PRF1_0_5']['F1'],'PQ':summary['CA_PQ']['PQ'],'AP':summary['CA_AP_uniform'],
        'AP50':summary['CA_AP50_uniform'],'mCov':summary['CA_mCov'],
        'macro_best_GT_recall':sum(r['diagnostics']['macro_best_gt_recall']*r['diagnostics']['gt_instance_count'] for r in rows)/n,
        **{k:summary['owner_surface'][k] for k in ('correct_owner_vertices','wrong_owner_vertices','unassigned_with_geometry_target_vertices',
              'no_geometry_target_vertices','target_vertices','Correct_owner_Coverage','Wrong_owner_Coverage','Unassigned_Coverage','No_geometry_Coverage')},
        **{k:summary['CA_PRF1_0_5'][k] for k in ('TP','FP','FN')},
        'merge_predictions':summary['structure'].get('merge_prediction_count'),'split_GT':summary['structure'].get('split_gt_count')}
def stats(gt,pred,protocol,native,nearest):
    overlap=build_overlap(gt,pred,protocol);score=owner_coverage(overlap)
    ref=np.full(gt.vertex_count,-1,np.int64);matched_gt=np.full(gt.vertex_count,-1,np.int64)
    alignment={str(r['instance_uid']):int(r['raw_gt_id']) for r in score['scoring_owner_alignment']}
    for instance in pred.instances:
        indices=instance.vertex_indices
        assert np.all(ref[indices]<0)
        identity=int(instance.metadata['native_instance_id']);ref[indices]=identity
        matched_gt[indices]=alignment.get(str(instance.instance_uid),-1)
    exists=nearest>=0;projected=np.full(gt.vertex_count,-1,np.int64);projected[exists]=native[nearest[exists]];projected[projected<=0]=-1
    np.testing.assert_array_equal(ref,projected)
    target=overlap.gt_object_mask
    quality=np.zeros(gt.vertex_count,np.uint8)
    quality[target&exists&(ref<=0)]=1;quality[target&exists&(ref>0)]=3
    quality[target&exists&(ref>0)&(matched_gt==gt.instance_id)]=2
    assert int(np.sum(target&(quality==2)))==score['correct_owner_vertices']
    assert int(np.sum(target&(quality==3)))==score['wrong_owner_vertices']
    return {'overlap':overlap,'quality':quality,'target':target,'ref_id':ref,'alignment':alignment}
def matrix(before,after,mask=None):
    if mask is not None:before,after=before[mask],after[mask]
    mat=np.bincount(before.astype(np.int64)*4+after,minlength=16).reshape(4,4)
    assert int(mat.sum())==len(before)
    return mat.astype(int).tolist()
def paired(gt,before,after,protocol,qbefore,qafter,targets):
    d=paired_revision_metrics(gt,before,after,protocol)
    old,new=qbefore['overlap'],qafter['overlap'];np.testing.assert_array_equal(old.gt_ids,new.gt_ids)
    a=old.recall.max(0);b=new.recall.max(0)
    for row,x,y in zip(d['per_gt'],a,b):row.update(before_best_recall=float(x),after_best_recall=float(y),delta_best_recall=float(y-x))
    d['fixed_selected_target_GT_ids']=sorted(targets)
    d['target_definition']='Frozen original human-seed target list intersect current qualified observed GT, for posthoc analysis only'
    grouped={}
    for group,choose in [('selected_targets',lambda r:r['raw_gt_id'] in targets),('other_targets',lambda r:r['raw_gt_id'] not in targets)]:
        rows=[r for r in d['per_gt'] if choose(r)]
        # The evaluator keys are recorded below after schema inspection, never guessed silently.
        grouped[group]={'objects':len(rows),'rows':rows}
    d['groups']=grouped
    mask=qbefore['target'];assert np.array_equal(mask,qafter['target'])
    d['quality_order']=['no_geometry','unassigned_with_geometry','correct_owner','wrong_owner']
    d['quality_transition_matrix']=matrix(qbefore['quality'],qafter['quality'],mask)
    d['owner_alignment_changes']={uid:[qbefore['alignment'].get(uid),qafter['alignment'].get(uid)]
        for uid in sorted(set(qbefore['alignment'])|set(qafter['alignment']))
        if qbefore['alignment'].get(uid)!=qafter['alignment'].get(uid)}
    d['quality_changed_on_same_identity_vertices']=int(np.sum(mask&(qbefore['quality']!=qafter['quality'])&(qbefore['ref_id']==qafter['ref_id'])))
    return d

def main():
    assert not (OUT/'evaluation_complete.json').exists()
    freeze_path=OUT/'predictions_freeze.json';freeze=json.loads(freeze_path.read_text());freeze_sha=sha256_file(freeze_path)
    assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT' and freeze['GT_used_for_mapping'] is False
    mapping=json.loads((OUT/'mapping_complete.json').read_text());assert mapping['status']=='PASS'
    prior=json.loads((PRIOR/'comparison.json').read_text())
    for p,d in freeze['new_code_sha256'].items():verify(p,d)
    for p,d in freeze['generated_output_sha256'].items():verify(p,d)
    for s,v in freeze['scenes'].items():
        verify(OUT/s/'complete.json',v['completion_sha256'])
        for rec in v['predictions'].values():verify(rec['path'],rec['sha256'])
    cfg=json.loads(CONFIG.read_text());protocol=Protocol.from_dict(cfg)
    assert cfg['frozen'] is False and protocol.evaluation_profile=='object_observed_repair'
    verify(CONFIG,prior['profile_config_sha256'])
    commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    assert commit==prior['evaluator_commit']=='05f6bb0d6c9a2491a51e990d217fd7f1397f5f6b'
    policy=json.loads((OUT/'policy.json').read_text())
    scores={};states={};pairs={};all_predictions={};all_diagnostics={};rows=[];protected={str(CONFIG):sha256_file(CONFIG)}
    for scene,record in freeze['scenes'].items():
        emit('load_fixed_GT_after_prediction_freeze',scene=scene)
        gt_path=OLD/'mesh'/scene/'gt.npz';cache=gt_path.parent/'fixed_gt_to_tsdf.npz'
        baseline_manifest=json.loads((gt_path.parent/'P1-A1_holes_geodesic/adapter_manifest.json').read_text())
        for p,d in ((gt_path,baseline_manifest['gt_file_sha256']),(cache,baseline_manifest['correspondence_file_sha256'])):
            verify(p,d);protected[str(p)]=d
        for name,d in baseline_manifest['evaluator_code_sha256'].items():p=CODE/'unified_eval'/name;verify(p,d);protected[str(p)]=d
        gt=load_gt(gt_path)
        with np.load(cache) as z:nearest=z['nearest_native_index'].copy()
        targets=json.loads((Path(policy['repair_source_roots'][scene])/'evaluation_summary.json').read_text())['target_GT_ids']
        targets=set(targets)&set(np.unique(gt.instance_id[gt.valid_vertex_mask&~gt.ignore_vertex_mask]).tolist())
        scores[scene]={};states[scene]={};predictions={};qualities={};pairs[scene]={};manifests=[]
        for name,rec in record['predictions'].items():
            semantic,condition,stage=name.split('_',1)[0],None,name.rsplit('_',1)[1]
            middle=name[len(semantic)+1:-(len(stage)+1)];condition=middle
            emit('adapt_and_score',scene=scene,prediction=name)
            target=OUT/'evaluation'/scene/name
            result=adapt_surface(rec['path'],gt_path,CONFIG,cache,target,
                method_name='P1-A1_'+name,method_commit=freeze['new_code_sha256'][str(Path(__file__).parent/'strict_vote_ops.py')],
                exported_instance_ids=np.asarray(rec['inventory'],np.int64),
                source_provenance={'fixed_raw_repair_batch':policy['repair_source_roots'][scene],
                    'vote_rule':semantic,'condition':condition,'prediction_freeze_sha256':freeze_sha,
                    'prediction_frozen_before_GT':True,'GT_used_for_mapping':False,'geometry_and_mask_edits_frozen':True})
            pred=load_prediction(target/'canonical_prediction.npz');diag=load_prediction(target/'diagnostic_support_prediction.npz')
            scores[scene][name]=fields(result['summary']);predictions[name]=pred;manifests.append(result['manifest'])
            all_predictions.setdefault(name,[]).append((gt,pred));all_diagnostics.setdefault(name,[]).append(diag)
            if condition=='baseline':
                oldname=('original' if semantic=='original' else 'abstain')+'_'+stage
                for key,value in prior['per_scene'][scene][oldname].items():
                    if key in scores[scene][name]:assert scores[scene][name][key]==value,(scene,name,key)
                ep=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native')/scene/'P1-A1/surface_p0/surface_evidence.npz' if semantic=='original' else PRIOR/'ablation'/scene/'abstain/surface_evidence.npz'
            else:ep=(Path(policy['repair_source_roots'][scene])/'validated'/condition/'native/surface_evidence.npz') if semantic=='original' else OUT/scene/condition/'native/surface_evidence.npz'
            with np.load(ep) as z:state=z['state'].copy()
            with np.load(rec['path']) as z:native=z['instance_id'].copy()
            qualities[name]=stats(gt,pred,protocol,native,nearest)
            if stage=='final':assert np.all(state[native<=0]!=2)
            category=state if stage=='native' else np.where(native>0,2,state).astype(np.uint8)
            states[scene][name]={'counts_U_T_assigned_C':np.bincount(category,minlength=4).astype(int).tolist(),
                'three_state_total':int(np.sum(native<=0)),'state_sha256':sha256_file(ep)}
            rows.append({'scope':scene,'semantic':semantic,'condition':condition,'stage':stage,**scores[scene][name],
                'U':states[scene][name]['counts_U_T_assigned_C'][0],'T':states[scene][name]['counts_U_T_assigned_C'][1],
                'C':states[scene][name]['counts_U_T_assigned_C'][3],'three_state_total':states[scene][name]['three_state_total']})
        for key in ('geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256'):
            assert len({m[key] for m in manifests})==1,'Paired input mismatch: '+key
        for stage in ('native','final'):
            for semantic in ('original','strict'):
                before=semantic+'_baseline_'+stage
                for condition in ('seed_only','full_track'):
                    after=semantic+'_'+condition+'_'+stage
                    d=paired(gt,predictions[before],predictions[after],protocol,qualities[before],qualities[after],targets)
                    pairs[scene][before+' -> '+after]=d
                    dump(OUT/'evaluation'/scene/after/'paired_vs_same_rule_baseline.json',d)
            for condition in ('baseline','seed_only','full_track'):
                before='original_'+condition+'_'+stage;after='strict_'+condition+'_'+stage
                d=paired(gt,predictions[before],predictions[after],protocol,qualities[before],qualities[after],targets)
                pairs[scene][before+' -> '+after]=d
                dump(OUT/'evaluation'/scene/after/'paired_vs_original_vote_rule.json',d)
    pooled={}
    for name,items in all_predictions.items():
        summary,_,_=evaluate_scenes(items,protocol,diagnostic_predictions=all_diagnostics[name])
        pooled[name]=fields(summary);dump(OUT/'evaluation/pooled'/name/'metrics.json',{'summary':summary})
        semantic=name.split('_',1)[0];stage=name.rsplit('_',1)[1];condition=name[len(semantic)+1:-(len(stage)+1)]
        vals=np.sum([states[s][name]['counts_U_T_assigned_C'] for s in states],axis=0)
        rows.append({'scope':'pooled_room0_room2','semantic':semantic,'condition':condition,'stage':stage,**pooled[name],
            'U':int(vals[0]),'T':int(vals[1]),'C':int(vals[3]),'three_state_total':int(vals[[0,1,3]].sum())})
    for p,d in protected.items():verify(p,d)
    for p,d in freeze['source_sha256'].items():verify(p,d)
    for p,d in freeze['generated_output_sha256'].items():verify(p,d)
    verify(freeze_path,freeze_sha)
    result={'status':'PASS','protocol':'Replica-CA-v3','evaluation_profile':'object_observed_repair',
        'protocol_status':'DEVELOPMENT_NOT_OFFICIAL_NOT_FROZEN','evaluator_commit':commit,
        'profile_config_sha256':sha256_file(CONFIG),'predictions_freeze_sha256':freeze_sha,
        'all_baseline_metric_controls_exact':True,'geometry_scope_correspondence_identical':True,
        'baseline_modified':False,'per_scene':scores,'pooled':pooled,'states':states,
        'paired':{s:{k:{a:b for a,b in v.items() if a not in ('per_gt','groups')} for k,v in ps.items()} for s,ps in pairs.items()},
        'GT_and_evaluator_sha256':protected}
    dump(OUT/'comparison.json',result)
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(OUT/'evaluation_complete.json',{'status':'PASS','predictions_evaluated':24,'baseline_metric_controls_exact':True,
        'all_inputs_and_predictions_hashes_unchanged':True,'comparison_sha256':sha256_file(OUT/'comparison.json')})
    emit('evaluation_complete',status='PASS',pooled=pooled)

if __name__=='__main__':main()
