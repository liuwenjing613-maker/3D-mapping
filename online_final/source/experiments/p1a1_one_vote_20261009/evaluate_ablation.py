"""Evaluate frozen ablation predictions with one unchanged current V3 profile."""
from pathlib import Path
import csv
import json
import subprocess
import sys
import time
import numpy as np

CODE=Path('/home/chenkejun/CVPR/worktrees/v3-object-observed-repair-20261009')
OUT=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
OLD=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009')
CONFIG=CODE/'unified_eval/configs/replica_ca_v3.object_observed_repair.development.json'
sys.path.insert(0,str(CODE))
from unified_eval.evaluate import evaluate_scenes
from unified_eval.io import load_gt,load_prediction,sha256_file
from unified_eval.repair_profile import adapt_surface,paired_revision_metrics
from unified_eval.geometry import load_fixed_surface_correspondence
from unified_eval.metrics import build_overlap
from unified_eval.schema import Protocol


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def emit(stage,**values):
    row={'stage':stage,'updated_unix':time.time(),**values}
    dump(OUT/'evaluation_progress.json',row);print(json.dumps(row),flush=True)


def verify(path,digest):
    assert sha256_file(path)==digest,'Checksum changed: '+str(path)


def fields(summary):
    rows=summary['per_scene']
    denominator=sum(r['diagnostics']['gt_instance_count'] for r in rows)
    recall=sum(r['diagnostics']['macro_best_gt_recall']*r['diagnostics']['gt_instance_count'] for r in rows)/denominator
    return {'AP':summary['CA_AP_uniform'],'AP50':summary['CA_AP50_uniform'],
        'PQ':summary['CA_PQ']['PQ'],'F1':summary['CA_PRF1_0_5']['F1'],'mCov':summary['CA_mCov'],
        'macro_best_GT_recall':recall,**{k:summary['owner_surface'][k] for k in (
            'Correct_owner_Coverage','Wrong_owner_Coverage','Unassigned_Coverage','No_geometry_Coverage',
            'correct_owner_vertices','wrong_owner_vertices','unassigned_with_geometry_target_vertices')},
        **{k:summary['CA_PRF1_0_5'][k] for k in ('TP','FP','FN')},
        'merge_predictions':summary['structure'].get('merge_prediction_count'),
        'split_GT':summary['structure'].get('split_gt_count'),'status':summary['status']}


def gt_distribution(scene,gt,cache,scope_path,xyz):
    with np.load(OUT/'ablation'/scene/'collision_point_counts.npz') as z:
        bad=z['colliding_frame_count'];resolved=z['depth_resolved_frame_count']
    correspondence=load_fixed_surface_correspondence(cache,xyz,gt.xyz_ref)
    matched=correspondence.nearest_native_index>=0
    collision=np.zeros(gt.vertex_count,bool);resolved_reference=np.zeros(gt.vertex_count,bool)
    collision[matched]=bad[correspondence.nearest_native_index[matched]]>0
    resolved_reference[matched]=resolved[correspondence.nearest_native_index[matched]]>0
    scope=json.loads(scope_path.read_text())
    rows=[]
    for obj in scope['objects']:
        vertices=gt.raw_instance_id==obj['raw_gt_id']
        observed=(gt.instance_id==obj['raw_gt_id'])&gt.valid_vertex_mask&~gt.ignore_vertex_mask
        rows.append({**obj,'reference_vertices_nearest_to_colliding_surface':int(np.count_nonzero(vertices&collision)),
            'observed_target_vertices_nearest_to_colliding_surface':int(np.count_nonzero(observed&collision)),
            'observed_target_collision_fraction':float(np.count_nonzero(observed&collision)/np.count_nonzero(observed)) if np.any(observed) else None,
            'reference_vertices_nearest_to_depth_resolved_surface':int(np.count_nonzero(vertices&resolved_reference))})
    rows.sort(key=lambda r:-r['reference_vertices_nearest_to_colliding_surface'])
    result={'GT_used_only_after_prediction_freeze':True,
        'definition':'Fixed 1cm GT-reference-to-full-native correspondence; count reference vertices whose nearest native point had any collision',
        'thin_object_status':'No ad-hoc semantic thin-object list or GT-based policy selection; inspect raw mask/RGB overlays and per-object distributions',
        'per_object':rows}
    dump(OUT/'ablation'/scene/'collision_gt_distribution.json',result)


def main():
    assert not (OUT/'evaluation_complete.json').exists(),'Evaluation already complete'
    freeze_path=OUT/'predictions_freeze.json'
    freeze=json.loads(freeze_path.read_text());assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    assert freeze['GT_read_for_mapping'] is False and freeze['baseline_modified'] is False
    freeze_sha=sha256_file(freeze_path)
    config=json.loads(CONFIG.read_text())
    protocol=Protocol.from_dict(config)
    assert protocol.evaluation_profile=='object_observed_repair' and config['frozen'] is False
    code_commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
    assert code_commit=='05f6bb0d6c9a2491a51e990d217fd7f1397f5f6b'
    for path,digest in freeze['code_sha256'].items():verify(path,digest)
    for scene,record in freeze['scenes'].items():
        verify(OUT/'ablation'/scene/'mapping_complete.json',record['mapping_complete_sha256'])
        for pred in record['predictions'].values():verify(pred['path'],pred['sha256'])
    predictions={f'{policy}_{stage}':[] for stage in ('native','final') for policy in ('original','abstain','depth')}
    diagnostics={name:[] for name in predictions}
    results={};pairs={};rows=[];integrity={}
    for scene in ('room0','room2'):
        emit('verify_GT_cache_and_profile',scene=scene)
        gt_path=OLD/'mesh'/scene/'gt.npz';gt=load_gt(gt_path)
        fixed_cache=OLD/'mesh'/scene/'fixed_gt_to_tsdf.npz'
        results[scene]={};pairs[scene]={};scene_manifests=[]
        for stage in ('native','final'):
            cached_name='P1-A1_native' if stage=='native' else 'P1-A1_holes_geodesic'
            cached_dir=OLD/'mesh'/scene/cached_name
            cached_manifest=json.loads((cached_dir/'adapter_manifest.json').read_text())
            for path,digest in ((gt_path,cached_manifest['gt_file_sha256']),
                (fixed_cache,cached_manifest['correspondence_file_sha256']),
                (CONFIG,cached_manifest['profile_config_sha256'])):verify(path,digest)
            for file,digest in cached_manifest['evaluator_code_sha256'].items():verify(CODE/'unified_eval'/file,digest)
            stage_preds={}
            for policy in ('original','abstain','depth'):
                name=f'{policy}_{stage}';record=freeze['scenes'][scene]['predictions'][name]
                emit('adapt_and_score_frozen_prediction',scene=scene,stage_name=stage,policy=policy)
                target=OUT/'evaluation'/scene/name
                provenance={**cached_manifest['source_provenance'],'vote_policy':policy,
                    'original_geometry_and_associations_unchanged':True,'prediction_freeze_sha256':freeze_sha,
                    'prediction_frozen_before_GT':True,'GT_used_for_mapping':False}
                result=adapt_surface(record['path'],gt_path,CONFIG,fixed_cache,target,
                    method_name='P1-A1_'+name,method_commit=freeze['code_sha256'][str(Path(__file__).parent/'one_vote.py')],
                    exported_instance_ids=np.asarray(record['inventory'],np.int64),source_provenance=provenance)
                pred=load_prediction(target/'canonical_prediction.npz');diag=load_prediction(target/'diagnostic_support_prediction.npz')
                if policy=='original':
                    expected=json.loads((cached_dir/'metrics.json').read_text())['summary']
                    for key in ('CA_AP_uniform','CA_AP50_uniform','CA_PQ','CA_PRF1_0_5','CA_mCov','owner_surface'):
                        assert result['summary'][key]==expected[key],f'Original evaluation control differs: {scene}, {stage}, {key}'
                stage_preds[policy]=pred
                predictions[name].append((gt,pred));diagnostics[name].append(diag)
                values=fields(result['summary']);results[scene][name]=values
                rows.append({'scope':scene,'stage':stage,'policy':policy,**values})
                scene_manifests.append(result['manifest'])
            for policy in ('abstain','depth'):
                comparison=paired_revision_metrics(gt,stage_preds['original'],stage_preds[policy],protocol)
                first=build_overlap(gt,stage_preds['original'],protocol)
                second=build_overlap(gt,stage_preds[policy],protocol)
                old_recall=first.recall.max(axis=0) if len(first.pred_uids) else np.zeros(len(first.gt_ids))
                new_recall=second.recall.max(axis=0) if len(second.pred_uids) else np.zeros(len(second.gt_ids))
                for row,a,b in zip(comparison['per_gt'],old_recall,new_recall):
                    row.update(before_best_recall=float(a),after_best_recall=float(b),delta_best_recall=float(b-a))
                dump(OUT/'evaluation'/scene/f'{policy}_{stage}'/'paired_vs_original.json',comparison)
                pairs[scene][f'{policy}_{stage}']={k:v for k,v in comparison.items() if k!='per_gt'}
        for key in ('geometry_xyz_sha256','correspondence_sha256','correspondence_file_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256'):
            assert len({m[key] for m in scene_manifests})==1,'Paired comparison differs: '+key
        integrity[scene]={key:scene_manifests[0][key] for key in ('geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256')}
        with np.load(freeze['scenes'][scene]['predictions']['original_native']['path']) as z:xyz=z['xyz_m']
        gt_distribution(scene,gt,fixed_cache,OLD/'mesh'/scene/'gt_scope.json',xyz)
    pooled={}
    for name,items in predictions.items():
        summary,_,_=evaluate_scenes(items,protocol,diagnostic_predictions=diagnostics[name])
        stage=name.split('_')[-1];policy=name.removesuffix('_'+stage)
        values=fields(summary);pooled[name]=values
        dump(OUT/'evaluation'/'pooled'/name/'metrics.json',{'summary':summary})
        rows.append({'scope':'pooled_room0_room2','stage':stage,'policy':policy,**values})
    for path,digest in freeze['code_sha256'].items():verify(path,digest)
    for scene,record in freeze['scenes'].items():
        mapping=json.loads((OUT/'ablation'/scene/'mapping_complete.json').read_text())
        for path,digest in mapping['source_sha256'].items():verify(path,digest)
        for pred in record['predictions'].values():verify(pred['path'],pred['sha256'])
    verify(freeze_path,freeze_sha)
    result={'status':'PASS','protocol':'Replica-CA-v3','evaluation_profile':'object_observed_repair',
        'protocol_status':'DEVELOPMENT_NOT_OFFICIAL_NOT_FROZEN','evaluator_commit':code_commit,
        'profile_config_sha256':sha256_file(CONFIG),'predictions_freeze_sha256':freeze_sha,
        'original_metric_controls_exact':True,'geometry_scope_correspondence_identical':True,
        'baseline_modified':False,'per_scene':results,'pooled':pooled,'paired_vs_original':pairs,'integrity':integrity}
    dump(OUT/'comparison.json',result)
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(OUT/'evaluation_complete.json',{'status':'PASS','comparison_sha256':sha256_file(OUT/'comparison.json'),
        'predictions_freeze_sha256':freeze_sha,'original_controls_exact':True,'baseline_modified':False,
        'paired_inputs_and_sources_hash_verified_after_scoring':True})
    emit('evaluation_complete',pooled=pooled)


if __name__=='__main__':main()
