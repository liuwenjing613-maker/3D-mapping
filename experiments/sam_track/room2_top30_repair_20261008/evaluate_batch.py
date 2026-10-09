"""Post-hoc-only evaluation, keeping the exact existing unified v3 code/flags."""
from pathlib import Path
import json,sys,time,traceback
import numpy as np
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'))
import evaluate_pilot_v3 as e
import fixed_surface_repair as f
import run_repairs as r

def main():
    repair_complete=json.loads((ROOT/'repair_complete.json').read_text())
    assert repair_complete['status']=='PASS'
    repair_root=Path(repair_complete['outputs_root'])
    oldfreeze=json.loads((e.ROOT/'v3/evaluation_freeze.json').read_text())
    assert oldfreeze['flags']==e.FLAGS and oldfreeze['code_sha256']==r.sha(e.__file__)
    surfaces={'baseline':e.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz',
              'seed_only':repair_root/'seed_only/final/instance_surface.npz','full_track':repair_root/'full_track/final/instance_surface.npz'}
    for condition in ['seed_only','full_track']:
        done=json.loads((repair_root/condition/'complete.json').read_text())
        assert done['status']=='PASS' and done['predictions_frozen_before_any_GT_evaluation'] and not done['GT_used_for_repair']
        for path,digest in done['output_sha256'].items():assert r.sha(path)==digest
    freeze={'status':'PREDICTIONS_FROZEN_BEFORE_GT','predictions':{name:{'path':str(p),'sha256':r.sha(p)} for name,p in surfaces.items()},
            'repair_complete_sha256':r.sha(ROOT/'repair_complete.json'),'wrapper_sha256':r.sha(__file__),
            'evaluator_sha256':r.sha(e.__file__),'flags':e.FLAGS,'protocol_sha256':r.sha(e.PROTOCOL),
            'effective_protocol':oldfreeze['effective_protocol'],'GT_used_for_repair':False,
            'target_definition':'fixed human-seed native support; nearest valid GT within 1cm; >=10 hits and >=5% per seed, unchanged previous rule'}
    path=ROOT/'evaluation_freeze.json'
    if path.exists():assert json.loads(path.read_text())==freeze
    else:f.atomic_json(path,freeze)
    parser=e.ArgumentParser();e.add_protocol_args(parser)
    protocol,raw,debug=e.load_protocol(e.PROTOCOL,parser.parse_args(['--config',str(e.PROTOCOL),*e.FLAGS]))
    assert raw==oldfreeze['effective_protocol']
    scores={'baseline':json.loads((e.ROOT/'v3/baseline/room2/complete.json').read_text())}
    assert scores['baseline']['source_map_sha256']==r.sha(surfaces['baseline'])
    for condition in ['seed_only','full_track']:
        scores[condition]=e.evaluate('room2',surfaces[condition],repair_root/condition/'v3',protocol)
        for key in ['flags','protocol_sha256','GT_sha256','evaluation_code_sha256']:
            assert scores[condition][key]==scores['baseline'][key]
    # All prediction and repair sources were frozen before this first GT read.
    gt=e.load_gt(e.INPUT/'room2/ground_truth/gt.npz');baseline=f.load_arrays(surfaces['baseline'])
    overlap_paths={'baseline':e.ROOT/'v3/baseline/room2/overlap.npz',**{name:repair_root/name/'v3/overlap.npz' for name in ['seed_only','full_track']}}
    overlaps={name:e.load_overlap(p) for name,p in overlap_paths.items()}
    per={name:e.per_gt(z) for name,z in overlaps.items()};allowed=overlaps['baseline']['gt_ids']
    association=json.loads((ROOT/'global_association.json').read_text());objects={row['canonical_track_id']:row for row in association['objects']}
    aliases={int(a):int(b) for a,b in association['aliases'].items()}
    seedrows=json.loads((ROOT/'seed_manifest.json').read_text())['seeds'];targets=set();source=[];canonical_counts={}
    tree=cKDTree(gt.xyz_ref)
    for row in seedrows:
        if row['status']!='READY':continue
        z=f.load_arrays(ROOT/'cases'/row['case_uid']/'seed_projected_support.npz')
        pts,tracks=z['surface_point_index'],z['track_id']
        distance,nearest=tree.query(baseline['xyz_m'][pts],workers=8)
        valid=(distance<.01)&gt.valid_vertex_mask[nearest]&~gt.ignore_vertex_mask[nearest]
        case_targets=set();object_sources=[]
        for obj in row['objects']:
            oid=obj['track_id'];cid=aliases.get(oid,oid)
            keep=valid&(tracks==oid)&np.isin(gt.instance_id[nearest],allowed)
            gids,n=np.unique(gt.instance_id[nearest[keep]],return_counts=True);total=max(1,int(keep.sum()))
            significant=(n>=10)&(n/total>=.05);case_targets.update(int(g) for g in gids[significant])
            counts=canonical_counts.setdefault(cid,{})
            for gid,count in zip(gids,n):counts[int(gid)]=counts.get(int(gid),0)+int(count)
            object_sources.append({'track_id':oid,'canonical_track_id':cid,'native_mask_id':obj['native_mask_id'],'nearest_valid_native_points':int(keep.sum()),
                'GT_overlap':[{'GT_id':int(g),'native_points':int(count),'fraction':float(count/total),'included':bool(s)} for g,count,s in zip(gids,n,significant)]})
        targets.update(case_targets);source.append({'case_uid':row['case_uid'],'ROI':row['source_choice']['ROI'],
                         'targets':sorted(case_targets),'objects':object_sources})
    mappingpath=ROOT/'fixed_surface_mapping.npz'
    beforehash=r.sha(mappingpath) if mappingpath.exists() else None
    mapping=f.fixed_mapping(mappingpath,baseline['xyz_m'],gt.xyz_ref,.01)
    assert beforehash is None or beforehash==mapping['cache_sha256']
    maps={name:f.load_arrays(path) for name,path in surfaces.items()}
    for value in maps.values():f.require_same_geometry(baseline,value)
    fixed=[]
    valid=gt.valid_vertex_mask&~gt.ignore_vertex_mask
    for cid,obj in objects.items():
        counts=canonical_counts.get(cid,{})
        if not counts:fixed.append({'canonical_track_id':cid,'persistent_id':obj['persistent_id'],'status':'NO_EVALUABLE_GT_SEED_SUPPORT'});continue
        gid=max(counts,key=lambda g:(counts[g],-g));pid=obj['persistent_id']
        fixed.append({'canonical_track_id':cid,'member_track_ids':obj['member_track_ids'],'persistent_id':pid,
                      'GT_id':gid,'seed_GT_fraction':counts[gid]/sum(counts.values()),'seed_GT_counts':counts,'status':'POSTHOC_ONLY',
                      'conditions':{name:f.target_diagnostics(mapping,value['instance_id'],gt.instance_id,valid,{pid:gid})['targets'][0] for name,value in maps.items()}})
    comparisons={}
    b=scores['baseline']['metrics'];before=per['baseline']
    for name in ['seed_only','full_track']:
        after=per[name];m=scores[name]['metrics']
        lost=[g for g in before if g not in targets and before[g]['matched_IoU_gt_0_5'] and not after[g]['matched_IoU_gt_0_5']]
        degraded=[g for g in before if g not in targets and after[g]['best_IoU']<before[g]['best_IoU']-.01]
        delta=float(np.mean([after[g]['best_IoU']-before[g]['best_IoU'] for g in targets])) if targets else None
        structural=sum(int(after[g]['split'])+after[g]['merged_predictions_touching_GT']+after[g]['duplicate_predictions']-
                       int(before[g]['split'])-before[g]['merged_predictions_touching_GT']-before[g]['duplicate_predictions'] for g in targets)
        f1delta=m['CA_PRF1_0_5']['F1']-b['CA_PRF1_0_5']['F1'];pqdelta=m['CA_PQ']['PQ']-b['CA_PQ']['PQ']
        improvement=delta is not None and (delta>=.01 or structural<0) and delta>=-1e-8
        regression=bool(lost) or (delta is not None and delta<-.01) or structural>0
        success=improvement and not regression and not degraded and f1delta>=-1e-8 and pqdelta>=-1e-8
        comparisons[name]={'scene_metrics':m,'F1_delta':f1delta,'PQ_delta':pqdelta,'target_mean_IoU_delta':delta,
             'target_structure_count_delta':structural,'strict_success':bool(success),
             'outcome':'improved' if success else 'mixed' if improvement and regression else 'regressed' if regression else 'small_or_unchanged',
             'non_target_previously_correct_lost':lost,'non_target_IoU_degraded_over_0_01':degraded,
             'targets':[{'before':before[g],'after':after[g]} for g in sorted(targets)]}
    gid_groups={}
    for row in fixed:
        if row.get('GT_id') is not None:gid_groups.setdefault(row['GT_id'],[]).append(row['canonical_track_id'])
    result={'status':'PASS','scene':'room2','formal_benchmark_result':False,'protocol_frozen':bool(raw.get('frozen',False)),
            'protocol_metric_status':b['status'],'predictions_frozen_before_GT':True,'GT_used_for_repair':False,
            'v3_code_flags_protocol_unchanged':True,'prediction_freeze_sha256':r.sha(ROOT/'evaluation_freeze.json'),
            'baseline_v3_metrics':b,'comparisons':comparisons,'target_GT_ids':sorted(targets),'target_sources':source,
            'fixed_surface_diagnostics':fixed,'fixed_mapping_sha256':mapping['cache_sha256'],
            'multiple_selected_parts_of_same_GT_instance':{str(g):ids for g,ids in gid_groups.items() if len(ids)>1},
            'cases':[{'case_uid':row['case_uid'],'ROI':row['source_choice']['ROI'],'status':row['status'],
                'target_GT_ids':next((s['targets'] for s in source if s['case_uid']==row['case_uid']),[])} for row in seedrows]}
    for row in freeze['predictions'].values():assert r.sha(row['path'])==row['sha256']
    f.atomic_json(ROOT/'evaluation_summary.json',result)
    print(json.dumps({'status':'PASS','baseline':b,'comparisons':{k:{x:v[x] for x in ['scene_metrics','target_mean_IoU_delta','strict_success','non_target_previously_correct_lost','non_target_IoU_degraded_over_0_01']} for k,v in comparisons.items()},
                      'targets':sorted(targets),'multiple_selected_parts_of_same_GT_instance':result['multiple_selected_parts_of_same_GT_instance']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:f.atomic_json(ROOT/'evaluation_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
