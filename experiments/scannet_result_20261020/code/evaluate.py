"""Use main's locked R2 scoring functions with the archived ScanNet GT adapter."""
from pathlib import Path
import json,hashlib,sys,csv
import numpy as np
C=Path(__file__).resolve().parent
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
OLD=Path('/data/chenkejun/CVPR/evaluation_results/scannet_v3_r2_reeval_20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources
from unified_eval.current_protocol import CURRENT_CONFIG,require_current_protocol
from unified_eval.repair_profile import adapt_surface,paired_revision_metrics
from unified_eval.io import load_gt,load_prediction,sha256_file,repair_evaluator_code_hashes
from unified_eval.schema import Protocol
read=lambda p:json.loads(Path(p).read_text())
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def evaluate_scene(scene,include_partial=False,baseline_only=False):
    verify_sources();require_current_protocol(CURRENT_CONFIG)
    cfg=read(OLD/'effective_protocol.json');current=read(CURRENT_CONFIG)
    exceptions={'dataset','dataset_adaptation','scenes','review_required_gt_ids'}
    for key in set(current)|set(cfg):
        if key not in exceptions:assert cfg.get(key)==current.get(key),key
    config=C/'scannet_r2.json'
    if config.exists():assert read(config)==cfg
    else:save(config,cfg)
    protocol=Protocol.from_dict(cfg)
    gtpath=OLD/scene/'ground_truth/gt.npz';gt=load_gt(gtpath);scope=read(OLD/'fixed_scope_lock.json')['scope'][scene]
    assert gt.metadata['gt_scope_sha256']==scope['gt_scope_sha256']
    assert gt.metadata['evaluator_code_sha256']==repair_evaluator_code_hashes()
    frames=read(C/'configs'/(scene+'.json'))['frame_ids']
    assert frames==gt.metadata['observation_metadata']['frame_ids']
    fsha=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest()
    assert fsha==gt.metadata['observation_metadata']['frame_list_sha256']
    expected_count={'scene0011_00':27,'scene0608_00':34}[scene]
    baseline=read(R/'baseline'/(scene+'_predictions_freeze.json'))
    assert baseline['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    outputs={'baseline':{s:Path(baseline['scenes'][scene]['predictions']['abstain_'+s]['path']) for s in ['native','final']}}
    freezes={'baseline':R/'baseline'/(scene+'_predictions_freeze.json')}
    if not baseline_only:
        source=R/scene/'e1_repair/e1/E1_expanded_history'
        freezes['E1']=source/'predictions_freeze.json'
        freeze=read(freezes['E1']);assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
        for group in ['output_sha256','code_sha256']:
            for p,h in freeze[group].items():assert sha256_file(p)==h,p
        outputs['E1']={s:source/scene/s/'instance_surface.npz' for s in ['native','final']}
    if include_partial:
        source=R/scene/'partial'
        freezes['swap_partial']=source/'predictions_freeze.json'
        freeze=read(freezes['swap_partial']);assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
        for group in ['output_sha256','code_sha256']:
            for p,h in freeze[group].items():assert sha256_file(p)==h,p
        outputs['swap_partial']={s:source/'whole_to_partial'/scene/s/'instance_surface.npz' for s in ['native','final']}
    out=R/scene/'evaluation_r2';out.mkdir(exist_ok=True)
    cache=out/'fixed_correspondence.npz';rows=[];manifests=[]
    old_inputs={str(gtpath):sha256_file(gtpath),str(OLD/'effective_protocol.json'):sha256_file(OLD/'effective_protocol.json')}
    for name,stages in outputs.items():
        for stage,surface in stages.items():
            dest=out/(name+'_'+stage)
            with np.load(surface) as z,np.load(outputs['baseline'][stage]) as b:
                np.testing.assert_array_equal(z['xyz_m'],b['xyz_m'])
                np.testing.assert_array_equal(z['rgb'],b['rgb'])
                gray=int((z['instance_id']<=0).sum())
            provenance=dict(input_scope_verified=True,frame_list_sha256=fsha,frame_count=len(frames),
                input_config_sha256=sha256_file(C/'configs'/(scene+'.json')),prediction_freeze_sha256=sha256_file(freezes[name]),
                prediction_frozen_before_GT=True,GT_used_for_mapping=False,
                adaptation='main cd260569 locked scoring functions; archived ScanNet R2 27/34 GT; explicit approved dataset adapter')
            if (dest/'adapter_manifest.json').exists():
                manifest=read(dest/'adapter_manifest.json')
                pred=load_prediction(dest/'canonical_prediction.npz')
                assert pred.metadata['source_instance_surface_sha256']==sha256_file(surface)
                summary=read(dest/'metrics.json')['summary']
            else:
                result=adapt_surface(surface,gtpath,config,cache,dest,method_name=name+'_'+stage,
                    method_commit='source_sha256:'+sha256_file(surface),source_provenance=provenance)
                manifest=result['manifest'];summary=result['summary']
            manifests.append(manifest);pr=summary['CA_PRF1_0_5']
            assert pr['TP']+pr['FN']==expected_count
            row=dict(scene=scene,method=name,stage=stage,GT=expected_count,AP=summary['CA_AP_uniform'],
                AP50=summary['CA_AP50_uniform'],F1=pr['F1'],PQ=summary['CA_PQ']['PQ'],mCov=summary['CA_mCov'],
                TP=pr['TP'],FP=pr['FP'],FN=pr['FN'],gray_total=gray,owner_surface=summary['owner_surface'])
            rows.append(row);print('SCORED',json.dumps(row),flush=True)
    locks=['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256']
    for manifest in manifests[1:]:
        for key in locks:assert manifest[key]==manifests[0][key],key
    for method in outputs:
        if method=='baseline':continue
        for stage in ['native','final']:
            before=load_prediction(out/('baseline_'+stage)/'canonical_prediction.npz')
            after=load_prediction(out/(method+'_'+stage)/'canonical_prediction.npz')
            pair=paired_revision_metrics(gt,before,after,protocol)
            save(out/'paired'/(method+'_'+stage+'.json'),pair)
    history=[r for r in read(OLD/'comparison.json')['rows'] if r['scene']==scene]
    save(out/'comparison.json',dict(status='PASS',scene=scene,GT=expected_count,rows=rows,historical_rows=history,
        current_source='main',scoring_commit='cd260569e89de1e9525f82c45ee10842b80616b0',
        profile='object_observed_repair / revision 2',frozen=False,
        entry='approved ScanNet wrapper calls main locked scoring functions; Replica default GT whitelist unchanged',
        paired_geometry_scope_correspondence_equal=True,fixed_input_sha256=old_inputs,
        limitations=['Historical P1-A1 outputs use different voting rules; only new baseline/E1/partial form the fixed-surface repair experiment.',
                     'Tracking uses only the existing stride10 RGB sequence, with original frame IDs.']))
    save(out/'protocol_freeze.json',dict(fixed_correspondence_sha256=sha256_file(cache),evaluator_code_sha256=repair_evaluator_code_hashes(),
        main_config_sha256=sha256_file(CURRENT_CONFIG),dataset_config_sha256=sha256_file(config),gt_sha256=sha256_file(gtpath)))
    for p,h in old_inputs.items():assert sha256_file(p)==h
    verify_sources()
if __name__=='__main__':evaluate_scene(sys.argv[1],baseline_only=True)
