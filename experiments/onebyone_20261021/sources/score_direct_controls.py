"""Score saved whole/partial direct stages for phase-matched comparison."""
from bootstrap import C,R,SOURCE,PARTIAL,COUNT,SCENE,GROUP,CFG,MAIN,verify_sources
from pathlib import Path
import json,subprocess,sys,time,traceback
import numpy as np
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,require_current_protocol
from unified_eval.io import sha256_file
read=lambda p:json.loads(Path(p).read_text());sha=sha256_file
OUT=R/'direct_controls_r2'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    verify_sources();assert require_current_protocol(CURRENT_CONFIG)
    assert read(R/'evaluation_complete.json')['status']=='PASS'
    folders={'whole_direct':SOURCE/'original_repair'/SCENE/'full_track','partial_direct':Path(CFG['partial_transactions']).parent/'direct'/SCENE}
    surfaces={f'{v}_{s}':folder/s/'instance_surface.npz' for v,folder in folders.items() for s in ['native','final']}
    freeze=dict(status='EXISTING_GT_FREE_PREDICTIONS_LOCKED_BEFORE_RESCORING',source_sha256={str(p):sha(p) for p in surfaces.values()},
        frozen_original_pipeline=sha(SOURCE/'original_repair/predictions_freeze.json'),frozen_partial_pipeline=sha(PARTIAL/'predictions_freeze.json'))
    dump(OUT/'prediction_lock.json',freeze)
    gtpath=Path(CURRENT_PROTOCOL['canonical_GT'][SCENE]['file'])
    cache_root=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache') if SCENE in ['room0','room2'] else Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/geometry_cache')
    cache=cache_root/SCENE/'fixed_gt_to_tsdf_r2.npz'
    assert sha(cache)==read(R/'evaluation_r2/protocol_freeze.json')['fixed_correspondence_sha256']
    rows=[];locks=['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256','profile_semantics_sha256','evaluator_code_sha256']
    for stage in ['native','final']:
        ref=read(R/'evaluation_r2'/f'direct_{stage}/adapter_manifest.json')
        for variant,folder in folders.items():
            surface=surfaces[variant+'_'+stage];dest=OUT/f'{variant}_{stage}'
            provenance=read(R/'evaluation_r2/provenance'/f'direct_{stage}.json')
            provenance.update(condition=variant,surface_sha256=sha(surface),prediction_freeze_sha256=sha(OUT/'prediction_lock.json'),batch=str(OUT))
            pp=OUT/'provenance'/f'{variant}_{stage}.json';dump(pp,provenance)
            if not (dest/'metrics.json').exists():
                with (OUT/f'{variant}_{stage}.log').open('w') as log:
                    p=subprocess.run([sys.executable,'-B','-m','unified_eval.cli','adapt-surface','--gt',str(gtpath),'--surface',str(surface),
                        '--fixed-correspondence',str(cache),'--source-provenance',str(pp),'--method-name',f'point_control_{GROUP}_{SCENE}_{variant}_{stage}',
                        '--method-commit',sha(C/'score_direct_controls.py'),'--out',str(dest)],cwd=MAIN,stdout=log,stderr=subprocess.STDOUT)
                assert p.returncode==0,(variant,stage)
            m=read(dest/'adapter_manifest.json')
            for k in locks:assert m[k]==ref[k],k
            sm=read(dest/'metrics.json')['summary'];labels=np.load(surface)['instance_id']
            e=np.load(folder/'native/surface_evidence.npz')
            rows.append(dict(variant=variant,stage=stage,F1=sm['CA_PRF1_0_5']['F1'],PQ=sm['CA_PQ']['PQ'],AP=sm['CA_AP_uniform'],AP50=sm['CA_AP50_uniform'],mCov=sm['CA_mCov'],
                TP=sm['CA_PRF1_0_5']['TP'],FP=sm['CA_PRF1_0_5']['FP'],FN=sm['CA_PRF1_0_5']['FN'],gray_total=int((labels<=0).sum()),owner_surface=sm['owner_surface']))
    for p,d in freeze['source_sha256'].items():assert sha(p)==d,p
    dump(OUT/'comparison.json',dict(status='PASS',group=GROUP,scene=SCENE,rows=rows,same_phase_same_geometry_scope_correspondence_and_scorer=True))
    print('DIRECT_CONTROLS_COMPLETE',GROUP,SCENE,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:dump(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
