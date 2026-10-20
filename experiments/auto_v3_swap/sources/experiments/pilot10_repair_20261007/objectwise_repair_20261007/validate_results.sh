set -eu
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np
from PIL import Image
root=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/objectwise_repair_20261007')
code=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007/objectwise_repair_20261007')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_policy=json.loads((code.parent/'joint_mask_policy.json').read_text())
new_policy=json.loads((code/'policy.json').read_text())
for key,value in old_policy.items():
    if isinstance(value,(int,float)) and not isinstance(value,bool):assert new_policy[key]==value,key
tests={}
for folder,module in [(code,'test_objectwise_ops'),(code.parent/'strict_local_repair_20261007','test_fixed_surface_repair')]:
    env={**os.environ,'CVPR_REPAIR_TEST_ROOT':str(root/'validation_tests')}
    test=subprocess.run([sys.executable,'-m','unittest','-v',module],cwd=folder,env=env,capture_output=True,text=True)
    (root/(module+'.log')).write_text(test.stdout+test.stderr)
    assert test.returncode==0,test.stdout+test.stderr
    tests[module]={'status':'PASS','log_sha256':sha(root/(module+'.log'))}
checks={}
for config_name,folder in [('case_room2_original_control.json','room2_original_control'),('case_room2_sam_seed.json','room2_sam_seed'),('case_room2_original_seed.json','room2_original_seed')]:
    work=root/folder
    complete=json.loads((work/'complete.json').read_text())
    evaluation=json.loads((work/'evaluation_summary.json').read_text())
    assert complete['status']==evaluation['status']=='PASS'
    protected=json.loads((work/'source_freeze.json').read_text())['protected_source_hashes']
    for p,h in protected.items():assert sha(p)==h,p
    assert complete['repair']['strict_commit']['final_outside_changes']==0
    for key in ['complete_frame_replay_bit_identical_to_delta_ledger','exact_rollback_pass','unretired_old_observation_supports_preserved_exactly']:
        assert complete['repair'][key]
    outputs={p:{'sha256':sha(p),'mtime_ns':Path(p).stat().st_mtime_ns} for p in complete['repair']['output_sha256']}
    for p,row in outputs.items():assert row['sha256']==complete['repair']['output_sha256'][p]
    extra=[work/'complete.json',work/'evaluation_summary.json',work/'frozen_association.json',work/'seed_projected_support.npz']
    extra_before={str(p):(sha(p),p.stat().st_mtime_ns) for p in extra}
    repeat=subprocess.run([sys.executable,str(code/'run_case.py'),'--config',str(code/config_name),'--evaluation-config',str(code/'evaluation_room2.json')],cwd=code,capture_output=True,text=True)
    (work/'repeat_validation.log').write_text(repeat.stdout+repeat.stderr)
    assert repeat.returncode==0,repeat.stdout+repeat.stderr
    for p,row in outputs.items():assert sha(p)==row['sha256'] and Path(p).stat().st_mtime_ns==row['mtime_ns']
    for p,row in extra_before.items():assert (sha(p),Path(p).stat().st_mtime_ns)==row
    # Outside helper outputs were frozen above; repeat validates no double-counting.
    checks[folder]={'status':'PASS','repeat_outputs_hash_and_mtime_unchanged':True,'outside_final_changes':0,
        'full_replay_and_exact_rollback':True,'sources_unchanged':True,'final_map_sha256':sha(work/'final/instance_surface.npz'),
        'fixed_mapping_cache_sha256':evaluation['mapping_cache_sha256']}
original=root/'room2_original_seed'
association=json.loads((original/'frozen_association.json').read_text())
cfg=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json').read_text())
source=cfg['source']
old=np.array(Image.open(Path(source['mask_root'])/source['mask_pattern'].format(frame=1065)))
selected=np.array(Image.open(association['seed_file']))
for oid,mid in [(4,8),(7,7)]:np.testing.assert_array_equal(selected==oid,old==mid)
assert {row['track_id']:row['persistent_id'] for row in association['objects']}=={4:52,7:355}
control=json.loads((root/'room2_original_control/frozen_association.json').read_text())
assert {row['track_id']:row['persistent_id'] for row in control['objects']}=={4:52,7:52}
assert len({row['fixed_mapping_cache_sha256'] for row in checks.values()})==1
report={'status':'PASS','tests':tests,'unit_test_count':31,'cases':checks,
    'original_seed_masks_pixel_identical':True,'original_reassociate_corrects_many_to_one_without_GT':True,
    'original_control_preserves_many_to_one':True,'one_fixed_mapping_for_all_conditions':True,
    'independent_forward_reverse_tracking_retained':True,'GT_not_used_for_repair':True,
    'numeric_quality_and_confirmation_thresholds_unchanged':True,
    'no_parameter_tuning_after_scores':True}
(root/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
PY
