"""Lock the isolated trial against the archived baseline and current R2 evaluator."""
from pathlib import Path
import hashlib,json,time
C=Path('/home/chenkejun/CVPR/experiments/trusted_full_online_20261104')
O=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
BASE=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=read(C/'upstream_source_lock.json');files={};excluded=[]
for name,h in old['sha256'].items():
    p=Path(name)
    assert sha(p)==h,('UPSTREAM_CHANGED',name)
    if p.is_relative_to(BASE):
        q=C/p.relative_to(BASE)
        if not q.exists():
            assert p.is_relative_to(BASE/'main'),('MISSING_SOURCE',name)
            excluded.append(name);continue
    else:q=p
    files[str(q)]=sha(q)
for p in C.glob('*.py'):files[str(p)]=sha(p)
for p in (C/'online').glob('*/trusted_*.py'):files[str(p)]=sha(p)
for p in (C/'online').glob('*/test_*.py'):files[str(p)]=sha(p)
tests=[O/'policy_tests/integration_test.json',O/'tests/exact_tracking_cache/complete.json',O/'comparison_selfcheck/comparison.json',O/'tests/shared_prefix_guard/complete.json',O/'tests/model_cache_integration.json',O/'tests/model_digest_memo/complete.json',O/'tests/model_cache_prefix_fence.json',O/'policy_tests/family_skip_compatibility.json',O/'tests/empty_fixed_compatibility.json']
for p in tests:assert read(p)['status']=='PASS',str(p)
run_id=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
lock=dict(run_id=run_id,sha256=files,upstream_run_id=old['run_id'],upstream_lock_sha256=sha(C/'upstream_source_lock.json'),excluded_main_only_files=len(excluded),unix=time.time())
(C/'source_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
plan=dict(status='AUTHORIZED_READY',variant='online_B',scenes=['room0','room1','room2','office0','office1','office2','office3','office4'],checkpoints=[50,100,200,300,400],baseline_results='/data/chenkejun/CVPR/results/unified_full_depth_20261101',algorithm_change='Archived trusted partition, including association feedback and committed partitioned pixel history',policy_source='/home/chenkejun/CVPR/experiments/online_v2_trusted_partition_20261027/full_run/room2',prefix50='Read-only identical committed baseline before first protected continuation',tracking_reuse='Only exact RGB/seed/ordered objects/model/inference-code/config fingerprint; no association or repair reuse after prefix50',evaluation='main object_observed_repair revision2 fixed350 shared ordered geometry and correspondence, same finalB, no finalGT in prefix',source_run_id=run_id,source_lock_sha256=sha(C/'source_lock.json'),tests={str(p):sha(p) for p in tests},max_parallel_scenes=4,gpu_pool=list(range(8)),resource_policy='Shared GPU locks plus dynamic RAM reservation; 2 scene pilots then up to4; no other jobs stopped',unix=time.time())
(O/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'status':'PASS','run_id':run_id,'locked_files':len(files),'source_lock':str(C/'source_lock.json')}))
