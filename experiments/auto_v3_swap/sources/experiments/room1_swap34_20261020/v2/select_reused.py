"""Use frozen room1 measurements; the only rule change is stage ordering."""
from pathlib import Path
import copy
import hashlib
import json
from filter_rules import run_filters
from semantic_exemption import apply_semantic_exemption

C = Path(__file__).resolve().parent
R = Path('/data/chenkejun/CVPR/results/room1_swap34_20261020')
OLD = Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/room1')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, value):
    p = R / 'v2' / name
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    assert not (R / 'v2/complete.json').exists()
    (R / 'v2').mkdir(exist_ok=True)
    (R / 'v1').symlink_to(OLD / 'v1', target_is_directory=True)
    (R / 'v2/semantic').symlink_to(OLD / 'v2/semantic', target_is_directory=True)
    (R / 'v2/measurements.json').symlink_to(OLD / 'v2/measurements.json')
    original_complete = read(OLD / 'v2/complete.json')
    assert original_complete['status'] == 'PASS'
    for name, key in [('selection.json', 'selection_sha256'), ('seed_manifest.json', 'manifest_sha256')]:
        assert sha(OLD / 'v2' / name) == original_complete[key]
    locks = read(OLD / 'v2/source_lock.json')
    for path, digest in locks['source_sha256'].items():
        assert sha(path) == digest, path
    measurements = read(R / 'v2/measurements.json')
    semantic = read(R / 'v2/semantic/classification.json')
    before = read(OLD / 'v2/selection.json')
    raw = run_filters(measurements)
    save('selection_without_exemption.json', raw)
    after = apply_semantic_exemption(raw, semantic)
    assert after['stages'][:2] == before['stages'][:2]
    assert [s['stage'] for s in after['stages']] == [
        'semantic', 'covisible_iou_only', 'full_support_containment', 'skip_repair_B', 'pr_filter']
    after['selection_variant'] = 'containment_before_skip_repair_with_semantic_stage5_exemption'
    save('selection.json', after)
    kept = set(after['retained_group_ids'])
    old_ids = set(before['retained_group_ids'])
    assert kept == old_ids - {'G015', 'G051'} and len(kept) == 22
    manifest = read(R / 'v1/seed_manifest.json')
    selected = copy.deepcopy(manifest)
    selected['groups'] = [dict(copy.deepcopy(g), original_track_id=tid)
                          for tid, g in enumerate(manifest['groups'], 1) if g['id'] in kept]
    selected.update(status='automatic_v2_complete', scene='room1', mask_source_directory=str(R / 'v1'),
                    original_manifest_sha256=sha(R / 'v1/seed_manifest.json'),
                    selection_sha256=sha(R / 'v2/selection.json'), manual_overrides=[], GT_used=False)
    save('seed_manifest.json', selected)
    (R / 'v2/masks').symlink_to(OLD / 'v1/masks', target_is_directory=True)
    comparison = dict(status='PASS', old_count=len(old_ids), new_count=len(kept),
                      added=sorted(kept-old_ids), removed=sorted(old_ids-kept),
                      old_stages=before['stages'], new_stages=after['stages'],
                      measurements_sha256=sha(R / 'v2/measurements.json'),
                      source_measurements=str(OLD / 'v2/measurements.json'),
                      original_selection_sha256=sha(OLD / 'v2/selection.json'),
                      only_filter_order_changed=True, semantic_stage5_exemption_unchanged=True,
                      GT_used=False)
    save('order_comparison.json', comparison)
    save('source_lock.json', dict(source_sha256=locks['source_sha256'],
         original_source_lock_sha256=sha(OLD / 'v2/source_lock.json'),
         measurements_sha256=sha(R / 'v2/measurements.json'),
         code_sha256={p.name: sha(p) for p in C.glob('*.py')}, GT_used=False))
    save('complete.json', dict(status='PASS', input=len(manifest['groups']), retained=len(kept),
         stages=after['stages'], manual_overrides=[], selection_sha256=sha(R / 'v2/selection.json'),
         manifest_sha256=sha(R / 'v2/seed_manifest.json')))
    print(json.dumps(comparison), flush=True)

if __name__ == '__main__':
    main()
