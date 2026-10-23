"""Independently verify completed branch checkpoints; never modifies map inputs."""
from pathlib import Path
import hashlib, json, time
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
OLD = Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
PRIOR_A = Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023/checkpoint')
HASH_CACHE = {}

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    p = Path(p).resolve()
    stat = p.stat()
    key = (str(p), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]

def verify_hashes(values):
    for path, expected in values.items():
        assert sha(path) == expected, ('changed source', path)
    return len(values)

def scalar_values(value):
    return {k: v for k, v in value.items() if not isinstance(v, (dict, list))}

def check(group, n):
    root = OUT / str(n) / group
    commit_path = root / 'repair/commit_complete.json'
    if not commit_path.exists():
        return {'status': 'PENDING', 'branch': group, 'checkpoint': n}
    m = read(root / 'mapping/mapping_complete.json')
    assert m['status'] == 'PASS' and m['frame_count'] == n
    assert m['frame_ids'] == list(range(0, n * 5, 5))
    assert m['surface_policy'] == 'checkpoint' and m['surface_rebuild_frames'] == [n]
    for k in ['geometry_incremental', 'association_incremental', 'checkpoint_strict_republication_complete',
              'strict_same_frame_point_at_most_one_vote', 'all_prior_layers_retained',
              'historical_committed_entries_unchanged', 'prior_source_hashes_unchanged']:
        assert m[k] is True, k
    assert m['GT_used'] is False and m['future_inputs_used'] is False
    previous_path = (PRIOR_A / 'repair/commit_complete.json' if group == 'A' and n == 300 else
                     OLD / 'repair100/B/commit_complete.json' if group == 'B' and n == 200 else
                     OUT / str(n - 100) / group / 'repair/commit_complete.json')
    previous = read(previous_path)
    assert m['prior_commit_sha256'] == sha(previous_path)
    first = read(root / f'mapping/frame{n-99}_state_audit.json')
    assert first['status'] == 'PASS' and first['repaired_state_used']
    assert first['frame_id'] == (n - 100) * 5
    assert first['checkpoint_sha256'] == previous['checkpoint_sha256']
    assert first['restored_map_version'] == first['map_version_used_before_association'] == previous['map_version']
    restore = read(root / 'mapping/restore_audit.json')
    assert restore['status'] == 'PASS' and restore['all_source_pixel_samples_exact'] and restore['all_journal_PIDs_exact']
    selection = read(root / 'candidates/selection_complete.json')
    qa = read(root / 'candidates/online_qa.json')
    track = read(root / 'tracking/complete.json')
    repair = read(root / 'repair/repair_complete.json')
    commit = read(commit_path)
    for value in [selection, qa, track, repair, commit]:
        assert value['status'] == 'PASS'
    assert commit['group'] == group and commit['processed_frames'] == n
    assert commit['last_raw_frame'] == (n - 1) * 5
    assert commit['GT_used'] is False and commit['original_baseline_and_prior_commit_unchanged']
    commit_hashes = {commit[key]: commit[key + '_sha256'] for key in
                     ['checkpoint', 'identity_counts', 'journal_manifest', 'source_pixel_support',
                      'observation_lineage', 'audit', 'final_map', 'geometry']}
    verify_hashes(commit_hashes)
    assert commit['repair_complete_sha256'] == sha(root / 'repair/repair_complete.json')
    audit = read(commit['audit'])
    required = ['prior_committed_entries_preserved_before_current_repair', 'every_old_entry_original_sample_voxels_exact',
                'unchanged_supports_preserved', 'roundtrip_checkpoint_exact', 'identity_counts_roundtrip_exact',
                'durable_sample_pixel_roundtrip_exact', 'surface_journal_reproduces_all_candidates_and_votes',
                'baseline_and_final_accumulated_votes_exact', 'delta_and_rollback_exact', 'surface_evidence_fields_exact',
                'native_labels_exact', 'strict_max_one_vote_per_frame_surface_point',
                'original_baseline_and_prior_commit_unchanged']
    assert audit['status'] == 'PASS'
    for key in required:
        assert audit[key] is True, key
    checked = verify_hashes(m['source_hashes']) + verify_hashes(audit['input_sha256'])
    checked += verify_hashes(audit['source_mask_sha256'])
    access_count, blocked = 0, []
    for folder in [root / 'io_audit', root / 'mapping/io_audit']:
        for path in folder.glob('*.jsonl'):
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    if row.get('probe'):
                        continue
                    access_count += 1
                    if row.get('blocked'):
                        blocked.append(row)
    assert not blocked, blocked[:3]
    with np.load(commit['final_map'], allow_pickle=False) as final:
        labels = final['instance_id']
        map_summary = {'surface_points': len(labels), 'assigned_points': int(np.count_nonzero(labels > 0)),
                       'positive_instances': len(np.unique(labels[labels > 0])),
                       'geometry_xyz_sha256': hashlib.sha256(final['xyz_m'].tobytes()).hexdigest()}
    timing = read(root / 'mapping/mapping_timing.json')
    return {'status': 'PASS', 'branch': group, 'checkpoint': n,
            'previous_commit': str(previous_path), 'previous_commit_sha256': sha(previous_path),
            'commit': str(commit_path), 'commit_sha256': sha(commit_path),
            'first_resumed_frame': first, 'surface_rebuild_frames': m['surface_rebuild_frames'],
            'source_files_verified': checked, 'commit_checks': {k: audit[k] for k in required},
            'causal_access_audit': {'recorded_nonprobe_opens': access_count, 'blocked_nonprobe_opens': len(blocked)},
            'candidate_range': selection['candidate_raw_frame_range'],
            'retained_seeds': selection['retained'], 'tracking': scalar_values(track),
            'mapping_timing': {k: timing[k] for k in ['total_seconds', 'restore_seconds', 'main_mapping_seconds',
                               'main_loop_including_checkpoint_seconds', 'checkpoint_export_seconds', 'stage_seconds']},
            'selection_production_seconds': selection['production_seconds'],
            'selection_gpu_wait_seconds': selection['gpu_wait_seconds'],
            'repair_summary': scalar_values(repair), 'commit_summary': scalar_values(commit),
            'final_surface': map_summary, 'GT_used_by_algorithm': False}

def main():
    rows = [check(g, n) for g in ['A', 'B'] for n in ([300, 400] if g == 'A' else [200, 300, 400])]
    adapter = read(OUT / 'adapter_audit.json')
    unchanged = []
    for row in adapter['unchanged_numerical_files']:
        assert sha(HERE / row['file']) == sha(row['source']) == row['source_sha256']
        unchanged.append(row['file'])
    status = 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'IN_PROGRESS'
    result = {'status': status, 'scene': 'room0', 'branches': ['A', 'B'], 'mapping_frames': 400,
              'mapping_raw_frame_ids': list(range(0, 2000, 5)), 'generated_unix': time.time(),
              'unchanged_numerical_files': unchanged, 'checkpoints': rows,
              'scope': 'Full optimized causal branch runs; A/B are candidate-search strategies, not original/optimized modes.',
              'not_claimed': 'Full-scene equivalence to the every-frame baseline; earlier real A200 Gate threshold flip remains documented.'}
    dest = OUT / 'verification/causal_audit.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': status, 'checkpoints': [{k: r[k] for k in ['branch', 'checkpoint', 'status']} for r in rows],
                      'verified_files': len(HASH_CACHE), 'path': str(dest)}))

if __name__ == '__main__':
    main()
