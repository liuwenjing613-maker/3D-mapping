"""Read-only audit of the frozen P1-A1 frame-to-surface vote ledger; no GT."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

BASE = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
OUT = Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/baseline')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def audit(scene):
    started = time.monotonic()
    native = BASE / f'native/{scene}/P1-A1/surface_p0'
    report_path = native / 'materialization_report.json'
    report = json.loads(report_path.read_text())
    assoc_path = BASE / f'native/{scene}/P1-A1/association/associations.jsonl'
    source_paths = [report_path, assoc_path, native/'surface_instance_frame_votes.npz',
                    native/'surface_evidence.npz', native/'instance_surface.npz']
    before = {str(p):sha(p) for p in source_paths}
    assert before[str(assoc_path)] == report['association_decisions_sha256']
    assert before[str(native/'instance_surface.npz')] == report['instance_surface_sha256']
    assert before[str(native/'surface_evidence.npz')] == report['surface_evidence_sha256']
    lookup = {}
    maximum = 0
    for line in assoc_path.read_text().splitlines():
        row = json.loads(line)
        key = (int(row['frame_id']), int(row['mask_local_id']))
        assert key not in lookup and int(row['instance_id']) > 0
        lookup[key] = int(row['instance_id'])
        maximum = max(maximum, int(row['instance_id']))
    base = maximum + 1
    with np.load(native/'surface_instance_frame_votes.npz', allow_pickle=False) as z:
        keys = z['surface_point_index'].astype(np.int64) * base + z['instance_id']
        votes = z['frame_votes'].astype(np.int64)
    order = np.argsort(keys, kind='stable')
    keys, votes = keys[order], votes[order]
    assert np.all(keys[1:] > keys[:-1]) and np.all(votes > 0)
    replay = np.zeros(len(keys), np.int64)
    count = int(report['tsdf_surface_points'])
    point_observed_frames = np.zeros(count, np.int32)
    conflict_frames = np.zeros(count, np.int32)
    extra_votes = np.zeros(count, np.int32)
    shared_id_frames = np.zeros(count, np.int32)
    rows, support_hashes = [], {}
    records = report['frame_records']
    assert [int(r['frame_id']) for r in records] == list(range(0, 2000, 5))
    for index, record in enumerate(records):
        fid = int(record['frame_id'])
        path = Path(record['support_file'])
        digest = sha(path)
        assert digest == record['support_sha256']
        support_hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as z:
            assert int(z['frame_id'][0]) == fid
            points = z['surface_point_index'].astype(np.int64)
            local = z['mask_local_id'].astype(np.int64)
        assert np.all((points >= 0) & (points < count)) and np.all(local > 0)
        table = np.full(int(local.max()) + 1 if len(local) else 1, -1, np.int64)
        for mid in np.unique(local):
            table[mid] = lookup[(fid, int(mid))]
        raw_keys = points * base + table[local]
        frame_keys, multiplicity = np.unique(raw_keys, return_counts=True)
        assert len(frame_keys) == int(record['frame_instance_votes'])
        positions = np.searchsorted(keys, frame_keys)
        assert np.all(positions < len(keys)) and np.array_equal(keys[positions], frame_keys)
        replay[positions] += 1
        frame_points, id_counts = np.unique(frame_keys // base, return_counts=True)
        point_observed_frames[frame_points] += 1
        bad = id_counts > 1
        bad_points = frame_points[bad]
        conflict_frames[bad_points] += 1
        extra_votes[bad_points] += (id_counts[bad] - 1).astype(np.int32)
        shared_points = np.unique(frame_keys[multiplicity > 1] // base)
        shared_id_frames[shared_points] += 1
        row = {'frame_id': fid, 'region_surface_pairs':len(points),
               'distinct_point_instance_votes':len(frame_keys),
               'observed_surface_points':len(frame_points),
               'distinct_multi_id_point_frame_events':int(np.count_nonzero(bad)),
               'extra_same_frame_votes':int(np.sum(id_counts[bad] - 1)),
               'max_distinct_ids_at_one_point':int(id_counts.max()) if len(id_counts) else 0,
               'same_id_region_duplicates_already_removed':len(points)-len(frame_keys),
               'same_id_shared_point_frame_events':len(shared_points)}
        rows.append(row)
        if (index+1)%100 == 0:
            print(json.dumps({'stage':'read_only_vote_audit','scene':scene,'frames':index+1}),flush=True)
    np.testing.assert_array_equal(replay, votes)
    observed_point_frames = int(point_observed_frames.sum())
    bad_point_frames = int(conflict_frames.sum())
    extra = int(extra_votes.sum())
    assert int(votes.sum()) == observed_point_frames + extra
    raw_report = json.loads((BASE/f'raw_inputs/{scene}/surface_regions/materialization_report.json').read_text())
    assert extra == int(raw_report['same_frame_multi_id_points'])
    assert all(sha(p) == digest for p,digest in before.items())
    result = {'status':'PASS', 'scene':scene, 'purpose':'read_only_frozen_vote_correctness_audit',
              'GT_read':False, 'baseline_modified':False,
              'frame_count':len(rows), 'tsdf_surface_points':count,
              'baseline_aggregate_bit_exactly_reproduced':True,
              'frame_support_checksums_verified':len(support_hashes),
              'source_sha256':before,
              'total_point_instance_votes':int(votes.sum()),
              'total_observed_point_frames':observed_point_frames,
              'distinct_multi_id_point_frame_events':bad_point_frames,
              'distinct_multi_id_surface_points':int(np.count_nonzero(conflict_frames)),
              'extra_same_frame_votes':extra,
              'multi_id_point_frame_fraction':bad_point_frames/observed_point_frames,
              'max_distinct_ids_at_one_point_in_one_frame':max(r['max_distinct_ids_at_one_point'] for r in rows),
              'same_id_region_duplicates_already_removed':sum(r['same_id_region_duplicates_already_removed'] for r in rows),
              'same_id_shared_point_frame_events':sum(r['same_id_shared_point_frame_events'] for r in rows),
              'old_statistic_semantics':'same_frame_multi_id_points sums (distinct_ID_count - 1) across point-frame groups; it is not a distinct surface-point count',
              'seconds':round(time.monotonic()-started,2), 'per_frame':rows}
    dump(OUT/f'{scene}_frozen_vote_audit.json',result)
    np.savez_compressed(OUT/f'{scene}_audit_point_counts.npz', observed_frame_count=point_observed_frames,
                        multi_id_frame_count=conflict_frames, extra_same_frame_votes=extra_votes,
                        shared_same_id_frame_count=shared_id_frames)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_frame','source_sha256')},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    import sys
    OUT.mkdir(parents=True,exist_ok=True)
    audit(sys.argv[1])
