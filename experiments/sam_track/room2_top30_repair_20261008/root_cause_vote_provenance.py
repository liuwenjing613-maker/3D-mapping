"""Read-only original/SAM vote provenance for newly unassigned surface points."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
sys.path.insert(0, '/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
import run_repairs as r


def main():
    report = json.loads((ROOT / 'root_cause_20261008/root_cause_audit.json').read_text())
    for path, digest in report['source_hashes'].items():
        assert r.sha(path) == digest
    scene = r.Scene('room2')
    folder = ROOT / 'validated/full_track'
    evidence = r.load_npz(folder / 'native/surface_evidence.npz')
    final = r.load_npz(folder / 'final/instance_surface.npz')['instance_id']
    lost = (scene.final > 0) & (final <= 0)
    retained_old_votes = np.zeros(len(final), np.int32)
    same_frame_old_and_new = np.zeros(len(final), np.int32)
    new_identity_votes = np.zeros(len(final), np.int32)
    new_identity_only_frames = np.zeros(len(final), np.int32)
    unchanged_frame_old_votes = np.zeros(len(final), np.int32)
    retained_baseline_identity_votes = np.zeros(len(final), np.int32)
    actual = np.zeros(len(final), np.int32)
    for fid in sorted(scene.records):
        path = folder / 'transactions' / ('f%06d.npz' % fid)
        if path.exists():
            z = r.load_npz(path)
            base = int(z['instance_base'][0])
            retained, updated = z['retained_old_frame_keys'], z['new_frame_keys']
            retained_p = retained // base
            updated_p, updated_id = updated // base, updated % base
            select = lost[updated_p] & (updated_id > scene.maximum_id)
            sam_p = updated_p[select]
            np.add.at(new_identity_votes, sam_p, 1)
            both = np.intersect1d(retained_p, sam_p)
            same_frame_old_and_new[both] += 1
            new_identity_only_frames[np.setdiff1d(sam_p, retained_p)] += 1
            np.add.at(retained_baseline_identity_votes, retained_p[retained % base == scene.final[retained_p]], 1)
        else:
            z = r.load_npz(Path(scene.records[fid]['support_file']))
            p, local = z['surface_point_index'].astype(np.int64), z['mask_local_id']
            base = scene.maximum_id + 1
            lookup = np.zeros(int(local.max()) + 1, np.int64)
            for mid in np.unique(local):
                lookup[mid] = scene.lookup[(fid, int(mid))]
            retained = np.unique(p * base + lookup[local])
            retained_p = retained // base
            np.add.at(unchanged_frame_old_votes, retained_p, 1)
            np.add.at(retained_baseline_identity_votes, retained_p[retained % base == scene.final[retained_p]], 1)
            updated_p = retained_p
        np.add.at(retained_old_votes, retained_p, 1)
        np.add.at(actual, updated_p, 1)
    np.testing.assert_array_equal(actual, evidence['total_frame_votes'])
    examples = []
    pairs = [(10, 357), (350, 10), (8, 356), (357, 10), (10, 358)]
    for first, second in pairs:
        ids = np.flatnonzero(lost & (evidence['top1_instance_id'] == first) & (evidence['top2_instance_id'] == second))
        index = int(ids[np.argsort(evidence['total_frame_votes'][ids])[len(ids) // 2]])
        examples.append({'surface_point_index': index, 'xyz_m': scene.xyz[index].tolist(),
                         'baseline_final_id': int(scene.final[index]), 'baseline_top1_id': int(scene.evidence['top1_instance_id'][index]),
                         'baseline_top1_votes': int(scene.evidence['top1_votes'][index]), 'baseline_total_votes': int(scene.evidence['total_frame_votes'][index]),
                         'final_top1_id': first, 'final_top1_votes': int(evidence['top1_votes'][index]),
                         'final_top2_id': second, 'final_top2_votes': int(evidence['top2_votes'][index]),
                         'final_total_votes': int(evidence['total_frame_votes'][index]), 'final_top1_fraction': float(evidence['confidence'][index]),
                         'retained_original_vote_units': int(retained_old_votes[index]),
                         'retained_baseline_identity_vote_units': int(retained_baseline_identity_votes[index]),
                         'new_identity_vote_units': int(new_identity_votes[index]),
                         'same_frame_retained_original_and_new_identity': int(same_frame_old_and_new[index])})
    counts = {
        'newly_gray_with_retained_original_votes': int((lost & (retained_old_votes > 0)).sum()),
        'newly_gray_with_retained_votes_for_baseline_identity': int((lost & (retained_baseline_identity_votes > 0)).sum()),
        'newly_gray_with_new_identity_votes': int((lost & (new_identity_votes > 0)).sum()),
        'newly_gray_with_retained_original_and_new_identity': int((lost & (retained_old_votes > 0) & (new_identity_votes > 0)).sum()),
        'newly_gray_with_same_frame_retained_original_and_new_identity': int((lost & (same_frame_old_and_new > 0)).sum()),
        'newly_gray_with_old_and_new_only_across_different_frames': int((lost & (retained_old_votes > 0) & (new_identity_votes > 0) & (same_frame_old_and_new == 0)).sum()),
        'newly_gray_with_votes_from_completely_unchanged_frames': int((lost & (unchanged_frame_old_votes > 0)).sum()),
        'retained_original_vote_units_on_newly_gray': int(retained_old_votes[lost].sum()),
        'new_identity_vote_units_on_newly_gray': int(new_identity_votes[lost].sum()),
        'same_frame_original_new_point_occurrences': int(same_frame_old_and_new[lost].sum()),
    }
    result = {'status': 'PASS', 'all_400_frame_vote_totals_reproduced_exactly': True,
              'definitions': {'new_identity': 'Persistent ID > original maximum 348, thus definitely SAM-originated in this batch',
                              'retained_original': 'Raw original observation support retained during each frame rebuild; updated and unchanged frames included',
                              'same_frame': 'Retained original and new-ID support on the exact same surface point in one frame; not necessarily the top-two pair'},
              'counts': counts, 'actual_point_examples': examples, 'source_hashes': report['source_hashes'],
              'source_code_sha256': r.sha(__file__), 'predictions_votes_masks_unchanged': True}
    scene.verify_unchanged()
    for path, digest in report['source_hashes'].items():
        assert r.sha(path) == digest
    (ROOT / 'root_cause_20261008/vote_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
