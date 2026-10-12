"""Independent pilot cases: exact baseline controls, seed and short-track replacement."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib, json, os, sys, time, traceback
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from evidence_replacement import EvidenceReplacement
from repair_association import associate_seed, compose_frame_mask

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
SEEDS = ROOT / 'human_seeds_20ad6139511b'
BASE = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
INPUT = Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')
SNAPSHOT = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
sys.path.insert(0, str(SNAPSHOT / 'revisable_instance_map/src'))
from revisable_instance_map.surface_evidence import project_frame_regions, frame_instance_keys, reduce_surface_votes, CONFIRMED
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.offline_surface_assignment import assign_surface, AssignmentSettings
FIELDS = ['top1_instance_id', 'top1_votes', 'top2_instance_id', 'top2_votes', 'total_frame_votes', 'confidence', 'margin', 'state']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def code_hashes():
    here = Path(__file__).parent
    return {f: sha(here / f) for f in ['run_repairs.py', 'repair_association.py', 'evidence_replacement.py']}


class Scene:
    def __init__(self, scene):
        self.scene = scene
        self.native = BASE / f'native/{scene}/P1-A1/surface_p0'
        self.final_source = BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
        self.report = json.loads((self.native / 'materialization_report.json').read_text())
        self.records = {r['frame_id']: r for r in self.report['frame_records']}
        assert sorted(self.records) == list(range(0, 2000, 5))
        self.evidence = load_npz(self.native / 'surface_evidence.npz')
        self.pair = load_npz(self.native / 'surface_instance_frame_votes.npz')
        native_map = load_npz(self.native / 'instance_surface.npz')
        final_map = load_npz(self.final_source)
        self.xyz, self.rgb = native_map['xyz_m'], native_map['rgb']
        self.original, self.final = native_map['instance_id'], final_map['instance_id']
        for array in ['xyz_m', 'rgb']:
            np.testing.assert_array_equal(native_map[array], final_map[array])
            np.testing.assert_array_equal(native_map[array], self.evidence[array])
        geometry = INPUT / scene / 'geometry/surface.ply'
        assert sha(geometry) == self.report['tsdf_surface_sha256']
        cloud = o3d.io.read_point_cloud(str(geometry))
        assert cloud.has_normals() and len(cloud.points) == len(self.xyz)
        np.testing.assert_allclose(np.asarray(cloud.points), self.xyz, atol=1e-5, rtol=0)
        self.normals = np.asarray(cloud.normals).astype(np.float32)
        self.tree = cKDTree(self.xyz.astype(np.float64))
        self.source = ReplicaFrameSource(INPUT / scene / 'configs/raw.json')
        associations = BASE / f'native/{scene}/P1-A1/association/associations.jsonl'
        assert sha(associations) == self.report['association_decisions_sha256']
        self.lookup = {}
        self.maximum_id = 0
        for line in associations.read_text().splitlines():
            row = json.loads(line)
            self.lookup[(row['frame_id'], row['mask_local_id'])] = row['instance_id']
            self.maximum_id = max(self.maximum_id, row['instance_id'])
        self.settings = AssignmentSettings()
        self.settings_sha = hashlib.sha256(json.dumps(asdict(self.settings), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        post = json.loads((self.final_source.parent.parent / 'materialization_report.json').read_text())
        assert self.settings_sha == post['settings_sha256']
        self.source_hashes = {str(p): sha(p) for p in [self.native / 'instance_surface.npz', self.final_source,
            self.native / 'surface_evidence.npz', self.native / 'surface_instance_frame_votes.npz', associations]}

    def verify_unchanged(self):
        assert all(sha(p) == digest for p, digest in self.source_hashes.items())

    def baseline_control(self):
        out = ROOT / 'repair/baseline_controls' / self.scene
        path = out / 'complete.json'
        if path.exists():
            d = json.loads(path.read_text())
            assert d['code_sha256'] == code_hashes() and d['source_hashes'] == self.source_hashes
            return d
        started = time.monotonic()
        variants, _, stats = assign_surface(self.xyz, self.normals, self.rgb, self.evidence,
                                            self.pair, self.original, self.settings)
        np.testing.assert_array_equal(variants['holes_geodesic'], self.final)
        self.verify_unchanged()
        report = {'status': 'PASS', 'scene': self.scene, 'final_map_bit_identical': True,
                  'native_evidence_controls': str(ROOT / 'preflight' / (self.scene + '.json')),
                  'postprocessing_settings_sha256': self.settings_sha,
                  'source_hashes': self.source_hashes, 'code_sha256': code_hashes(),
                  'GT_used': False, 'seconds': round(time.monotonic() - started, 2)}
        dump(path, report)
        print(json.dumps({'baseline_control': self.scene, 'seconds': report['seconds'], 'status': 'PASS'}), flush=True)
        return report

    def old_frame(self, fid, base):
        frame = self.source.load_frame(fid)
        support_file = Path(self.records[fid]['support_file'])
        assert sha(support_file) == self.records[fid]['support_sha256']
        support = load_npz(support_file)
        mask_file = self.source.mask_root / self.source.config['source']['mask_pattern'].format(frame=fid)
        assert sha(mask_file) == str(support['source_mask_sha256'][0])
        points, local, stats = project_frame_regions(frame, self.tree, pixel_stride=2, max_distance_m=.015, workers=8)
        np.testing.assert_array_equal(points, support['surface_point_index'])
        np.testing.assert_array_equal(local, support['mask_local_id'])
        table = np.full(int(frame.mask_local.max()) + 1, -1, np.int32)
        for mid in np.unique(frame.mask_local):
            if mid:
                table[mid] = self.lookup[(fid, int(mid))]
        keys = frame_instance_keys(points, local, table, base)
        assert len(keys) == self.records[fid]['frame_instance_votes']
        return frame, table, keys, points

    def prepare_case(self, seed):
        uid = seed['case_uid']
        out = ROOT / 'repair' / uid
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'frozen_association.json'
        if path.exists():
            report = json.loads(path.read_text())
            assert report['code_sha256'] == code_hashes() and report['seed_sha256'] == seed['seed_sha256']
            return report
        base = self.maximum_id + len(seed['objects']) + 1
        fid = seed['source_choice']['frame']
        seed_file = SEEDS / seed['seed_file']
        assert sha(seed_file) == seed['seed_sha256']
        labels = np.array(Image.open(seed_file), dtype=np.uint16)
        frame, old_table, old_keys, visible = self.old_frame(fid, base)
        pts, locals_, stats = project_frame_regions(replace(frame, mask_local=labels), self.tree,
                                                   pixel_stride=2, max_distance_m=.015, workers=8)
        ids = [o['track_id'] for o in seed['objects']]
        rows = associate_seed(pts, locals_, visible, self.final, ids, self.maximum_id)
        if seed['source_choice']['choice'] == 'original':
            # The selected original masks are a native evidence control. Retain
            # their exact frozen associations, including any existing many-to-one.
            for row, obj in zip(rows, seed['objects']):
                row['persistent_id'] = self.lookup[(fid, obj['native_mask_id'])]
                row['association'] = 'exact_original_observation_control'
        for row, obj in zip(rows, seed['objects']):
            row['native_mask_id'] = obj['native_mask_id']
        report = {'status': 'PASS', 'case_uid': uid, 'scene': self.scene,
                  'seed_frame': fid, 'seed_sha256': seed['seed_sha256'],
                  'source_manifest_sha256': sha(SEEDS / 'seed_manifest.json'),
                  'baseline_final_sha256': sha(self.final_source), 'instance_base': base,
                  'association_threshold_dice': .5, 'objects': rows,
                  'projection': stats, 'GT_used': False, 'code_sha256': code_hashes()}
        dump(path, report)
        np.savez_compressed(out / 'seed_projected_support.npz', surface_point_index=pts, track_id=locals_)
        return report

    def repair(self, seed, condition):
        started = time.monotonic()
        uid = seed['case_uid']
        out = ROOT / 'repair' / uid / condition
        out.mkdir(parents=True, exist_ok=True)
        done = out / 'complete.json'
        if done.exists():
            report = json.loads(done.read_text())
            assert report['code_sha256'] == code_hashes() and report['seed_sha256'] == seed['seed_sha256']
            assert sha(out / 'final/instance_surface.npz') == report['final_map_sha256']
            return report
        association = self.prepare_case(seed)
        frozen_ids = {r['track_id']: r['persistent_id'] for r in association['objects']}
        base = association['instance_base']
        keys = self.pair['surface_point_index'].astype(np.int64) * base + self.pair['instance_id']
        order = np.argsort(keys, kind='stable')
        ledger = EvidenceReplacement(keys[order], self.pair['frame_votes'][order])
        original_digest = ledger.digest()
        fid = seed['source_choice']['frame']
        frames = [fid] if condition == 'seed_only' else seed['mapping_vote_frames']
        track_report = None
        if condition == 'seed_plus_short_track':
            track_report = json.loads((ROOT / 'tracking' / uid / 'complete.json').read_text())
            assert track_report['status'] == 'PASS' and track_report['seed_sha256'] == seed['seed_sha256']
            tracked = {r['frame']: r for r in track_report['frames']}
        transactions, scope_arrays = [], []
        for frame_id in frames:
            frame, old_table, old_keys, old_points = self.old_frame(frame_id, base)
            selected_file = SEEDS / seed['seed_file'] if condition == 'seed_only' else Path(tracked[frame_id]['label_file'])
            expected_hash = seed['seed_sha256'] if condition == 'seed_only' else tracked[frame_id]['label_sha256']
            assert sha(selected_file) == expected_hash
            selected = np.array(Image.open(selected_file), dtype=np.uint16)
            if frame_id == fid:
                np.testing.assert_array_equal(selected, np.array(Image.open(SEEDS / seed['seed_file'])))
            combined, table = compose_frame_mask(frame.mask_local, selected, old_table, frozen_ids)
            points, local, stats = project_frame_regions(replace(frame, mask_local=combined), self.tree,
                                                       pixel_stride=2, max_distance_m=.015, workers=8)
            new_keys = frame_instance_keys(points, local, table, base)
            selected_points = np.unique(points[local > int(frame.mask_local.max())])
            scope_arrays.append(selected_points)
            changed = np.setxor1d(old_keys, new_keys)
            assert np.all(np.isin(changed // base, selected_points))
            tx = f'{uid}:{condition}:f{frame_id:06d}'
            assert ledger.replace(tx, old_keys, new_keys)
            updated_digest = ledger.digest()
            assert not ledger.replace(tx, old_keys, new_keys) and ledger.digest() == updated_digest
            tx_file = out / f'f{frame_id:06d}_transaction.npz'
            np.savez_compressed(tx_file, old_keys=old_keys, new_keys=new_keys, instance_base=base,
                                selected_surface_point_index=selected_points)
            transactions.append({'transaction_id': tx, 'frame': frame_id,
                'old_votes': len(old_keys), 'new_votes': len(new_keys),
                'removed_pairs': len(np.setdiff1d(old_keys, new_keys)),
                'added_pairs': len(np.setdiff1d(new_keys, old_keys)),
                'scope_surface_points': len(selected_points), 'selected_mask_sha256': expected_hash,
                'transaction_sha256': sha(tx_file), 'repeat_is_idempotent': True,
                'unselected_pixels_and_outside_scope_votes_unchanged': True})
        scope = np.unique(np.concatenate(scope_arrays))
        evidence = reduce_surface_votes(ledger.keys, ledger.votes, len(self.xyz), base,
            self.report['min_confirmed_votes'], self.report['min_confirmed_ratio'])
        native_labels = np.where(evidence['state'] == CONFIRMED, evidence['top1_instance_id'], -1).astype(np.int32)
        keep = np.ones(len(self.xyz), bool)
        keep[scope] = False
        for field in FIELDS:
            np.testing.assert_array_equal(evidence[field][keep], self.evidence[field][keep])
        pair = {'surface_point_index': (ledger.keys // base).astype(np.int32),
                'instance_id': (ledger.keys % base).astype(np.int32), 'frame_votes': ledger.votes.astype(np.int32)}
        if ledger.digest() == original_digest:
            final = self.final.copy()
            stats = {'same_as_verified_baseline': True}
        else:
            variants, _, stats = assign_surface(self.xyz, self.normals, self.rgb, evidence, pair,
                                                native_labels, self.settings)
            final = variants['holes_geodesic']
        if seed['source_choice']['choice'] == 'original' and condition == 'seed_only':
            assert ledger.digest() == original_digest
            np.testing.assert_array_equal(final, self.final)
        for name in ['native', 'final']:
            (out / name).mkdir(exist_ok=True)
        np.savez_compressed(out / 'native/surface_evidence.npz', xyz_m=self.xyz, rgb=self.rgb, **evidence)
        np.savez_compressed(out / 'native/surface_instance_frame_votes.npz', **pair)
        np.savez_compressed(out / 'native/instance_surface.npz', xyz_m=self.xyz, rgb=self.rgb, instance_id=native_labels)
        np.savez_compressed(out / 'final/instance_surface.npz', xyz_m=self.xyz, rgb=self.rgb, instance_id=final)
        np.save(out / 'intervention_surface_points.npy', scope)
        digest = ledger.digest()
        for tx in reversed(transactions):
            ledger.undo(tx['transaction_id'])
        assert ledger.digest() == original_digest
        self.verify_unchanged()
        native_changed = int(np.sum(native_labels != self.original))
        final_changed = int(np.sum(final != self.final))
        report = {'status': 'PASS', 'case_uid': uid, 'scene': self.scene, 'ROI': seed['ROI'],
                  'condition': condition, 'choice': seed['source_choice']['choice'],
                  'seed_sha256': seed['seed_sha256'], 'association_sha256': sha(ROOT / 'repair' / uid / 'frozen_association.json'),
                  'code_sha256': code_hashes(), 'base_source_hashes': self.source_hashes,
                  'postprocessing_settings_sha256': self.settings_sha,
                  'scope': 'replace selected pixels; inherit every unselected original pixel; deduplicate complete frame votes',
                  'instance_base': base, 'transactions': transactions,
                  'mapping_frames': frames, 'vote_weight_per_frame_surface_identity': 1,
                  'intervention_surface_points': len(scope), 'native_label_changes': native_changed,
                  'final_label_changes': final_changed,
                  'final_changes_outside_intervention': int(np.sum((final != self.final) & keep)),
                  'selected_surface_publication_fraction_before': float(np.mean(self.final[scope] > 0)) if len(scope) else None,
                  'selected_surface_publication_fraction_after': float(np.mean(final[scope] > 0)) if len(scope) else None,
                  'baseline_ledger_sha256': original_digest, 'repaired_ledger_sha256': digest,
                  'repeat_is_idempotent': True, 'undo_restores_original_votes_exactly': True,
                  'original_inputs_unchanged': True, 'GT_used_for_repair': False,
                  'final_map_sha256': sha(out / 'final/instance_surface.npz'),
                  'postprocessing_statistics': stats, 'seconds': round(time.monotonic() - started, 2)}
        dump(done, report)
        print(json.dumps({k: report[k] for k in ['case_uid', 'condition', 'native_label_changes', 'final_label_changes', 'seconds']}), flush=True)
        return report


def main():
    root = ROOT / 'repair'
    root.mkdir(exist_ok=True)
    manifest = json.loads((SEEDS / 'seed_manifest.json').read_text())
    baseline = json.loads((BASE / 'run_manifest.json').read_text())
    assert manifest['status'] == baseline['status'] == 'PASS'
    for filename, expected in baseline['source_sha256'].items():
        assert sha(SNAPSHOT / filename) == expected, filename
    status = {'status': 'RUNNING', 'phase': 'baseline_full_postprocessing_control',
              'source_manifest_sha256': sha(SEEDS / 'seed_manifest.json'), 'code_sha256': code_hashes(),
              'total_cases': 10, 'ready_cases': 9, 'target_objects': 55,
              'baseline_controls': [], 'completed_conditions': [], 'skipped_cases': [],
              'GT_used_for_repair': False}
    dump(root / 'status.json', status)
    scenes = {}
    for seed in manifest['seeds']:
        if seed['scene'] not in scenes:
            data = Scene(seed['scene'])
            scenes[seed['scene']] = data
            status['baseline_controls'].append(data.baseline_control())
            dump(root / 'status.json', status)
    status['phase'] = 'seed_only_repair'
    for seed in manifest['seeds']:
        if seed['status'] != 'READY':
            record = {'case_uid': seed['case_uid'], 'scene': seed['scene'], 'ROI': seed['ROI'],
                      'status': 'SKIP_UNRELIABLE', 'both_conditions_equal_baseline': True,
                      'included_in_denominator': True}
            dump(root / seed['case_uid'] / 'skip.json', record)
            status['skipped_cases'].append(record)
        else:
            report = scenes[seed['scene']].repair(seed, 'seed_only')
            status['completed_conditions'].append({k: report[k] for k in ['case_uid', 'condition', 'final_label_changes', 'seconds']})
        dump(root / 'status.json', status)
    status['phase'] = 'short_track_repair'
    dump(root / 'status.json', status)
    for seed in manifest['seeds']:
        if seed['status'] != 'READY':
            continue
        path = ROOT / 'tracking' / seed['case_uid'] / 'complete.json'
        while not path.exists():
            if (ROOT / 'tracking/failure.json').exists():
                raise RuntimeError('Tracking failed; completed seed-only results are preserved')
            status['waiting_for_tracking_case'] = seed['case_uid']
            dump(root / 'status.json', status)
            time.sleep(10)
        status.pop('waiting_for_tracking_case', None)
        report = scenes[seed['scene']].repair(seed, 'seed_plus_short_track')
        status['completed_conditions'].append({k: report[k] for k in ['case_uid', 'condition', 'final_label_changes', 'seconds']})
        dump(root / 'status.json', status)
    for scene in scenes.values():
        scene.verify_unchanged()
    status['status'] = 'PASS'
    status['phase'] = 'predictions_frozen_ready_for_posthoc_v3'
    dump(root / 'status.json', status)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(ROOT / 'repair/failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise
