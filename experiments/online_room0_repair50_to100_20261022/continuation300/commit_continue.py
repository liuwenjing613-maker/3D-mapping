"""Commit the authorized checkpoint repair to an isolated A/B identity checkpoint.

Every existing entry is reconstructed from its original sampled source pixels.
No mapping frame is advanced here. Surface votes and the association voxel
ledger remain separate authorities with their unchanged original semantics.
"""
from pathlib import Path
import os, sys, json, time, resource, traceback
START = time.perf_counter()
START_UNIX = time.time()
ATTEMPT_ID = str(int(START_UNIX * 1000)) + '-' + str(os.getpid())
def process_io():
    result = {}
    for line in Path('/proc/self/io').read_text().splitlines():
        key, value = line.split(':', 1)
        result[key] = int(value)
    return result
IO_START = process_io()
GROUP_REQUEST = os.environ.get('CVPR_TRACK_GROUP')
assert GROUP_REQUEST in ('A', 'B'), 'Set CVPR_TRACK_GROUP=A or B explicitly'
HERE = Path(__file__).resolve().parent
BASE_CODE = HERE.parent
sys.path.insert(0, str(HERE))
from bootstrap_continue import *
from dataclasses import replace, asdict
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.identity_evidence import ObservationSupport, FrameIdentityEvidence
from revisable_instance_map.observations import RawInstanceObservation, source_pixel_indices, extract_frame_observations
from history_core_ops import strict_frame
from joint_vote_ops import batch_delta

assert GROUP == GROUP_REQUEST and FIDS == list(range(0, CURRENT_RAW + 1, 5))
R = ROOT / 'repair'
PRIOR_RAW = (OLD_COUNT - 1) * 5
OUT = R / 'commit_candidate'
NAMESPACE = 'repair' + str(CHECKPOINT) + GROUP

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def emit(stage, **values):
    row = dict(status='PASS' if stage == 'commit_complete' else 'RUNNING', group=GROUP, checkpoint_processed_frames=CHECKPOINT, stage=stage,
               seconds=time.perf_counter() - START, unix=time.time(), **values)
    write(R / 'commit_progress.json', row)
    print(json.dumps(row), flush=True)

def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)

def sample_pixels(frame, observation, associator):
    # Exact original implementation from commit_state.py, not per-piece resampling.
    pixels = source_pixel_indices(frame, observation)
    if associator.valid_first_sampling:
        depth = frame.depth_m.ravel()[pixels]
        pixels = pixels[np.isfinite(depth) & (depth > 0) & (depth < 10)]
        if not len(pixels):
            return pixels, np.empty((0, 3), np.int32)
    count = min(len(pixels), associator.max_points_per_observation)
    pixels = pixels[np.linspace(0, len(pixels) - 1, num=count, dtype=np.int64)]
    rows, cols = np.divmod(pixels, frame.camera.width)
    depth = frame.depth_m.ravel()[pixels].astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
    pixels, rows, cols, depth = pixels[valid], rows[valid], cols[valid], depth[valid]
    camera = np.column_stack(((cols - frame.camera.cx) * depth / frame.camera.fx,
                              (rows - frame.camera.cy) * depth / frame.camera.fy, depth))
    world = camera @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
    return pixels, np.floor(world / associator.voxel_size_m).astype(np.int32)

def voxel_tuple(voxels):
    return tuple(tuple(int(value) for value in row) for row in np.unique(voxels, axis=0))

def cat(chunks):
    return np.concatenate(chunks) if chunks else np.empty(0, np.int64)

def aggregate(chunks):
    return np.unique(cat(chunks), return_counts=True)

def keys(points, ids, base):
    assert np.all((ids > 0) & (ids < base))
    return np.unique(points.astype(np.int64) * base + ids)

def project_layers(frame, layers, scene, base):
    chunks = []
    for layer in layers:
        points, ids = r.project_frame_regions(
            replace(frame, mask_local=layer), scene.tree,
            pixel_stride=2, max_distance_m=.015, workers=8)[:2]
        chunks.append(keys(points, ids, base))
    raw = np.unique(cat(chunks))
    effective = strict_frame(raw, base)
    assert len(effective) == len(np.unique(effective // base))
    return raw, effective

def encode(pair, base):
    encoded = pair['surface_point_index'].astype(np.int64) * base + pair['instance_id']
    order = np.argsort(encoded)
    return encoded[order], pair['frame_votes'][order].astype(np.int64)

def main():
    complete_path = R / 'repair_complete.json'
    complete = read(complete_path)
    assert complete['status'] == 'PASS' and complete['group'] == GROUP
    assert complete['mapping_frames'] == CHECKPOINT and complete['raw_cutoff'] == CURRENT_RAW
    assert complete['objects'] == read(R / 'seed_manifest.json')['selected_masks']
    assert complete['main_mapping_paused'] and complete['GT_used'] is False
    assert read(R / 'partial/mapping_complete.json')['status'] == 'PASS'
    output_complete = R / 'commit_complete.json'
    if output_complete.exists():
        old = read(output_complete)
        assert old['status'] == 'PASS' and old['group'] == GROUP
        assert old['repair_complete_sha256'] == digest(complete_path)
        assert old['checkpoint_sha256'] == digest(old['checkpoint'])
        assert old['audit_sha256'] == digest(old['audit'])
        print(json.dumps(dict(status='PASS', group=GROUP, reused_commit=True,
                              checkpoint=old['checkpoint'])), flush=True)
        return

    emit('verify_inputs')
    timings = {}
    tick = time.perf_counter()
    prior_path = B / 'prior_commit.json'
    prior = read(prior_path)
    assert prior['status'] == 'PASS' and prior['processed_frames'] == OLD_COUNT
    assert prior['last_raw_frame'] == PRIOR_RAW
    for key in ['checkpoint', 'journal_manifest', 'source_pixel_support', 'audit']:
        assert digest(prior[key]) == prior[key + '_sha256'], key
    prior_audit = read(prior['audit'])
    assert prior_audit['status'] == 'PASS'
    prior_manifest = read(prior['journal_manifest'])
    prior_layers = {row['frame_id']: row['layer_files'] for row in prior_manifest['frames']}
    assert sorted(prior_layers) == FIDS[:OLD_COUNT]
    prior_pixels = r.load_npz(prior['source_pixel_support'])
    prior_lineage = read(prior['observation_lineage'])
    assert digest(prior['observation_lineage']) == prior['observation_lineage_sha256']
    prior_sample_rows = {str(oid): i for i, oid in enumerate(prior_pixels['observation_ids'])}
    prior_lineage_rows = {row['observation_id']: row for row in prior_lineage['entries']}
    assert set(prior_sample_rows) == set(prior_lineage_rows)
    manifest_path = R / 'journal/manifest.json'
    manifest = read(manifest_path)
    assert manifest['status'] == 'PASS' and manifest['group'] == GROUP
    assert manifest['raw_cutoff'] == CURRENT_RAW
    manifest_by_frame = {row['frame_id']: row for row in manifest['frames']}
    assert len(manifest['frames']) == len(manifest_by_frame) == CHECKPOINT
    assert sorted(manifest_by_frame) == FIDS
    association_path = B / 'association/observation_support_3cm.npz'
    original_observations_path = B / 'association/observations.jsonl'
    final_root = R / 'partial/whole_to_partial/room0'
    final_map_path = final_root / 'final/instance_surface.npz'
    input_paths = [
        complete_path, R / 'partial/mapping_complete.json', prior_path,
        Path(prior['checkpoint']), Path(prior['audit']), Path(prior['source_pixel_support']),
        Path(prior['journal_manifest']), Path(prior['observation_lineage']),
        association_path, original_observations_path, manifest_path,
        R / 'global_association.json', final_map_path,
        final_root / 'native/instance_surface.npz',
        final_root / 'native/surface_evidence.npz',
        final_root / 'native/surface_instance_frame_votes.npz',
    ]
    input_hashes = {str(path): digest(path) for path in input_paths}
    association = read(R / 'global_association.json')
    assert association['GT_used'] is False
    base = int(association['instance_base'])
    scene = r.Scene('room0')
    assert scene.commit == prior
    associator = OnlineVoxelAssociator.from_checkpoint(association_path)
    assert associator.last_processed_frame_id == CURRENT_RAW
    associator.identity_evidence.validate()
    parameters = associator.checkpoint_parameters()
    old_version = associator.map_version
    old_next_id = associator.next_instance_id
    old_entries = dict(associator.observation_support)
    assert old_version == prior['map_version'] + (CHECKPOINT - OLD_COUNT)
    original_observations = {}
    for line in original_observations_path.read_text().splitlines():
        observation = RawInstanceObservation(**json.loads(line))
        assert observation.frame_id in FIDS
        assert observation.observation_id not in original_observations
        original_observations[observation.observation_id] = observation
    entries_by_frame = defaultdict(list)
    for entry in old_entries.values():
        assert entry.frame_id in FIDS
        entries_by_frame[entry.frame_id].append(entry)
    checkpoint_prior = OnlineVoxelAssociator.from_checkpoint(prior['checkpoint'])
    historical = {key: value for key, value in old_entries.items() if value.frame_id <= PRIOR_RAW}
    assert historical == checkpoint_prior.observation_support, 'Committed historical entries changed during continuation'
    assert set(historical) == set(prior_sample_rows)
    del checkpoint_prior
    lineage_counts = Counter('prior_committed_entry' if entry.frame_id <= PRIOR_RAW else
                             'new_original_CF' for entry in old_entries.values())
    assert sum(lineage_counts.values()) == len(old_entries)
    timings['input_restore_and_identity_validation'] = time.perf_counter() - tick

    entries, events, receipts, lineage = [], [], [], []
    sample_pixels_out, sample_voxels_out = [], []
    unchanged_ids, revised_ids, added_records = [], [], []
    all_before, all_after, removed, added = [], [], [], []
    seen, emitted_ids = set(), set()
    checked_source_hashes = {}
    journal_hashes = {}
    source_observation_counts = Counter()
    event_start = max((int(event.get('event_id', 0)) for event in associator.revision_events), default=0)
    timings.update(frame_io_and_lineage=0., old_entry_sampling_and_split=0.,
                   new_adopted_sampling=0., independent_surface_replay=0.)

    def append_entry(entry, pixels, voxels, source, layer_index, journal_row):
        assert entry.observation_id not in emitted_ids, entry.observation_id
        assert len(pixels) == len(voxels)
        assert voxel_tuple(voxels) == entry.voxels, entry.observation_id
        assert entry.frame_id in FIDS
        emitted_ids.add(entry.observation_id)
        entries.append(entry)
        sample_pixels_out.append(np.asarray(pixels, np.int32))
        sample_voxels_out.append(np.asarray(voxels, np.int32).reshape(-1, 3))
        lineage.append(dict(
            observation_id=entry.observation_id, frame_id=entry.frame_id,
            instance_id=entry.instance_id, assignment_version=entry.assignment_version,
            layer_index=layer_index, journal_layer=journal_row['path'],
            journal_sha256=journal_row['sha256'],
            source_observation_id=source['observation_id'],
            sampling_source_kind=source['kind'],
            sampling_source_mask=source['mask_path'],
            sampling_source_mask_sha256=source['mask_sha256'],
            sampling_source_mask_local_id=source['mask_local_id'],
            sampling_source_pixel_count=source['pixel_count'],
            sampled_pixel_count=len(pixels), unique_voxels=len(entry.voxels),
        ))

    for frame_number, fid in enumerate(FIDS, 1):
        tick = time.perf_counter()
        frame = scene.source.load_frame(fid)
        old_layers = scene.identity_layers(fid, frame)
        old_count = scene.expected_layer_count(fid)
        row = manifest_by_frame[fid]
        assert row['old_layer_count'] == old_count == len(old_layers)
        journal_rows = row['layer_files']
        assert len(journal_rows) == old_count + 1
        new_layers = []
        for layer_index, layer_row in enumerate(journal_rows):
            path = Path(layer_row['path'])
            assert path.parent.resolve() == (R / 'journal').resolve()
            assert path.name == f'f{fid:06d}_layer{layer_index}.png'
            assert layer_row['kind'] == ('history' if layer_index < old_count else 'adopted' + str(CHECKPOINT))
            assert digest(path) == layer_row['sha256']
            journal_hashes[str(path)] = layer_row['sha256']
            with Image.open(path) as image:
                new_layers.append(np.asarray(image, np.uint16).copy())
            assert new_layers[-1].shape == frame.mask_local.shape
            assert int(new_layers[-1].max()) < base
        for layer_index in range(old_count):
            assert not np.any((old_layers[layer_index] == 0) & (new_layers[layer_index] > 0)), 'Historical layer gained pixels outside its source'
        raw_mask_path = scene.source.mask_root / scene.source.config['source']['mask_pattern'].format(frame=fid)
        raw_mask_sha = digest(raw_mask_path)
        checked_source_hashes[str(raw_mask_path)] = raw_mask_sha
        if fid <= PRIOR_RAW:
            for layer_row in prior_layers[fid]:
                path = Path(layer_row['path'])
                assert digest(path) == layer_row['sha256']
                checked_source_hashes[str(path)] = layer_row['sha256']
        timings['frame_io_and_lineage'] += time.perf_counter() - tick

        tick = time.perf_counter()
        raw_before, effective_before = project_layers(frame, old_layers, scene, base)
        raw_after, effective_after = project_layers(frame, new_layers, scene, base)
        transaction_path = R / 'partial/transactions' / f'f{fid:06d}.npz'
        transaction = r.load_npz(transaction_path)
        np.testing.assert_array_equal(raw_before, transaction['baseline_candidate_keys'])
        np.testing.assert_array_equal(effective_before, transaction['baseline_frame_keys'])
        np.testing.assert_array_equal(raw_after, transaction['whole_to_partial_candidate_keys'])
        np.testing.assert_array_equal(effective_after, transaction['whole_to_partial_frame_keys'])
        baseline_support = scene.support(fid)
        baseline_raw = keys(baseline_support['surface_point_index'],
                            baseline_support['candidate_persistent_id'], base)
        np.testing.assert_array_equal(raw_before, baseline_raw)
        all_before.append(effective_before)
        all_after.append(effective_after)
        removed.append(np.setdiff1d(effective_before, effective_after))
        added.append(np.setdiff1d(effective_after, effective_before))
        timings['independent_surface_replay'] += time.perf_counter() - tick

        tick = time.perf_counter()
        cached_original_samples = {}
        frame_revised, frame_unchanged = 0, 0
        for old in entries_by_frame[fid]:
            assert old.observation_id not in seen
            seen.add(old.observation_id)
            if fid <= PRIOR_RAW:
                idx = prior_sample_rows[old.observation_id]
                offset = prior_pixels['offsets']
                sl = slice(offset[idx], offset[idx + 1])
                pixels = prior_pixels['flat_source_pixel_index'][sl]
                voxels = prior_pixels['sampled_voxel_coordinates'][sl]
                layer_index = int(prior_pixels['layer_index'][idx])
                original_row = prior_lineage_rows[old.observation_id]
                assert int(prior_pixels['frame_ids'][idx]) == fid
                assert int(prior_pixels['instance_ids'][idx]) == old.instance_id
                assert original_row['sampling_source_mask_sha256'] == old.source_mask_sha256
                yy, xx = np.divmod(pixels, frame.camera.width)
                depth = frame.depth_m.ravel()[pixels].astype(np.float64)
                assert np.all(np.isfinite(depth) & (depth > 0) & (depth < 10))
                camera = np.column_stack(((xx-frame.camera.cx)*depth/frame.camera.fx,
                                          (yy-frame.camera.cy)*depth/frame.camera.fy, depth))
                world = camera @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
                np.testing.assert_array_equal(np.floor(world / associator.voxel_size_m).astype(np.int32), voxels)
                kind = original_row['sampling_source_kind']
                source = dict(observation_id=original_row['source_observation_id'], kind=kind,
                              mask_path=original_row['sampling_source_mask'],
                              mask_sha256=original_row['sampling_source_mask_sha256'],
                              mask_local_id=original_row['sampling_source_mask_local_id'],
                              pixel_count=original_row['sampling_source_pixel_count'])
            else:
                assert old.observation_id in original_observations
                observation = original_observations[old.observation_id]
                assert observation.frame_id == fid
                assert observation.source_mask_sha256 == raw_mask_sha == old.source_mask_sha256
                pixels, voxels = sample_pixels(frame, observation, associator)
                layer_index = 0
                kind = 'original_CF_sample'
                source = dict(observation_id=observation.observation_id, kind=kind,
                              mask_path=str(raw_mask_path), mask_sha256=observation.source_mask_sha256,
                              mask_local_id=observation.mask_local_id, pixel_count=observation.pixel_count)
            assert voxel_tuple(voxels) == old.voxels, (old.observation_id, 'Original sampling does not reproduce checkpoint')
            assert np.all(old_layers[layer_index].ravel()[pixels] == old.instance_id)
            source_observation_counts[kind] += 1
            ids = new_layers[layer_index].ravel()[pixels]
            if np.all(ids == old.instance_id):
                append_entry(old, pixels, voxels, source, layer_index, journal_rows[layer_index])
                unchanged_ids.append(old.observation_id)
                frame_unchanged += 1
                continue
            frame_revised += 1
            revised_ids.append(old.observation_id)
            pieces = []
            for pid in sorted(np.unique(ids[ids > 0]).tolist()):
                select = ids == pid
                suffix = '' if pid == old.instance_id else f'/{NAMESPACE}-id{pid}'
                new = replace(old, observation_id=old.observation_id + suffix,
                              instance_id=int(pid), voxels=voxel_tuple(voxels[select]),
                              assignment_version=old.assignment_version + 1)
                append_entry(new, pixels[select], voxels[select], source, layer_index, journal_rows[layer_index])
                pieces.append(dict(observation_id=new.observation_id, instance_id=int(pid),
                                   voxels=len(new.voxels), sample_pixels=int(select.sum())))
            events.append(dict(
                event_id=event_start + len(events) + 1, type='PIXEL_PARTIAL_E1_REPAIR' + str(CHECKPOINT),
                group=GROUP, observation_id=old.observation_id, frame_id=fid,
                source_layer=layer_index, old_instance_id=old.instance_id,
                new_instance_id=int(pieces[0]['instance_id']) if pieces else old.instance_id,
                old_assignment_version=old.assignment_version,
                new_assignment_version=old.assignment_version + 1,
                map_version_before=old_version, map_version_after=old_version + 1,
                pieces=pieces, withdrawn_sample_pixels=int((ids == 0).sum()),
                reason='authorized' + str(CHECKPOINT) + '_multilayer_partial_core_E1_source_pixel_journal'))
        timings['old_entry_sampling_and_split'] += time.perf_counter() - tick

        tick = time.perf_counter()
        adopted100 = new_layers[-1]
        adopted100_frame = replace(frame, mask_local=adopted100)
        adopted100_path = Path(journal_rows[-1]['path'])
        new_observations = extract_frame_observations(
            adopted100_frame, 'Replica', 'room0', journal_rows[-1]['sha256'],
            observation_namespace=NAMESPACE)
        for observation in new_observations:
            pixels, voxels = sample_pixels(adopted100_frame, observation, associator)
            pid = observation.mask_local_id
            assert np.all(adopted100.ravel()[pixels] == pid)
            entry = ObservationSupport(observation.observation_id, fid, observation.source_mask_sha256,
                                       pid, voxel_tuple(voxels), 1)
            source = dict(observation_id=observation.observation_id,
                          kind=NAMESPACE + '_adopted_original_sample',
                          mask_path=str(adopted100_path),
                          mask_sha256=observation.source_mask_sha256,
                          mask_local_id=observation.mask_local_id, pixel_count=observation.pixel_count)
            append_entry(entry, pixels, voxels, source, old_count, journal_rows[-1])
            added_records.append(dict(
                **asdict(observation), instance_id=pid, layer_index=old_count,
                source_mask=str(adopted100_path), sampled_pixels=len(pixels), voxels=len(entry.voxels)))
        timings['new_adopted_sampling'] += time.perf_counter() - tick
        receipts.append(dict(
            frame_id=fid, old_layer_count=old_count, new_layer_count=len(new_layers),
            old_entries_verified=len(entries_by_frame[fid]), unchanged_entries=frame_unchanged,
            revised_entries=frame_revised, new_adopted_entries=len(new_observations),
            baseline_candidate_keys=len(raw_before), final_candidate_keys=len(raw_after),
            baseline_effective_votes=len(effective_before), final_effective_votes=len(effective_after),
            removed_effective_votes=len(removed[-1]), added_effective_votes=len(added[-1]),
            all_old_samples_match_checkpoint=True, all_layer_candidates_match=True,
            strict_frame_votes_match=True,
        ))
        scene._layers.pop(fid, None)
        scene._support.pop(fid, None)
        if frame_number % 10 == 0:
            emit('reconstruct_lineage_and_surface_votes', processed_frames=frame_number,
                 old_entries_verified=len(seen), revised_entries=len(revised_ids),
                 added_entries=len(added_records))
    assert seen == set(old_entries)
    assert len(unchanged_ids) + len(revised_ids) == len(old_entries)

    tick = time.perf_counter()
    before_keys, before_votes = aggregate(all_before)
    after_keys, after_votes = aggregate(all_after)
    baseline_keys, baseline_votes = encode(scene.pair, base)
    np.testing.assert_array_equal(before_keys, baseline_keys)
    np.testing.assert_array_equal(before_votes, baseline_votes)
    expected_keys, expected_votes = encode(r.load_npz(final_root / 'native/surface_instance_frame_votes.npz'), base)
    np.testing.assert_array_equal(after_keys, expected_keys)
    np.testing.assert_array_equal(after_votes, expected_votes)
    removed_keys, added_keys = cat(removed), cat(added)
    delta_keys, delta_votes = batch_delta(before_keys, before_votes, removed_keys, added_keys)
    np.testing.assert_array_equal(delta_keys, after_keys)
    np.testing.assert_array_equal(delta_votes, after_votes)
    rollback_keys, rollback_votes = batch_delta(after_keys, after_votes, added_keys, removed_keys)
    np.testing.assert_array_equal(rollback_keys, before_keys)
    np.testing.assert_array_equal(rollback_votes, before_votes)
    assert np.all((after_votes > 0) & (after_votes <= CHECKPOINT))
    evidence = r.reduce_surface_votes(after_keys, after_votes, len(scene.xyz), base, 2, .67)
    saved_evidence = r.load_npz(final_root / 'native/surface_evidence.npz')
    for field in r.FIELDS:
        np.testing.assert_array_equal(evidence[field], saved_evidence[field])
    native = r.load_npz(final_root / 'native/instance_surface.npz')
    expected_native = np.where(evidence['state'] == r.CONFIRMED, evidence['top1_instance_id'], -1).astype(np.int32)
    np.testing.assert_array_equal(native['instance_id'], expected_native)
    for path in (final_map_path, final_root / 'native/instance_surface.npz'):
        current = r.load_npz(path)
        np.testing.assert_array_equal(current['xyz_m'], scene.xyz)
        np.testing.assert_array_equal(current['rgb'], scene.rgb)
    timings['surface_accumulation_delta_rollback_and_fields'] = time.perf_counter() - tick
    emit('surface_votes_exact', effective_vote_pairs=len(after_keys))

    tick = time.perf_counter()
    associator.identity_evidence = FrameIdentityEvidence.rebuild(entries)
    associator.observation_support = associator.identity_evidence.entries
    associator._rebuild_instance_index()
    associator.map_version = old_version + 1
    associator.revision_events.extend(events)
    reserved = [int(row['persistent_id']) for row in association['objects']]
    associator.next_instance_id = max(old_next_id, max([0, *associator.instances, *reserved]) + 1)
    assert associator.last_processed_frame_id == CURRENT_RAW
    assert associator.checkpoint_parameters() == parameters
    for observation_id in unchanged_ids:
        assert associator.observation_support[observation_id] == old_entries[observation_id]
    associator.identity_evidence.validate()
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT / 'association_checkpoint.npz'
    counts_path = OUT / 'identity_counts.npz'
    associator.save_checkpoint(checkpoint_path)
    associator.save_identity_counts(counts_path)
    restored = OnlineVoxelAssociator.from_checkpoint(checkpoint_path)
    assert restored.observation_support == associator.observation_support
    assert restored.next_instance_id == associator.next_instance_id
    assert restored.map_version == old_version + 1
    assert restored.last_processed_frame_id == CURRENT_RAW
    assert restored.checkpoint_parameters() == parameters
    assert restored.revision_events == associator.revision_events
    assert restored.voxel_to_instances == associator.voxel_to_instances
    restored.identity_evidence.validate()
    saved_counts = r.load_npz(counts_path)
    for field, values in restored.identity_evidence.export_arrays().items():
        np.testing.assert_array_equal(values, saved_counts[field])
    timings['identity_rebuild_save_and_roundtrip'] = time.perf_counter() - tick

    tick = time.perf_counter()
    provenance = dict(
        observation_ids=np.asarray([entry.observation_id for entry in entries], dtype=str),
        frame_ids=np.asarray([entry.frame_id for entry in entries], np.int32),
        instance_ids=np.asarray([entry.instance_id for entry in entries], np.int32),
        layer_index=np.asarray([row['layer_index'] for row in lineage], np.int16),
        offsets=np.r_[0, np.cumsum([len(pixels) for pixels in sample_pixels_out], dtype=np.int64)],
        flat_source_pixel_index=cat(sample_pixels_out).astype(np.int32),
        sampled_voxel_coordinates=np.concatenate(sample_voxels_out).astype(np.int32).reshape(-1, 3),
        voxel_size_m=np.asarray([associator.voxel_size_m], np.float64),
    )
    provenance_path = OUT / 'source_pixel_support.npz'
    atomic_npz(provenance_path, **provenance)
    saved_provenance = r.load_npz(provenance_path)
    for field, values in provenance.items():
        np.testing.assert_array_equal(values, saved_provenance[field])
    assert provenance['offsets'][-1] == len(provenance['flat_source_pixel_index'])
    journal_link = OUT / 'pixel_journal'
    if not journal_link.exists():
        journal_link.symlink_to(R / 'journal', target_is_directory=True)
    assert journal_link.resolve() == (R / 'journal').resolve()
    write(OUT / 'observation_lineage.json', dict(
        group=GROUP, status='PASS', sampling='original source sampling, then pixel filters; no per-piece resampling',
        pixel_support=str(provenance_path), pixel_support_sha256=digest(provenance_path), entries=lineage))
    write(OUT / 'added_observations.json', added_records)
    write(OUT / 'revision_events.json', events)
    timings['durable_pixel_provenance_and_receipts'] = time.perf_counter() - tick

    tick = time.perf_counter()
    scene.verify_unchanged()
    for path, sha in input_hashes.items():
        assert digest(path) == sha, path
    for path, sha in {**checked_source_hashes, **journal_hashes}.items():
        assert digest(path) == sha, path
    timings['final_source_hash_verification'] = time.perf_counter() - tick
    usage = resource.getrusage(resource.RUSAGE_SELF)
    current_io = process_io()
    report = dict(
        status='PASS', group=GROUP, attempt_id=ATTEMPT_ID, trigger_processed_frames=CHECKPOINT, processed_frames=CHECKPOINT, last_raw_frame=CURRENT_RAW,
        seconds=time.perf_counter() - START, started_unix=START_UNIX, ended_unix=time.time(),
        timings=timings, peak_cpu_rss_kib=usage.ru_maxrss, cpu_user_seconds=usage.ru_utime,
        cpu_system_seconds=usage.ru_stime,
        process_io_delta={key: current_io[key] - IO_START.get(key, 0) for key in current_io},
        io_note='Linux process counters: rchar/wchar include cache; read_bytes/write_bytes are storage I/O.',
        GPU_used=False, checkpoint_processed_frames=CHECKPOINT, old_map_version=old_version, map_version=associator.map_version,
        old_next_instance_id=old_next_id, next_instance_id=associator.next_instance_id,
        old_identity_entries=len(old_entries), old_entry_lineage_counts=dict(lineage_counts),
        old_entry_sampling_sources=dict(source_observation_counts),
        unchanged_observations=len(unchanged_ids), revised_observations=len(revised_ids),
        added_observations=len(added_records), identity_entries=len(associator.observation_support),
        prior_committed_entries_preserved_before_current_repair=True,
        every_old_entry_original_sample_voxels_exact=True, unchanged_supports_preserved=True,
        roundtrip_checkpoint_exact=True, identity_counts_roundtrip_exact=True,
        durable_sample_pixel_roundtrip_exact=True,
        surface_journal_reproduces_all_candidates_and_votes=True,
        baseline_and_final_accumulated_votes_exact=True, delta_and_rollback_exact=True,
        surface_evidence_fields_exact=True, native_labels_exact=True,
        strict_max_one_vote_per_frame_surface_point=True,
        association_evidence_rule='unchanged one vote per (frame, voxel, instance), reference counted',
        geometric_surface_indices_not_persistent_keys=True,
        original_baseline_and_prior_commit_unchanged=True,
        main_mapping_paused=True, continuation_started=False,
        checkpoint=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
        identity_counts=str(counts_path), identity_counts_sha256=digest(counts_path),
        pixel_journal=str(journal_link), journal_manifest=str(manifest_path),
        journal_manifest_sha256=input_hashes[str(manifest_path)],
        source_pixel_support=str(provenance_path), source_pixel_support_sha256=digest(provenance_path),
        observation_lineage=str(OUT / 'observation_lineage.json'),
        observation_lineage_sha256=digest(OUT / 'observation_lineage.json'),
        final_map=str(final_map_path), final_map_sha256=input_hashes[str(final_map_path)],
        geometry=str(B / 'geometry/tsdf_grid.npz'),
        geometry_sha256=digest(B / 'geometry/tsdf_grid.npz'),
        repair_complete_sha256=input_hashes[str(complete_path)],
        input_sha256=input_hashes, source_mask_sha256=checked_source_hashes,
        journal_layer_sha256=journal_hashes, GT_used=False, frames=receipts, prior_processed_frames=OLD_COUNT)
    audit_path = OUT / 'commit_audit.json'
    write(audit_path, report)
    write(output_complete, dict(
        status='PASS', state='COMMITTED_BRANCH_ONLY', group=GROUP, attempt_id=ATTEMPT_ID,
        checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256'],
        identity_counts=report['identity_counts'], identity_counts_sha256=report['identity_counts_sha256'],
        pixel_journal=report['pixel_journal'], journal_manifest=report['journal_manifest'],
        journal_manifest_sha256=report['journal_manifest_sha256'],
        source_pixel_support=report['source_pixel_support'],
        source_pixel_support_sha256=report['source_pixel_support_sha256'],
        observation_lineage=report['observation_lineage'],
        observation_lineage_sha256=report['observation_lineage_sha256'],
        audit=str(audit_path), audit_sha256=digest(audit_path), seconds=report['seconds'],
        timings=timings, last_raw_frame=CURRENT_RAW, processed_frames=CHECKPOINT,
        map_version=associator.map_version, next_instance_id=associator.next_instance_id,
        final_map=report['final_map'], final_map_sha256=report['final_map_sha256'],
        geometry=report['geometry'], geometry_sha256=report['geometry_sha256'],
        repair_complete_sha256=report['repair_complete_sha256'],
        main_mapping_paused=True, continuation_started=False,
        original_baseline_and_prior_commit_unchanged=True, GT_used=False))
    emit('commit_complete', map_version=associator.map_version,
         revised_observations=len(revised_ids), added_observations=len(added_records))
    print(json.dumps({key: report[key] for key in (
        'status', 'group', 'seconds', 'old_identity_entries', 'unchanged_observations',
        'revised_observations', 'added_observations', 'identity_entries', 'next_instance_id')}), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = dict(status='FAILED', group=GROUP, attempt_id=ATTEMPT_ID,
                       traceback=traceback.format_exc(), unix=time.time(),
                       seconds=time.perf_counter() - START)
        write(R / 'commit_attempts' / (ATTEMPT_ID + '_failure.json'), failure)
        write(R / 'commit_failure.json', failure)
        raise
