"""Increment geometry and identities; republish strict evidence on each current TSDF."""
from online_common import *
from dataclasses import asdict, replace
import sys
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import save_npz

sys.path.insert(0, str(SNAPSHOT / 'revisable_instance_map/src'))
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.geometry_tsdf import SparseTSDFGeometry
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.observations import extract_frame_observations
from revisable_instance_map.surface_evidence import project_frame_regions, reduce_surface_votes, STATE_NAMES, CONFIRMED
from revisable_instance_map.offline_surface_assignment import AssignmentSettings, assign_surface, surface_edges
from one_vote import effective_keys

def cloud_ply(path, xyz, colors):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)
    assert o3d.io.write_point_cloud(str(path), pc)

def strict_surface(frames, lookup, cloud, count, persist):
    xyz = np.asarray(cloud.points).astype(np.float32)
    rgb = np.clip(np.asarray(cloud.colors), 0, 1).astype(np.float32)
    if len(xyz) == 0:
        assert not persist, 'No TSDF surface at trigger checkpoint'
        return dict(processed_frames=count, last_frame_id=frames[-1].frame_id, surface_points=0,
            state_counts={name:0 for name in STATE_NAMES}, native_assigned_points=0,
            effective_votes=0, maximum_votes_per_point=0, same_frame_uniqueness_verified=True,
            observed_frame_ids=[f.frame_id for f in frames], note='No surface passes the unchanged TSDF weight threshold yet')
    tree = cKDTree(xyz)
    base = max(lookup.values()) + 1
    keys_per_frame, audit = [], []
    for frame in frames:
        points, local, stats = project_frame_regions(frame, tree, pixel_stride=2, max_distance_m=.015, workers=4)
        table = np.full(int(frame.mask_local.max()) + 1, -1, np.int64)
        for lid in np.unique(local):
            table[lid] = lookup[(frame.frame_id, int(lid))]
        keys = effective_keys(points, table[local], base, 'abstain')
        # Independent check from raw point/ID candidates, including same-ID duplicates.
        candidates = np.unique(np.column_stack((points, table[local])), axis=0)
        point_ids, candidate_counts = np.unique(candidates[:, 0], return_counts=True)
        valid_points = point_ids[candidate_counts == 1]
        expected = candidates[np.isin(candidates[:, 0], valid_points)]
        np.testing.assert_array_equal(keys, np.sort(expected[:, 0] * base + expected[:, 1]))
        assert len(keys) == len(np.unique(keys // base))
        keys_per_frame.append(keys)
        audit.append(dict(frame_id=frame.frame_id, candidate_pairs=len(candidates), effective_votes=len(keys),
                          ambiguous_points=int(np.sum(candidate_counts > 1)), **stats))
        if persist:
            np.savez_compressed(R / 'baseline/frame_support' / f'f{frame.frame_id:06d}.npz',
                frame_id=np.array([frame.frame_id]), surface_point_index=points, mask_local_id=local,
                source_mask_sha256=np.array([sha(R / f'inputs/masks/frame{frame.frame_id:06d}.png')]))
    keys, votes = np.unique(np.concatenate(keys_per_frame), return_counts=True)
    evidence = reduce_surface_votes(keys, votes, len(xyz), base, 2, .67)
    assert int(evidence['total_frame_votes'].max()) <= count
    native = np.where(evidence['state'] == CONFIRMED, evidence['top1_instance_id'], -1).astype(np.int32)
    pair = dict(surface_point_index=(keys // base).astype(np.int32), instance_id=(keys % base).astype(np.int32),
                frame_votes=votes.astype(np.int32))
    summary = dict(processed_frames=count, last_frame_id=frames[-1].frame_id, surface_points=len(xyz),
                   state_counts={name:int(np.sum(evidence['state'] == i)) for i,name in enumerate(STATE_NAMES)},
                   native_assigned_points=int(np.sum(native > 0)), effective_votes=int(votes.sum()),
                   maximum_votes_per_point=int(evidence['total_frame_votes'].max()),
                   same_frame_uniqueness_verified=True, observed_frame_ids=[f.frame_id for f in frames])
    if persist:
        baseline = R / 'baseline'
        np.savez_compressed(baseline / 'surface_evidence.npz', xyz_m=xyz, rgb=rgb, **evidence)
        np.savez_compressed(baseline / 'surface_instance_frame_votes.npz', **pair)
        np.savez_compressed(baseline / 'native/instance_surface.npz', xyz_m=xyz, rgb=rgb, instance_id=native)
        write(baseline / 'frame_vote_audit.json', audit)
        normals = np.asarray(cloud.normals).astype(np.float32)
        settings = AssignmentSettings()
        variants, _, stats = assign_surface(xyz, normals, rgb, evidence, pair, native, settings,
            progress=lambda stage, values: print('FINAL_ASSIGNMENT', stage, values, flush=True))
        labels = variants['holes_geodesic']
        np.savez_compressed(baseline / 'final/instance_surface.npz', xyz_m=xyz, rgb=rgb, instance_id=labels,
                            native_instance_ids=np.unique(labels[labels > 0]))
        summary.update(final_assigned_points=int(np.sum(labels > 0)), postprocessing_stats=stats,
                       final_unassigned_states={name:int(np.sum((labels <= 0) & (evidence['state'] == i)))
                                                for i,name in enumerate(STATE_NAMES)},
                       assignment_settings=asdict(settings), geometry_sha256=sha(R / 'geometry/surface.ply'))
        # Preserve the existing n02/p08 review graph definition on prefix geometry.
        graph_settings = replace(settings, normal_cos=.20, plane_tolerance_scale=.80,
                                 color_cost_weight=0., normal_cost_weight=0., query_chunk=25000)
        normalized = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        dist = tree.query(xyz, k=5, workers=4)[0][:, 4]
        scale = np.clip(dist, .75 * np.median(dist), 1.25 * np.median(dist))
        graph, _ = surface_edges(tree, xyz, normalized, rgb, scale, np.arange(len(xyz), dtype=np.int32), graph_settings)
        graph = graph.maximum(graph.T).tocsr()
        graph.sort_indices()
        save_npz(R / 'geometry/review_surface_graph_n02_p08.npz', graph)
        palette = np.random.default_rng(20261009).uniform(.15, .95, (max(base, int(labels.max())+1), 3))
        colors = np.full((len(xyz), 3), .62)
        colors[labels > 0] = palette[labels[labels > 0]]
        cloud_ply(R / 'ply/room0_prefix20_instances.ply', xyz, colors)
        remaining = labels <= 0
        state_colors = np.array([[0.,0.,0.],[1.,.8,0.],[.2,.8,.2],[1.,0.,0.]])
        cloud_ply(R / 'ply/room0_prefix20_three_states.ply', xyz[remaining], state_colors[evidence['state'][remaining]])
        write(baseline / 'mapping_summary.json', summary)
    return summary

def main():
    if (R / 'mapping_complete.json').exists():
        print('MAPPING_ALREADY_COMPLETE', flush=True)
        return
    for relative in ['baseline/frame_support', 'baseline/native', 'baseline/final', 'association', 'geometry', 'ply']:
        (R / relative).mkdir(parents=True, exist_ok=True)
    source = ReplicaFrameSource(C / 'config.json')
    assert source.frame_ids == FIDS
    associator = OnlineVoxelAssociator(allow_multiple_observations_per_instance_per_frame=True,
        improved_association=False, valid_first_sampling=False, association_mode='probabilistic',
        relative_weight_temperature=.1, null_candidate_score=None)
    geometry = SparseTSDFGeometry(block_count=100000)
    frames, decisions, observations, timeline, lookup = [], [], [], [], {}
    start = time.time()
    for index, fid in enumerate(FIDS, 1):
        import sitecustomize
        sitecustomize.current_cutoff = fid
        frame = source.load_frame(fid)
        new_observations = extract_frame_observations(frame, 'Replica', 'room0', sha(R / f'inputs/masks/frame{fid:06d}.png'))
        current = associator.process_frame(frame, new_observations)
        assert len(current) == len(new_observations)
        decisions.extend(current)
        observations.extend(asdict(o) for o in new_observations)
        for item in current:
            lookup[(fid, item['mask_local_id'])] = item['instance_id']
        integration = geometry.integrate(frame)
        frames.append(frame)
        cloud = geometry.extract_point_cloud(weight_threshold=1.).to_legacy()
        if index == 20:
            assert o3d.io.write_point_cloud(str(R / 'geometry/surface.ply'), cloud)
            geometry.save(R / 'geometry/tsdf_grid.npz')
        summary = strict_surface(frames, lookup, cloud, index, index == 20)
        summary.update(integration=integration, persistent_instances=len(associator.instances), seconds=time.time()-start)
        timeline.append(summary)
        write(R / 'mapping_progress.json', summary)
        write(R / 'mapping_timeline.json', timeline)
        print('PROCESSED', index, fid, summary['surface_points'], summary['state_counts'], flush=True)
    associator.save_checkpoint(R / 'association/observation_support_3cm.npz')
    for name, rows in [('associations', decisions), ('observations', observations)]:
        (R / 'association' / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    geometry_report = dict(status='PASS', scene='room0', frame_count=20, frame_ids=list(FIDS), last_frame_id=95,
        surface_points=timeline[-1]['surface_points'], surface_sha256=sha(R / 'geometry/surface.ply'), voxel_size_m=.01,
        no_future_geometry=True, source_config=str(C / 'config.json'), config_sha256=sha(C / 'config.json'))
    write(R / 'geometry/geometry_prefix20.json', geometry_report)
    write(R / 'mapping_complete.json', dict(status='PASS', frame_count=20, frame_ids=list(FIDS), last_frame_id=95,
        geometry_incremental=True, association_incremental=True, strict_republication_after_every_frame=True,
        GT_used=False, future_inputs_used=False, future_baseline_used=False, repair_applied=False,
        strict_same_frame_point_at_most_one_vote=True, final=timeline[-1], input_manifest_sha256=sha(R / 'inputs_manifest.json')))

if __name__ == '__main__':
    main()
