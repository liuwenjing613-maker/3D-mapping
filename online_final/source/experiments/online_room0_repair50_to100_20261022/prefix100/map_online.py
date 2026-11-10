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
    base = max(max(lookup.values()),REPAIR_MAX_ID) + 1
    keys_per_frame, audit = [], []
    for frame in frames:
        if frame.frame_id<=245:
            history,adopted=JOURNAL[frame.frame_id]
            hp,hi,hs=project_frame_regions(replace(frame,mask_local=history),tree,pixel_stride=2,max_distance_m=.015,workers=4)
            ap,ai,astats=project_frame_regions(replace(frame,mask_local=adopted),tree,pixel_stride=2,max_distance_m=.015,workers=4)
            points=np.r_[hp,ap];local=np.r_[hi,ai];table=np.arange(base,dtype=np.int64)
            stats=dict(history_projected_pairs=len(hp),adopted_projected_pairs=len(ap))
        else:
            points,local,stats=project_frame_regions(frame,tree,pixel_stride=2,max_distance_m=.015,workers=4)
            table=np.full(int(frame.mask_local.max())+1,-1,np.int64)
            for lid in np.unique(local):table[lid]=lookup[(frame.frame_id,int(lid))]
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
                source_mask_sha256=np.array([sha(R / f'inputs/masks/frame{frame.frame_id:06d}.png')]),
                candidate_persistent_id=table[local], repaired_source_pixel_journal=np.array([frame.frame_id<=245]))
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
        cloud_ply(R / 'ply/room0_prefix100_instances.ply', xyz, colors)
        remaining = labels <= 0
        state_colors = np.array([[0.,0.,0.],[1.,.8,0.],[.2,.8,.2],[1.,0.,0.]])
        cloud_ply(R / 'ply/room0_prefix100_three_states.ply', xyz[remaining], state_colors[evidence['state'][remaining]])
        write(baseline / 'mapping_summary.json', summary)
    return summary

JOURNAL={}
REPAIR_MAX_ID=0
def main():
    global JOURNAL,REPAIR_MAX_ID
    from PIL import Image
    TOP=R.parent;commit=read(TOP/'committed50.json')
    assert commit['status']=='COMMITTED' and sha(commit['checkpoint'])==commit['checkpoint_sha256']
    assert sha(commit['audit'])==commit['audit_sha256']
    for relative in ['baseline/frame_support','baseline/native','baseline/final','association','geometry','ply']:
        (R/relative).mkdir(parents=True,exist_ok=True)
    import sitecustomize
    sitecustomize.current_cutoff=245
    source=ReplicaFrameSource(C/'config.json');assert tuple(source.frame_ids)==tuple(FIDS)
    associator=OnlineVoxelAssociator.from_checkpoint(commit['checkpoint'])
    assert associator.last_processed_frame_id==245
    associator.identity_evidence.validate();first_version=associator.map_version
    REPAIR_MAX_ID=associator.next_instance_id-1
    geometry=SparseTSDFGeometry(block_count=100000)
    assert sha(commit['geometry'])==commit['geometry_sha256']
    geometry.grid=o3d.t.geometry.VoxelBlockGrid.load(commit['geometry'])
    geometry.integrated_frames=set(range(0,246,5))
    journal=Path(commit['pixel_journal'])
    for fid in range(0,246,5):
        JOURNAL[fid]=(np.asarray(Image.open(journal/f'f{fid:06d}_history.png')),np.asarray(Image.open(journal/f'f{fid:06d}_adopted.png')))
    old=Path('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021')
    decisions=[json.loads(x) for x in (old/'association/associations.jsonl').read_text().splitlines()]
    observations=[json.loads(x) for x in (old/'association/observations.jsonl').read_text().splitlines()]
    lookup={(x['frame_id'],x['mask_local_id']):x['instance_id'] for x in decisions}
    frames=[source.load_frame(fid) for fid in FIDS[:50]]
    timeline=read(old/'mapping_timeline.json')
    start=time.perf_counter();continuation=[]
    for index,fid in enumerate(FIDS[50:],51):
        sitecustomize.current_cutoff=fid
        frame=source.load_frame(fid)
        new_observations=extract_frame_observations(frame,'Replica','room0',sha(R/f'inputs/masks/frame{fid:06d}.png'))
        before_version=associator.map_version
        current=associator.process_frame(frame,new_observations)
        assert len(current)==len(new_observations)
        decisions.extend(current);observations.extend(asdict(o) for o in new_observations)
        for item in current:lookup[(fid,item['mask_local_id'])]=item['instance_id']
        integration=geometry.integrate(frame);frames.append(frame)
        cloud=geometry.extract_point_cloud(weight_threshold=1.).to_legacy()
        if index==100:
            assert o3d.io.write_point_cloud(str(R/'geometry/surface.ply'),cloud)
            geometry.save(R/'geometry/tsdf_grid.npz')
        summary=strict_surface(frames,lookup,cloud,index,index==100)
        summary.update(integration=integration,persistent_instances=len(associator.instances),seconds=time.perf_counter()-start,
            repair50_applied=True,association_map_version_before=before_version,association_map_version_after=associator.map_version,
            history_source='committed pixel journal for raw frames 0..245')
        continuation.append(summary);timeline.append(summary)
        write(R/'mapping_progress.json',summary);write(R/'mapping_timeline.json',timeline);write(R/'continuation_timeline.json',continuation)
        if index==51:
            write(R/'frame51_state_audit.json',dict(status='PASS',frame_id=fid,checkpoint_sha256=commit['checkpoint_sha256'],
                restored_map_version=first_version,map_version_used_before_association=before_version,
                repaired_state_used=(before_version==first_version),old_original_map_version=read(Path(commit['audit']))['old_map_version']))
            assert before_version==first_version
        print('PROCESSED',index,fid,summary['surface_points'],summary['state_counts'],flush=True)
    associator.identity_evidence.validate()
    associator.save_checkpoint(R/'association/observation_support_3cm.npz')
    associator.save_identity_counts(R/'association/identity_counts.npz')
    for name,rows in [('associations',decisions),('observations',observations)]:
        (R/'association'/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    write(R/'geometry/geometry_prefix100.json',dict(status='PASS',scene='room0',frame_count=100,frame_ids=list(FIDS),last_frame_id=495,
        surface_points=continuation[-1]['surface_points'],surface_sha256=sha(R/'geometry/surface.ply'),voxel_size_m=.01,
        no_future_geometry=True,source_config=str(C/'config.json'),config_sha256=sha(C/'config.json')))
    write(R/'mapping_complete.json',dict(status='PASS',frame_count=100,frame_ids=list(FIDS),last_frame_id=495,
        geometry_incremental=True,association_incremental=True,strict_republication_after_every_frame=True,
        GT_used=False,future_inputs_used=False,future_baseline_used=False,repair_applied=True,repair_trigger=50,
        repair50_commit_sha256=sha(TOP/'committed50.json'),repair100_applied=False,stopped_before_second_repair=True,
        strict_same_frame_point_at_most_one_vote=True,final=continuation[-1],input_manifest_sha256=sha(R/'inputs_manifest.json')))
if __name__=='__main__':main()
