"""Reproduce mesh/depth support, quantify tolerance sensitivity, prepare GT review evidence."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_object_observed_repair_development import observation_frames
from unified_eval.io import load_gt, sha256_array, sha256_file
from unified_eval.observable_surface import observed_reference_vertices_from_frame
from unified_eval.repair_profile import write_json
from unified_eval.schema import EvaluationError

SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']


def camera_inputs(args, scene):
    config = json.loads((args.input_config_root / (scene + '.json')).read_text())
    camera = config['camera']
    k = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1]])
    root = Path(config['source']['scene_root'])
    poses = np.loadtxt(root / config['source']['trajectory']).reshape(-1, 4, 4)
    return config, k, root, poses


def render_review(gt, raw_ids, frame_id, args, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config, k, root, poses = camera_inputs(args, gt.scene_id)
    selected = np.isin(gt.raw_instance_id, raw_ids)
    xyz = gt.xyz_ref[selected]
    if not len(xyz):
        return
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    margin = np.maximum((upper - lower) * .4, .12)
    context = np.all((gt.xyz_ref >= lower - margin) & (gt.xyz_ref <= upper + margin), axis=1)
    indices = np.flatnonzero(context)
    if len(indices) > 50000:
        indices = indices[::int(np.ceil(len(indices) / 50000))]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(*gt.xyz_ref[indices].T, s=1, c='#aaaaaa', alpha=.3)
    for ordinal, raw in enumerate(raw_ids):
        points = gt.xyz_ref[gt.raw_instance_id == raw]
        ax.scatter(*points.T, s=7 if len(points) < 1000 else 1, label=f'GT {raw}')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('Raw GT vertices and nearby GT context')
    ax.legend()
    image_ax = fig.add_subplot(1, 2, 2)
    if frame_id >= 0:
        image = np.asarray(Image.open(root / config['source']['rgb_pattern'].format(frame=frame_id)))
        camera = (xyz - poses[frame_id, :3, 3]) @ poses[frame_id, :3, :3]
        camera = camera[camera[:, 2] > 0]
        uvw = camera @ k.T
        pixels = uvw[:, :2] / uvw[:, 2:3]
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0])
        pixels = pixels[valid]
        if len(pixels):
            left, top = np.maximum(np.floor(pixels.min(axis=0) - 60), [0, 0]).astype(int)
            right, bottom = np.minimum(np.ceil(pixels.max(axis=0) + 61), [image.shape[1], image.shape[0]]).astype(int)
            image_ax.imshow(image[top:bottom, left:right])
            image_ax.scatter(pixels[:, 0] - left, pixels[:, 1] - top, s=6, c='red', alpha=.6)
        else:
            image_ax.imshow(image)
        image_ax.set_title(f'RGB frame {frame_id}; red = geometric projection')
    image_ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.text(.5, .01, 'Projection alone does not prove visibility. Use the depth/GT-ID support statistics for qualification.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .94))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def geometry_review(gt, objects):
    pending = [row['raw_gt_id'] for row in objects if row['review_required']]
    if not pending:
        return {}
    mesh = PlyData.read(gt.metadata['source_mesh'])
    faces = np.asarray(list(mesh['face'].data['vertex_indices']), dtype=np.int64)
    face_labels = gt.raw_instance_id[faces]
    rows = {}
    for raw in pending:
        xyz = gt.xyz_ref[gt.raw_instance_id == raw]
        pure = np.all(face_labels == raw, axis=1)
        any_face = np.any(face_labels == raw, axis=1)
        triangles = faces[pure]
        if triangles.shape[1] == 4:
            triangles = np.concatenate([triangles[:, [0, 1, 2]], triangles[:, [0, 2, 3]]])
        corners = gt.xyz_ref[triangles]
        area = float(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1).sum() / 2)
        rows[raw] = {'raw_vertex_count': len(xyz), 'bbox_extent_m': (xyz.max(axis=0) - xyz.min(axis=0)).astype(float).tolist(),
            'singular_values_m': np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False).astype(float).tolist(),
            'uniform_label_source_faces': int(pure.sum()), 'mixed_label_source_faces_touching_object': int((any_face & ~pure).sum()),
            'uniform_GT_face_area_m2': area, 'review_decision': 'PENDING_HUMAN_REVIEW_NOT_AUTOMATICALLY_INVALID'}
    return rows


def audit_scene(arguments):
    scene, args = arguments
    gt = load_gt(args.baseline_root / 'mesh' / scene / 'gt.npz')
    scope = json.loads((args.baseline_root / 'mesh' / scene / 'gt_scope.json').read_text())
    out = args.output_root / 'GT_review' / scene
    out.mkdir(parents=True, exist_ok=True)
    meta = gt.metadata['observation_metadata']
    with np.load(meta['support_file'], allow_pickle=False) as stored:
        ambiguity = stored['ambiguity_mask'].copy()
    for name, expected in meta['observer_code_sha256'].items():
        if sha256_file(Path(__file__).parents[1] / 'unified_eval' / name) != expected:
            raise EvaluationError('Observation implementation changed: ' + name)
    evidence = {row['frame_id']: row['source_hashes'] for row in meta['input_evidence']}
    counts = {tolerance: np.zeros(gt.vertex_count, dtype=np.uint32) for tolerance in (.01, .02, .03)}
    maximum = int(gt.raw_instance_id.max()) + 1
    best_visible, best_visible_frame = np.zeros(maximum, dtype=int), np.full(maximum, -1, dtype=int)
    best_projected, best_projected_frame = np.zeros(maximum, dtype=int), np.full(maximum, -1, dtype=int)
    object_frame_count = np.zeros(maximum, dtype=int)
    for frame in observation_frames(args, scene, gt, {}, out / 'progress.json'):
        if frame.source_hashes != evidence[frame.frame_id]:
            raise EvaluationError('Input/mesh bytes changed since development baseline: ' + scene)
        for tolerance in counts:
            mask = observed_reference_vertices_from_frame(gt.xyz_ref, gt.raw_instance_id, frame, tolerance)
            counts[tolerance] += mask
            if tolerance == .02:
                mask = mask & ~ambiguity
                per_object = np.bincount(gt.raw_instance_id[mask], minlength=maximum)
                object_frame_count += per_object > 0
                improved = per_object > best_visible
                best_visible[improved] = per_object[improved]
                best_visible_frame[improved] = frame.frame_id
        camera = (gt.xyz_ref - frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        positive = np.flatnonzero(camera[:, 2] > 0)
        uvw = camera[positive] @ frame.intrinsics.T
        pixels = np.floor(uvw[:, :2] / uvw[:, 2:3] + .5).astype(int)
        height, width = frame.depth_m.shape
        in_image = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        per_object = np.bincount(gt.raw_instance_id[positive[in_image]], minlength=maximum)
        improved = per_object > best_projected
        best_projected[improved] = per_object[improved]
        best_projected_frame[improved] = frame.frame_id
    if not np.array_equal(counts[.02], gt.observation_count):
        raise EvaluationError('Reproduced 2cm observed support differs from the locked baseline: ' + scene)
    geometry = geometry_review(gt, scope['objects'])
    rows, review_ids = [], []
    for obj in scope['objects']:
        if obj['object_status'] != 'TARGET':
            continue
        raw = obj['raw_gt_id']
        selected = gt.raw_instance_id == raw
        values = counts[.02][selected]
        values = np.where(ambiguity[selected], 0, values)
        observed = values[values > 0]
        row = {key: obj[key] for key in ['raw_gt_id', 'semantic_class', 'quality_verified', 'review_required', 'evaluable', 'reason', 'reference_vertex_count']}
        row.update({'trusted_observed_vertices': int(np.count_nonzero(values)), 'distinct_trusted_input_frames': int(object_frame_count[raw]),
            'best_trusted_frame': int(best_visible_frame[raw]), 'best_trusted_frame_vertices': int(best_visible[raw]),
            'best_projection_only_frame': int(best_projected_frame[raw]),
            'max_observation_count': int(values.max(initial=0)), 'median_observation_count_among_observed': float(np.median(observed)) if len(observed) else None,
            'observed_vertices_at_1cm_2cm_3cm': {str(tolerance): int(np.count_nonzero((counts[tolerance] > 0) & selected & ~ambiguity)) for tolerance in counts},
            'vertices_seen_at_least_1_2_3_5_times': {str(number): int(np.count_nonzero(values >= number)) for number in (1, 2, 3, 5)},
            'geometry_review': geometry.get(raw), 'review_status': 'PENDING_HUMAN_REVIEW' if obj['review_required'] else 'SOURCE_VERIFIED_NOT_UNIVERSAL_MANUAL_REVIEW'})
        needs_review = obj['review_required'] or (obj['quality_verified'] and not obj['observable']) or (
            obj['evaluable'] and (len(observed) <= 20 or object_frame_count[raw] <= 2))
        if needs_review:
            review_ids.append(raw)
            frame = best_visible_frame[raw] if best_visible_frame[raw] >= 0 else best_projected_frame[raw]
            image_name = f'GT_{raw}.png'
            render_review(gt, [raw], int(frame), args, out / image_name,
                f'{scene} GT {raw} ({obj["semantic_class"]}); observed={len(observed)}/{len(values)}, trusted frames={object_frame_count[raw]}')
            row['review_image'] = image_name
        rows.append(row)
    totals = []
    qualified_ids = [row['raw_gt_id'] for row in scope['objects'] if row['object_status'] == 'TARGET' and row['quality_verified']]
    for tolerance, count in counts.items():
        per_object = np.bincount(gt.raw_instance_id[(count > 0) & ~ambiguity], minlength=maximum)
        totals.append({'depth_tolerance_m': tolerance, 'evaluable_qualified_GT_count': int(sum(per_object[raw] > 0 for raw in qualified_ids)),
                       'observed_qualified_target_vertices': int(np.count_nonzero((count > 0) & ~ambiguity & np.isin(gt.raw_instance_id, qualified_ids)))})
    result = {'status': 'SECONDARY_OBSERVATION_AUDIT_NO_SCOPE_OR_THRESHOLD_CHANGE', 'scene_id': scene,
        'baseline_2cm_support_reproduced_exactly': True, 'observation_count_sha256': sha256_array(counts[.02]),
        'fixed_frame_count': 400, 'GT_scope_sha256': gt.metadata['gt_scope_sha256'], 'per_target': rows,
        'review_ids': review_ids, 'predeclared_depth_tolerance_sensitivity': totals,
        'no_new_size_or_pixel_threshold_was_introduced': True}
    write_json(out / 'GT_observation_review.json', result)
    np.savez_compressed(out / 'depth_tolerance_counts.npz', count_1cm=counts[.01], count_2cm=counts[.02], count_3cm=counts[.03])
    return {'scene_id': scene, 'review_ids': review_ids, 'depth_tolerance_sensitivity': totals, 'support_reproduced': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, default=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009'))
    parser.add_argument('--output-root', type=Path, default=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_audit_20261009'))
    parser.add_argument('--input-config-root', type=Path, default=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs'))
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    args.visibility = 'mesh'
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for job in as_completed([pool.submit(audit_scene, (scene, args)) for scene in args.scenes]):
            rows.append(job.result())
            write_json(args.output_root / 'observation_audit_progress.json', {'completed': rows, 'scenes': args.scenes})
    write_json(args.output_root / 'observation_audit_complete.json', {'status': 'COMPLETE_SECONDARY_AUDIT', 'per_scene': rows})


if __name__ == '__main__':
    main()
