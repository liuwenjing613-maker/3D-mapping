"""No-GT vote audit and equal-geometry RGB-D projection of all fixed views."""
from pathlib import Path
from colorsys import hsv_to_rgb
import hashlib, json, os, sys, time, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
BASE = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
INPUT = Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')
SNAPSHOT = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
sys.path.insert(0, str(SNAPSHOT / 'revisable_instance_map/src'))
from revisable_instance_map.frame_io import ReplicaFrameSource


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def palette(n):
    return np.array([[180, 180, 180]] + [np.array(hsv_to_rgb((i * .61803398875) % 1, .70, .98)) * 255 for i in range(1, n + 1)], dtype=np.uint8)


def visible_pixels(xyz, frame):
    camera = frame.camera
    pc = (xyz.astype(np.float64) - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
    front = (pc[:, 2] > .05) & (pc[:, 2] < 10)
    indices = np.flatnonzero(front)
    projected = np.rint(np.column_stack([camera.fx * pc[indices, 0] / pc[indices, 2] + camera.cx,
                                        camera.fy * pc[indices, 1] / pc[indices, 2] + camera.cy])).astype(np.int32)
    inside = (projected[:, 0] >= 0) & (projected[:, 0] < camera.width) & (projected[:, 1] >= 0) & (projected[:, 1] < camera.height)
    indices, projected = indices[inside], projected[inside]
    u, v = projected.T
    depth = frame.depth_m[v, u]
    back = np.column_stack([(u - camera.cx) * depth / camera.fx,
                           (v - camera.cy) * depth / camera.fy, depth])
    gap = np.linalg.norm(back - pc[indices], axis=1)
    valid = (depth > 0) & (depth < 10) & (gap <= .02)
    indices, projected, gap = indices[valid], projected[valid], gap[valid]
    # Selection depends exclusively on fixed geometry/depth, never on labels.
    order = np.argsort(gap, kind='stable')
    keys = projected[order, 1] * camera.width + projected[order, 0]
    unique, first = np.unique(keys, return_index=True)
    return unique, indices[order[first]]


def label_overlay(rgb, pixel, indices, labels, color_table):
    out = rgb.copy().reshape(-1, 3)
    visible_labels = labels[indices]
    selected = visible_labels > 0
    at = pixel[selected]
    out[at] = np.rint(out[at].astype(float) * .43 + color_table[visible_labels[selected]] * .57).astype(np.uint8)
    missing = pixel[~selected]
    out[missing] = np.rint(out[missing].astype(float) * .72 + 170 * .28).astype(np.uint8)
    return out.reshape(rgb.shape)


def delta_overlay(rgb, pixel, indices, baseline, repaired):
    out = np.rint(rgb.astype(float) * .68).astype(np.uint8).reshape(-1, 3)
    changed = baseline[indices] != repaired[indices]
    colors = np.full((len(indices), 3), [255, 173, 40], np.uint8)
    colors[(baseline[indices] <= 0) & (repaired[indices] > 0)] = [36, 235, 163]
    colors[(baseline[indices] > 0) & (repaired[indices] <= 0)] = [255, 70, 85]
    out[pixel[changed]] = colors[changed]
    return out.reshape(rgb.shape)


def object_audit(seed, association, xyz, original, before_evidence):
    uid = seed['case_uid']
    support = load(ROOT / 'repair' / uid / 'seed_projected_support.npz')
    pts, tracks = support['surface_point_index'], support['track_id']
    result = []
    for obj in association['objects']:
        p = np.unique(pts[tracks == obj['track_id']])
        identity = obj['persistent_id']
        row = {**obj, 'seed_points': len(p),
               'baseline_leader_votes_median': float(np.median(before_evidence['top1_votes'][p])) if len(p) else None,
               'baseline_correct_frozen_identity_fraction': float(np.mean(original[p] == identity)) if len(p) else None,
               'conditions': {}}
        for condition in ['seed_only', 'seed_plus_short_track']:
            path = ROOT / 'repair' / uid / condition
            e = load(path / 'native/surface_evidence.npz')
            pair = load(path / 'native/surface_instance_frame_votes.npz')
            final = load(path / 'final/instance_surface.npz')['instance_id']
            selected = (pair['instance_id'] == identity) & np.isin(pair['surface_point_index'], p)
            votes = np.zeros(len(xyz), np.int32)
            votes[pair['surface_point_index'][selected]] = pair['frame_votes'][selected]
            row['conditions'][condition] = {'desired_identity_votes_median': float(np.median(votes[p])) if len(p) else None,
                'leader_votes_median': float(np.median(e['top1_votes'][p])) if len(p) else None,
                'desired_identity_is_top1_fraction': float(np.mean(e['top1_instance_id'][p] == identity)) if len(p) else None,
                'desired_identity_is_final_fraction': float(np.mean(final[p] == identity)) if len(p) else None,
                'published_surface_fraction': float(np.mean(final[p] > 0)) if len(p) else None}
        result.append(row)
    return result


def main():
    out = ROOT / 'review'
    (out / 'assets').mkdir(parents=True, exist_ok=True)
    while not (ROOT / 'repair/status.json').exists() or json.loads((ROOT / 'repair/status.json').read_text())['status'] != 'PASS':
        if (ROOT / 'repair/failure.json').exists():
            raise RuntimeError('Repair failed')
        time.sleep(10)
    human = json.loads((ROOT / 'human_seeds_20ad6139511b/seed_manifest.json').read_text())
    seeds = {s['case_uid']: s for s in human['seeds']}
    cases = json.loads((Path(__file__).parent / 'review_manifest.json').read_text())['cases']
    rows = []
    for case in cases:
        uid, scene = case['case_uid'], case['scene']
        seed = seeds[uid]
        source = ReplicaFrameSource(INPUT / scene / 'configs/raw.json')
        baseline = load(BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz')
        xyz, original = baseline['xyz_m'], baseline['instance_id']
        maps = {'baseline': original}
        association, audits = None, []
        if seed['status'] == 'READY':
            association = json.loads((ROOT / 'repair' / uid / 'frozen_association.json').read_text())
            for condition in ['seed_only', 'seed_plus_short_track']:
                m = load(ROOT / 'repair' / uid / condition / 'final/instance_surface.npz')
                np.testing.assert_array_equal(xyz, m['xyz_m'])
                maps[condition] = m['instance_id']
            evidence = load(BASE / f'native/{scene}/P1-A1/surface_p0/surface_evidence.npz')
            audits = object_audit(seed, association, xyz, original, evidence)
        else:
            maps['seed_only'] = maps['seed_plus_short_track'] = original
        largest = max(int(labels.max()) for labels in maps.values())
        if association:
            largest = max(largest, max(obj['persistent_id'] for obj in association['objects']))
        table = palette(largest)
        target_colors = palette(len(seed.get('objects', [])))
        if association:
            for obj in reversed(association['objects']):
                table[obj['persistent_id']] = target_colors[obj['track_id']]
        views = []
        for view in case['views']:
            frame = source.load_frame(view['frame'])
            box = tuple(view['box'])
            pixel, indices = visible_pixels(xyz, frame)
            assets = {}
            prefix = uid + '_v' + str(view['number'])
            for condition, labels in maps.items():
                filename = prefix + '_' + condition + '.jpg'
                Image.fromarray(label_overlay(frame.rgb, pixel, indices, labels, table)).crop(box).save(out / 'assets' / filename, quality=91)
                assets[condition] = 'assets/' + filename
                if condition != 'baseline':
                    filename = prefix + '_' + condition + '_changes.jpg'
                    Image.fromarray(delta_overlay(frame.rgb, pixel, indices, original, labels)).crop(box).save(out / 'assets' / filename, quality=91)
                    assets[condition + '_changes'] = 'assets/' + filename
            filename = prefix + '_rgb.jpg'
            Image.fromarray(frame.rgb).crop(box).save(out / 'assets' / filename, quality=91)
            assets['RGB'] = 'assets/' + filename
            if seed['status'] == 'READY' and view['frame'] == seed['source_choice']['frame']:
                mask = np.array(Image.open(ROOT / 'human_seeds_20ad6139511b' / seed['seed_file']), dtype=np.uint16)
                rgb = frame.rgb.copy()
                selected = mask > 0
                rgb[selected] = np.rint(rgb[selected].astype(float) * .45 + target_colors[mask[selected]] * .55).astype(np.uint8)
                filename = prefix + '_human_seed.jpg'
                im = Image.fromarray(rgb).crop(box)
                draw = ImageDraw.Draw(im)
                font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
                font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
                for obj in seed['objects']:
                    y, x = np.nonzero(mask == obj['track_id'])
                    if len(x):
                        center = (int(np.median(x)) - box[0], int(np.median(y)) - box[1])
                        draw.text(center, str(obj['native_mask_id']), font=font, fill='white', stroke_width=2, stroke_fill='black')
                im.save(out / 'assets' / filename, quality=91)
                assets['human_seed'] = 'assets/' + filename
            views.append({**view, 'assets': assets, 'depth_visible_pixels': len(pixel)})
        rows.append({**case, 'views': views, 'objects': association['objects'] if association else [],
                     'object_colors': [c.tolist() for c in target_colors], 'vote_audit': audits})
        dump(out / 'progress.json', {'completed_cases': len(rows), 'total_cases': 10, 'last_case': uid})
        print('rendered', uid, flush=True)
    report = {'status': 'PASS', 'GT_used': False, 'cases': rows,
              'rendering': {'geometry_and_visibility_selection_identical_in_all_conditions': True,
                            'depth_visibility_tolerance_m': .02, 'uses_fixed_original_views_and_boxes': True},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(out / 'review_data.json', report)
    files = sorted(str(p) for p in (out / 'assets').glob('*.jpg'))
    dump(out / 'files.json', files)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(ROOT / 'review/failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise
