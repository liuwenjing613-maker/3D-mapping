"""Export archived SAM outputs and the exact input RGB, without inference or GT."""
from pathlib import Path
import hashlib, json, shutil, tarfile
import numpy as np
from PIL import Image

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
OUT = ROOT / 'tracking_review'
SEEDS = ROOT / 'human_seeds_20ad6139511b'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((SEEDS / 'seed_manifest.json').read_text())
    cases, files = [], {}
    for seed in manifest['seeds']:
        case = {k: seed[k] for k in ['case_uid', 'scene', 'ROI', 'status']}
        case['source_choice'] = seed.get('source_choice', {})
        if seed['status'] != 'READY':
            case['frames'], case['objects'] = [], []
            cases.append(case)
            continue
        uid = seed['case_uid']
        complete_file = ROOT / 'tracking' / uid / 'complete.json'
        complete = json.loads(complete_file.read_text())
        assert complete['status'] == 'PASS' and not complete['ground_truth_used']
        dest = OUT / 'assets' / uid
        dest.mkdir(parents=True, exist_ok=True)
        source_rgb = {r['frame']: r for r in complete['input_rgb']}
        rows, object_stats = [], {o['track_id']: {'areas': [], 'boxes': [], 'centers': []} for o in seed['objects']}
        for row in complete['frames']:
            fid = row['frame']
            rgb = Path(source_rgb[fid]['path'])
            labels_path = Path(row['label_file'])
            assert sha(rgb) == source_rgb[fid]['sha256']
            assert sha(labels_path) == row['label_sha256']
            labels = np.array(Image.open(labels_path), dtype=np.uint16)
            assert labels.shape == (680, 1200)
            assert set(np.unique(labels)).issubset({0, *object_stats.keys()})
            with Image.open(rgb) as im:
                assert im.size == (1200, 680)
            rgb_dest, ids_dest = dest / f'f{fid:06d}_rgb.jpg', dest / f'f{fid:06d}_ids.png'
            # RGB stays byte-identical. Labels are encoded as RGB24 so that the
            # browser reads exact track IDs through canvas, without quantization.
            shutil.copyfile(rgb, rgb_dest)
            packed = np.stack([labels & 255, labels >> 8, np.zeros_like(labels)], axis=2).astype(np.uint8)
            Image.fromarray(packed).save(ids_dest)
            restored = np.array(Image.open(ids_dest)).astype(np.uint32)
            restored = restored[:,:,0] | (restored[:,:,1] << 8) | (restored[:,:,2] << 16)
            np.testing.assert_array_equal(restored, labels)
            boxes, centers = {}, {}
            for oid, stats in object_stats.items():
                y, x = np.nonzero(labels == oid)
                area = len(x)
                assert area == row['area_pixels'][str(oid)]
                box = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if area else None
                center = [float(np.median(x)), float(np.median(y))] if area else None
                stats['areas'].append(area); stats['boxes'].append(box); stats['centers'].append(center)
                boxes[str(oid)], centers[str(oid)] = box, center
            if fid == seed['source_choice']['frame']:
                np.testing.assert_array_equal(labels, np.array(Image.open(SEEDS / seed['seed_file'])))
            rgb_rel, ids_rel = str(rgb_dest.relative_to(OUT)), str(ids_dest.relative_to(OUT))
            files[rgb_rel], files[ids_rel] = sha(rgb_dest), sha(ids_dest)
            rows.append({'frame': fid, 'offset': fid-seed['source_choice']['frame'], 'direction': row['direction'],
                         'seed': row['seed_restored_exactly'], 'mapping': row['eligible_for_mapping_votes'],
                         'RGB': rgb_rel, 'IDs': ids_rel, 'area_pixels': row['area_pixels'],
                         'boxes': boxes, 'centers': centers, 'missing_track_ids': row['missing_track_ids'],
                         'overlap_pixels_before_partition': row['overlap_pixels_before_partition'],
                         'source_RGB_sha256': source_rgb[fid]['sha256'], 'source_labels_sha256': row['label_sha256']})
        objects = []
        for obj in seed['objects']:
            values = object_stats[obj['track_id']]['areas']
            objects.append({**obj, 'present_frames': sum(v > 0 for v in values),
                            'missing_frames': [r['frame'] for r, v in zip(rows, values) if not v],
                            'min_area_pixels': min(values), 'max_area_pixels': max(values)})
        case.update(objects=objects, frames=rows, source_complete_sha256=sha(complete_file),
                    seed_frame=seed['source_choice']['frame'], raw_frame_count=len(rows),
                    mapping_frame_count=len(seed['mapping_vote_frames']), seed_preserved_exactly=True)
        cases.append(case)
        print(json.dumps({'case': uid, 'frames': len(rows), 'objects': len(objects)}, ensure_ascii=False), flush=True)
    data = {'status': 'PASS', 'cases': cases, 'source_seed_manifest_sha256': sha(SEEDS / 'seed_manifest.json'),
            'ground_truth_used': False, 'new_inference_run': False, 'native_resolution': [1200,680],
            'RGB_export': 'byte-identical original input', 'label_export': 'exact RGB24 track-ID roundtrip',
            'source_tracking_code_sha256': json.loads((ROOT / 'tracking/status.json').read_text())['code_sha256'],
            'total_frames': sum(len(c['frames']) for c in cases),
            'mapping_frames': sum(c.get('mapping_frame_count',0) for c in cases),
            'objects': sum(len(c['objects']) for c in cases), 'asset_hashes': files}
    assert data['total_frames'] == 353 and data['objects'] == 55 and data['mapping_frames'] == 77
    (OUT / 'tracking_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    archive = ROOT / 'tracking_review_bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(OUT / 'assets', arcname='assets')
        tar.add(OUT / 'tracking_data.json', arcname='tracking_data.json')
    print(json.dumps({'status': 'PASS', 'total_frames': data['total_frames'], 'mapping_frames': data['mapping_frames'],
                      'bundle_bytes': archive.stat().st_size, 'bundle_sha256': sha(archive)}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
