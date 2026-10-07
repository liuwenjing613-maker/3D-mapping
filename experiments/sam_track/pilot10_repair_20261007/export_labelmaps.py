"""Lossless projected ID maps so users can toggle individual 3D masks."""
from pathlib import Path
import json, time, traceback
import numpy as np
from PIL import Image
import audit_and_render as render


def main():
    r = render
    root = r.ROOT
    while not (root / 'review/review_data.json').exists():
        if (root / 'review/failure.json').exists():
            raise RuntimeError('Review rendering failed')
        time.sleep(10)
    review = json.loads((root / 'review/review_data.json').read_text())
    records = []
    for case in review['cases']:
        uid, scene = case['case_uid'], case['scene']
        data = r.load(r.BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz')
        xyz = data['xyz_m']
        maps = {'baseline': data['instance_id']}
        for condition in ['seed_only', 'seed_plus_short_track']:
            maps[condition] = data['instance_id'] if case['status'] != 'READY' else r.load(root / 'repair' / uid / condition / 'final/instance_surface.npz')['instance_id']
        source = r.ReplicaFrameSource(r.INPUT / scene / 'configs/raw.json')
        views = []
        for view in case['views']:
            frame = source.load_frame(view['frame'])
            pixels, indices = r.visible_pixels(xyz, frame)
            assets = {}
            for condition, labels in maps.items():
                code = np.zeros(frame.depth_m.shape, np.uint32).reshape(-1)
                code[pixels] = np.where(labels[indices] > 0, labels[indices] + 1, 1).astype(np.uint32)
                code = code.reshape(frame.depth_m.shape)
                channels = np.stack([code & 255, (code >> 8) & 255, (code >> 16) & 255], axis=-1).astype(np.uint8)
                filename = uid + '_v' + str(view['number']) + '_' + condition + '_ids.png'
                target = root / 'review/assets' / filename
                Image.fromarray(channels).crop(tuple(view['box'])).save(target)
                recovered = np.array(Image.open(target), np.uint32)
                recovered = recovered[:, :, 0] | (recovered[:, :, 1] << 8) | (recovered[:, :, 2] << 16)
                x0, y0, x1, y1 = view['box']
                np.testing.assert_array_equal(recovered, code[y0:y1, x0:x1])
                assets[condition] = 'assets/' + filename
            views.append({'number': view['number'], 'frame': view['frame'], 'ID_assets': assets})
        records.append({'case_uid': uid, 'views': views})
        r.dump(root / 'review/id_map_progress.json', {'completed_cases': len(records), 'total_cases': 10})
    r.dump(root / 'review/id_maps.json', {'status': 'PASS', 'GT_used': False, 'cases': records,
        'encoding': 'RGB24; 0=no visible TSDF, 1=unpublished, positive instance ID+1',
        'exact_decode_roundtrip': True, 'code_sha256': r.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        render.dump(render.ROOT / 'review/id_map_failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise
