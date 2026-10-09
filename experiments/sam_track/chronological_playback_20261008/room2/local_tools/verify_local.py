"""Verify every native-resolution frame before publishing the room2 player."""
from pathlib import Path
import hashlib
import json
import shutil
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
WEB = WORK / 'results/固定案例_三模型对比_20261006/room2_tracking_playback_20261008'
SOURCE = WORK / 'outputs/room2_top30_repair_20261008'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


receipt = json.loads((HERE / 'complete.json').read_text(encoding='utf-8-sig'))
assert receipt['status'] == 'PASS' and receipt['frames'] == 2000 and receipt['tracks'] == 25
assert receipt['verified_tracking_PNGs'] == 28000
assert sha(HERE / 'playback_data.json') == receipt['data_sha256']
data = json.loads((HERE / 'playback_data.json').read_text(encoding='utf-8-sig'))
assert data['scene'] == 'room2' and data['all_tracking_masks'] == 25 and data['unique_repair_identities'] == 23
assert data['native_masks_lossless'] and data['raw_rgb_original_bytes']
assert not data['GT_used'] and not data['tracking_rerun'] and not data['repair_modified']
assert [frame['frame'] for frame in data['frames']] == list(range(2000))
assert [track['track'] for track in data['tracks']] == list(range(1, 26))
assert len({track['persistent_id'] for track in data['tracks']}) == 23
freeze = json.loads((SOURCE / 'input_freeze.json').read_text(encoding='utf-8-sig'))
assert data['annotation_sha256'] == freeze['annotation_sha256']
rgb_hashes = {int(Path(path).stem[-6:]): digest for path, digest in freeze['raw_rgb_hashes'].items()}
assert sorted(rgb_hashes) == list(range(2000))
verified = 0
for chunk in data['chunks']:
    path = WEB / chunk['path']
    assert path.resolve().is_relative_to(WEB.resolve())
    assert path.stat().st_size == chunk['bytes'] and sha(path) == chunk['sha256']
    with path.open('rb') as stream:
        for frame in data['frames'][chunk['first_frame']:chunk['first_frame'] + data['chunk_frames']]:
            assert stream.tell() == frame['offset']
            rgb = stream.read(frame['rgb_bytes'])
            assert hashlib.sha256(rgb).hexdigest() == rgb_hashes[frame['frame']]
            masks = []
            for field, expected in [('mask_bytes', 'areas'), ('used_bytes', 'used_areas')]:
                encoded = stream.read(frame[field])
                pairs = np.frombuffer(zlib.decompress(encoded), dtype='<u4').reshape(-1, 2)
                assert np.all(pairs[:, 0] > 0)
                assert int(pairs[:, 0].astype(np.uint64).sum()) == data['width'] * data['height']
                assert not np.any(pairs[:, 1] >> 25)
                areas = [int(pairs[:, 0][(pairs[:, 1] & (1 << (oid - 1))) != 0].sum(dtype=np.uint64)) for oid in range(1, 26)]
                assert areas == frame[expected], (frame['frame'], expected)
                masks.append(pairs)
            assert frame['mapping_frame'] == (frame['frame'] % 5 == 0)
            assert all(frame['used_areas'][oid - 1] == 0 for oid in range(1, 26) if oid not in frame['accepted'])
            assert frame['mapping_frame'] or (not frame['accepted'] and not any(frame['used_areas']))
            verified += 1
        assert stream.tell() == chunk['bytes'], 'Unused or missing bytes in frame pack'
    if verified % 500 == 0:
        print(f'Verified {verified}/2000 frames', flush=True)
assert verified == 2000 and len(data['chunks']) == 80
final_map = SOURCE / 'validated/full_track/final/instance_surface.npz'
assert sha(final_map) == data['source_hashes'][next(path for path in data['source_hashes'] if path.endswith('validated/full_track/final/instance_surface.npz'))]
shutil.copyfile(HERE / 'playback_data.json', WEB / 'playback_data.json')
shutil.copyfile(HERE / 'complete.json', WEB / 'source_receipt.json')
validation = {'status': 'PASS', 'verified_frames': verified, 'verified_packs': 80, 'verified_RGB_sha256': 2000,
              'verified_raw_and_used_mask_area_vectors': 4000, 'native_masks_and_overlap_membership_lossless': True,
              'tracking_repair_map_and_v3_modified': False, 'data_sha256': receipt['data_sha256'],
              'map_sha256': sha(final_map), 'URL': 'http://127.0.0.1:8785/room2_tracking_playback_20261008/index.html'}
(HERE / 'local_validation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
print(json.dumps(validation))
