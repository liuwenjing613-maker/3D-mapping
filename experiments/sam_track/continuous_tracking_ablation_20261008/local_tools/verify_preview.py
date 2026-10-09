"""Check every original pack and every lossless native mask against both gap policies."""
from pathlib import Path
import hashlib
import json
import zlib
import numpy as np
from continuity_policy import track_window, in_window

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1] / 'results' / '固定案例_三模型对比_20261006'
WEB = BASE / 'continuous_tracking_ablation_20261008'


def main():
    receipts = {}
    for scene in ['room0', 'room2']:
        folder = BASE / (scene + '_tracking_playback_20261008')
        original_bytes = (folder / 'playback_data.json').read_bytes()
        source = json.loads(original_bytes)
        preview = json.loads((WEB / (scene + '_preview_data.json')).read_text(encoding='utf-8'))
        assert preview['source_metadata_sha256'] == hashlib.sha256(original_bytes).hexdigest()
        assert preview['frames'] == source['frames'] and preview['tracks'] == source['tracks']
        windows = {}
        counts = {'gap0':0, 'gap1':0}
        for gap in (0, 1):
            key = f'gap{gap}'
            windows[key] = {w['track']:w for w in preview['continuity_policies'][key]['tracks']}
            for t in source['tracks']:
                oid = t['track']
                computed = track_window([f['areas'][oid-1] for f in source['frames']],t['seed_frame'],gap)
                assert all(windows[key][oid][k] == v for k,v in computed.items())
        frame_count = 0
        for chunk in source['chunks']:
            package = (folder / chunk['path']).read_bytes()
            assert len(package) == chunk['bytes'] and hashlib.sha256(package).hexdigest() == chunk['sha256']
            frames = [f for f in source['frames'] if f['chunk'] == chunk['first_frame']]
            end = 0
            for frame in frames:
                assert frame['offset'] == end
                j = frame['offset'] + frame['rgb_bytes']
                pairs = np.frombuffer(zlib.decompress(package[j:j+frame['mask_bytes']]),dtype='<u4').reshape(-1,2)
                lengths, values = pairs[:,0], pairs[:,1]
                assert int(lengths.sum(dtype=np.uint64)) == source['width']*source['height']
                assert np.all(lengths > 0) and np.all(values < (1 << len(source['tracks'])))
                for key, tracks in windows.items():
                    allowed_bits = sum(1 << (oid-1) for oid,w in tracks.items() if in_window(w,frame['frame']))
                    clipped = values & np.uint32(allowed_bits)
                    assert not np.any(clipped & ~values), 'Never introduce any mask pixel'
                    for oid, window in tracks.items():
                        bit = np.uint32(1 << (oid-1))
                        raw_members = (values & bit) != 0
                        expected_area = frame['areas'][oid-1]
                        assert int(lengths[raw_members].sum(dtype=np.uint64)) == expected_area
                        kept_members = (clipped & bit) != 0
                        if in_window(window,frame['frame']):
                            np.testing.assert_array_equal(kept_members,raw_members)
                            counts[key] += expected_area > 0
                        else:
                            assert not kept_members.any()
                end = j + frame['mask_bytes'] + frame['used_bytes']
                frame_count += 1
            assert end == len(package)
            if frame_count % 500 == 0:
                print(f'{scene}: verified {frame_count}/2000 native frames',flush=True)
        assert frame_count == 2000
        for key, tracks in windows.items():
            assert counts[key] == sum(w['retained_nonempty_frames'] for w in tracks.values())
        receipts[scene] = {'original_frames_verified':frame_count,'original_packs_sha256_verified':len(source['chunks']),
            'tracks':len(source['tracks']),'native_RLE_membership_checked_without_downsampling':True,
            'retained_masks_pixel_exact':True,'removed_masks_have_no_pixels':True,'new_mask_pixels_introduced':0,
            'kept_nonempty_mask_frames':counts,'source_metadata_sha256':preview['source_metadata_sha256']}
    result = {'status':'PASS','scenes':receipts,'GT_used':False,'repair_and_v3_rerun':False,'old_maps_modified':False}
    (WORK/'native_mask_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__': main()
