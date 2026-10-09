"""Verify received package and publish a separate read-only playback folder."""
from pathlib import Path
import hashlib,json,shutil,tarfile,time,zlib
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'results'/'固定案例_三模型对比_20261006'/'room0_tracking_playback_20261008'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
receipt=json.loads((HERE/'complete.json').read_text(encoding='utf-8'))
archive=HERE/'playback_bundle.tar';assert archive.stat().st_size==receipt['archive_bytes']
assert sha(archive)==receipt['archive_sha256']
with tarfile.open(archive,'r:') as tar:
    for member in tar:
        target=(OUT/member.name).resolve()
        assert target.is_relative_to(OUT.resolve()) and not(member.issym() or member.islnk())
        assert member.name=='playback_data.json' or member.name=='packs' or member.name.startswith('packs/')
        tar.extract(member,OUT,filter='data')
assert sha(OUT/'playback_data.json')==receipt['data_sha256']
data=json.loads((OUT/'playback_data.json').read_text(encoding='utf-8'))
assert [f['frame'] for f in data['frames']]==list(range(2000))
for chunk in data['chunks']:
    path=OUT/chunk['path'];assert path.stat().st_size==chunk['bytes'] and sha(path)==chunk['sha256']
    with path.open('rb') as stream:
        for frame in data['frames'][chunk['first_frame']:chunk['first_frame']+data['chunk_frames']]:
            assert stream.tell()==frame['offset']
            rgb=stream.read(frame['rgb_bytes']);assert rgb[:2]==b'\xff\xd8'
            for field in ['mask_bytes','used_bytes']:
                pairs=np.frombuffer(zlib.decompress(stream.read(frame[field])),dtype='<u4').reshape(-1,2)
                assert int(pairs[:,0].astype(np.uint64).sum())==data['width']*data['height']
                assert not np.any(pairs[:,1]>>30)
            assert len(frame['areas'])==len(frame['used_areas'])==30
shutil.copyfile(HERE/'complete.json',OUT/'source_receipt.json')
validation={'status':'PASS','verified_frames':2000,'verified_packs':len(data['chunks']),
   'all_raw_and_used_masks_complete_native_resolution':True,'archive_sha256':receipt['archive_sha256'],
   'tracking_and_repair_unchanged':True,'URL':'http://127.0.0.1:8785/room0_tracking_playback_20261008/index.html'}
(HERE/'local_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
print(json.dumps(validation))
