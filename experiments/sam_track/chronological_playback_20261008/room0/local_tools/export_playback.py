"""Package exact frozen RGB and all 30 raw tracking masks for chronological replay."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, os, struct, tarfile, time, zlib
import numpy as np
from PIL import Image

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d')
OUT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_tracking_playback_20261008')
PACK=OUT/'packs'
CHUNK=25

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n');os.replace(tmp,path)

def encode(bits):
    flat=bits.ravel();start=np.r_[0,np.flatnonzero(flat[1:]!=flat[:-1])+1]
    pairs=np.empty((len(start),2),dtype='<u4')
    pairs[:,0]=np.diff(np.r_[start,len(flat)]);pairs[:,1]=flat[start]
    # Verify lossless native membership before compression, including overlaps.
    np.testing.assert_array_equal(np.repeat(pairs[:,1],pairs[:,0]),flat)
    payload=zlib.compress(pairs.tobytes(),level=5)
    assert zlib.decompress(payload)==pairs.tobytes()
    return payload

def main():
    started=time.monotonic();PACK.mkdir(exist_ok=True)
    assert not (OUT/'complete.json').exists(),'Inspect complete package before overwriting'
    freeze=json.loads((ROOT/'input_freeze.json').read_text())
    seeds=json.loads((ROOT/'seed_manifest.json').read_text())['seeds']
    assoc=json.loads((ROOT/'global_association.json').read_text())
    canonical={oid:v['canonical_track_id'] for v in assoc['objects'] for oid in v['member_track_ids']}
    persistent={oid:v['persistent_id'] for v in assoc['objects'] for oid in v['member_track_ids']}
    colors=json.loads((ROOT/'review/instance_colors.json').read_text())['id_colors']
    timeline=json.loads((ROOT/'validated/full_track/frame_decisions.json').read_text())
    decisions={v['frame']:v for v in timeline['frames']}
    reports=[];tracks=[];report_hashes={}
    for row in seeds:
        if row['status']!='READY':continue
        uid=row['case_uid'];path=ROOT/'cases'/uid/'tracking/complete.json'
        rep=json.loads(path.read_text());assert rep['status']=='PASS' and rep['completed_frames']==2000
        assert rep['directions_use_independent_seed_only_states'] and not rep['GT_used']
        records={v['frame']:v for v in rep['frames']};assert sorted(records)==list(range(2000))
        reports.append((row,records));report_hashes[str(path)]=sha(path)
        for obj in row['objects']:
            oid=obj['track_id'];tracks.append({'track':oid,'ROI':row['source_choice']['ROI'],'case_uid':uid,
                 'mask':obj['native_mask_id'],'seed_frame':row['source_choice']['frame'],
                 'canonical':canonical[oid],'persistent_id':persistent[oid],
                 'color':colors[str(persistent[oid])],'source_choice':row['source_choice']['choice']})
    tracks.sort(key=lambda v:v['track']);assert [v['track'] for v in tracks]==list(range(1,31))
    protected={str(ROOT/'input_freeze.json'):sha(ROOT/'input_freeze.json'),str(ROOT/'global_association.json'):sha(ROOT/'global_association.json'),
               str(ROOT/'validated/full_track/frame_decisions.json'):sha(ROOT/'validated/full_track/frame_decisions.json'),
               str(ROOT/'validated/full_track/final/instance_surface.npz'):sha(ROOT/'validated/full_track/final/instance_surface.npz'),**report_hashes}
    rgb_sources={int(Path(p).stem[-6:]):(Path(p),digest) for p,digest in freeze['raw_rgb_hashes'].items()}
    assert sorted(rgb_sources)==list(range(2000))
    frames=[];packfiles=[];source_digest=hashlib.sha256();width=1200;height=680
    def read_case(job):
        row,records,fid=job;rec=records[fid];path=Path(rec['label_file'])
        digest=sha(path);assert digest==rec['label_sha256'],('Tracking PNG changed',path)
        with Image.open(path) as im:a=np.array(im,np.uint16)
        assert a.shape==(height,width)
        ids,counts=np.unique(a,return_counts=True);areas=dict(zip(map(int,ids),map(int,counts)))
        allowed={o['track_id'] for o in row['objects']}
        assert set(areas)-{0}<=allowed
        for oid in allowed:assert areas.get(oid,0)==rec['areas'][str(oid)]
        return row,a,areas,digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        for first in range(0,2000,CHUNK):
            path=PACK/('frames_%06d.bin'%first)
            with path.open('wb') as stream:
                for fid in range(first,min(first+CHUNK,2000)):
                    rgbpath,rgbhash=rgb_sources[fid];rgb=rgbpath.read_bytes()
                    assert hashlib.sha256(rgb).hexdigest()==rgbhash,('RGB changed',fid)
                    source_digest.update(bytes.fromhex(rgbhash))
                    raw=np.zeros((height,width),np.uint32);areas=[0]*30
                    decision=decisions.get(fid);checks=decision.get('objects',{}) if decision else {}
                    accepted=decision['accepted_track_ids'] if decision else []
                    initial=[int(oid) for oid,c in checks.items() if c.get('accepted') or c.get('rechecked_after_overlap_abstention')]
                    for row,labels,counts,digest in pool.map(read_case,[(row,recs,fid) for row,recs in reports]):
                        source_digest.update(bytes.fromhex(digest))
                        for obj in row['objects']:
                            oid=obj['track_id'];areas[oid-1]=counts.get(oid,0)
                            raw[labels==oid]|=np.uint32(1<<(oid-1))
                    accepted_bits=sum(1<<(oid-1) for oid in accepted)
                    used=raw & np.uint32(accepted_bits)
                    conflicts=np.zeros((height,width),bool);claims=np.zeros((height,width),np.uint8)
                    for cid in sorted({canonical[oid] for oid in initial}):
                        bits=sum(1<<(oid-1) for oid in initial if canonical[oid]==cid)
                        claims+=(raw & np.uint32(bits))!=0
                    conflicts=claims>1
                    if decision:assert int(conflicts.sum())==decision.get('cross_case_conflict_pixels',0),(fid,'Conflict reconstruction mismatch')
                    used[conflicts]=0
                    # Every accepted pixel must be from an unchanged raw mask and one canonical identity.
                    assert not np.any(used & ~raw)
                    used_areas=[int(np.count_nonzero(used & np.uint32(1<<(oid-1)))) for oid in range(1,31)]
                    raw_bytes=encode(raw);used_bytes=encode(used)
                    offset=stream.tell();stream.write(rgb);stream.write(raw_bytes);stream.write(used_bytes)
                    frames.append({'frame':fid,'chunk':first,'offset':offset,'rgb_bytes':len(rgb),
                       'mask_bytes':len(raw_bytes),'used_bytes':len(used_bytes),'areas':areas,'used_areas':used_areas,
                       'mapping_frame':fid%5==0,'accepted':accepted,
                       'rejections':{oid:c['reasons'] for oid,c in checks.items() if not c['accepted']},
                       'conflict_pixels':int(conflicts.sum())})
            packfiles.append({'first_frame':first,'path':'packs/'+path.name,'sha256':sha(path),'bytes':path.stat().st_size})
            dump(OUT/'status.json',{'status':'RUNNING','completed_frames':len(frames),'total_frames':2000,'seconds':round(time.monotonic()-started,1)})
            if len(frames)%100==0:print(json.dumps({'frames':len(frames),'seconds':round(time.monotonic()-started,1)}),flush=True)
    data={'status':'PASS','scene':'room0','frames_count':2000,'width':width,'height':height,'chunk_frames':CHUNK,
          'native_masks_lossless':True,'raw_rgb_original_bytes':True,'all_tracking_masks':30,'unique_repair_identities':29,
          'annotation_sha256':freeze['annotation_sha256'],'source_sequence_digest':source_digest.hexdigest(),
          'tracks':tracks,'frames':frames,'chunks':packfiles,
          'unreliable_cases':[{'ROI':r['source_choice']['ROI'],'status':r['status']} for r in seeds if r['status']!='READY'],
          'used_masks_definition':'Accepted raw foreground, with initial cross-canonical overlap pixels abstained exactly as run_batch.py; votes also require valid depth and surface projection.',
          'GT_used':False,'tracking_rerun':False,'repair_modified':False,'source_hashes':protected}
    for p,h in protected.items():assert sha(p)==h,('Protected input changed',p)
    dump(OUT/'playback_data.json',data)
    archive=OUT/'playback_bundle.tar'
    with tarfile.open(archive,'w') as tar:
        tar.add(OUT/'playback_data.json',arcname='playback_data.json');tar.add(PACK,arcname='packs')
    record={'status':'PASS','frames':2000,'tracks':30,'verified_tracking_PNGs':24000,
            'all_mask_RLE_roundtrips_exact':True,'used_mask_conflict_reconstruction_matches_every_mapping_frame':True,
            'RGB_bytes_unchanged':True,'protected_predictions_unchanged':True,'GT_used':False,'tracking_rerun':False,
            'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
            'data_sha256':sha(OUT/'playback_data.json'),'seconds':round(time.monotonic()-started,2)}
    dump(OUT/'complete.json',record);dump(OUT/'status.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':main()
