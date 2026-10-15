"""Lossless track-ID previews for every mapping frame, after each track freezes."""
from pathlib import Path
import json,shutil,hashlib,time
import numpy as np
from PIL import Image
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015');O=R/'review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 O.mkdir(exist_ok=True);(O/'rgb').mkdir(exist_ok=True)
 config=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json').read_text());src=config['source']
 freeze=json.loads((R/'input_freeze.json').read_text())
 for fid in range(0,2000,5):
  p=Path(src['scene_root'])/src['rgb_pattern'].format(frame=fid);assert sha(p)==freeze['raw_rgb_hashes'][str(p)]
  out=O/'rgb'/f'f{fid:06d}.jpg'
  if not out.exists():shutil.copyfile(p,out)
 cases=json.loads((R/'seed_diagnostics.json').read_text())['cases'];done=set();start=time.time()
 while len(done)<len(cases):
  if (R/'tracking_queue_failure.json').exists():raise RuntimeError('tracking queue failed')
  for case in cases:
   uid=case['case_uid'];track=R/'cases'/uid/'tracking/complete.json'
   if uid in done or not track.exists():continue
   receipt=O/'tracks'/uid/'receipt.json'
   if receipt.exists():
    assert json.loads(receipt.read_text())['source_complete_sha256']==sha(track);done.add(uid);continue
   d=json.loads(track.read_text());assert d['status']=='PASS';folder=receipt.parent;folder.mkdir(parents=True,exist_ok=True)
   n=0
   for row in d['frames']:
    if row['frame']%5:continue
    p=Path(row['label_file']);assert sha(p)==row['label_sha256'];a=np.asarray(Image.open(p));assert a.max()<=90
    out=folder/f'f{row["frame"]:06d}.png';Image.fromarray(a.astype(np.uint8)).save(out)
    np.testing.assert_array_equal(a,np.asarray(Image.open(out)));n+=1
   assert n==400;dump(receipt,{'source_complete_sha256':sha(track),'all_400_track_ID_arrays_exact':True,'frames':n});done.add(uid)
  dump(R/'review_assets_status.json',{'status':'PASS' if len(done)==len(cases) else 'RUNNING','cases':len(done),'total_cases':len(cases),'seconds':time.time()-start})
  if len(done)<len(cases):time.sleep(20)
 print('REVIEW_ASSETS_COMPLETE',len(done),flush=True)
if __name__=='__main__':main()
