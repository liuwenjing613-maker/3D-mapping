"""Display actual cross-seed tracking failures against human seeds; no map writes."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image,ImageDraw
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007');sys.path.insert(0,str(LEGACY))
import run_repairs as r
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert json.loads((ROOT/'tracking_queue_complete.json').read_text())['status']=='PASS'
 out=ROOT/'cross_seed_review';out.mkdir(exist_ok=True)
 rows=json.loads((ROOT/'seed_manifest.json').read_text())['seeds']
 cases={o['track_id']:row for row in rows for o in row.get('objects',[])}
 source=r.ReplicaFrameSource(r.INPUT/'room2/configs/raw.json');records=[]
 for a,b in [(4,24),(24,4),(5,25),(25,5)]:
  reference=cases[a];fid=reference['source_choice']['frame'];trackedcase=cases[b]
  seedfile=ROOT/reference['seed_file'];trackfile=ROOT/'cases'/trackedcase['case_uid']/'tracking'/('f%06d.png'%fid)
  seed=np.array(Image.open(seedfile))==a;track=np.array(Image.open(trackfile))==b
  rgb=source.load_frame(fid).rgb.copy();original=rgb.copy()
  union=seed|track;paint=np.zeros_like(rgb);paint[seed&~track]=[45,215,135];paint[track&~seed]=[245,80,165];paint[seed&track]=[255,212,58]
  rgb[union]=(rgb[union].astype(float)*.3+paint[union]*.7).astype(np.uint8)
  canvas=Image.new('RGB',(1200,724),'white');canvas.paste(Image.fromarray(rgb),(0,44))
  iou=float(np.sum(seed&track)/max(1,np.sum(seed|track)))
  ImageDraw.Draw(canvas).text((12,12),f'Human track{a} / SAM track{b} at f{fid}: IoU {100*iou:.2f}% | green=human, pink=SAM, yellow=overlap',fill='black')
  name=f'human{a}_sam{b}_f{fid:06d}.jpg';canvas.save(out/name,quality=94)
  record={'human_track_id':a,'propagated_track_id':b,'frame':fid,'human_ROI':reference['source_choice']['ROI'],
   'human_mask_id':next(o['native_mask_id'] for o in reference['objects'] if o['track_id']==a),
   'track_ROI':trackedcase['source_choice']['ROI'],'IoU':iou,'human_pixels':int(seed.sum()),'tracked_pixels':int(track.sum()),
   'seed_file':str(seedfile),'seed_sha256':sha(seedfile),'tracking_file':str(trackfile),'tracking_sha256':sha(trackfile),
   'asset':name,'asset_sha256':sha(out/name)}
  records.append(record)
 report={'status':'PASS','GT_used':False,'map_or_tracking_changed':False,'code_sha256':sha(__file__),'comparisons':records}
 (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
if __name__=='__main__':main()
