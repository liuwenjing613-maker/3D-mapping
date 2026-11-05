"""Add immutable original main final repair seeds to the seed inspection page."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from instance_repair.seed_final_abc import validate_selected
from instance_repair.two_audit_sources import file_record
from instance_repair.two_audit_cache import write_json
ROOT=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104')
OUT=ROOT/'stable_seed_filter_v1/review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return file_record(p)['sha256']
def build():
 dest=OUT/'original_images';dest.mkdir(exist_ok=True)
 rows=[];scenes={};locks={};guards={}
 for scene in ['room0','office2']:
  base=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/main')/scene/'400/A'
  c=base/'candidates';manifest=c/'final_abc/seed_manifest.json';receipt=c/'final_abc/receipt.json'
  guards[scene]=validate_selected(manifest,receipt)
  rec=read(receipt);v=Path(rec['context']['mask_root']);selected=read(manifest)
  semantics=read(rec['context']['semantics'])['records']
  classification=read(c/'v2/semantic/classification.json');rgb_rows={x['frame']:x for x in classification['frames']}
  cfg=read(base/'mapping/config.json');sr=Path(cfg['source']['scene_root'])
  locks[str(manifest)]=sha(manifest);locks[str(receipt)]=sha(receipt)
  for i,g in enumerate(selected['groups']):
   p=g['primary'];mp=v/p['mask'];assert sha(mp)==p['mask_sha256'];locks[str(mp)]=sha(mp)
   mask=np.asarray(Image.open(mp))>0;assert mask.shape==(680,1200) and int(mask.sum())==p['area']
   assert hashlib.sha256(p['frame'].to_bytes(4,'little')+np.packbits(mask.ravel()).tobytes()).hexdigest()==p['digest']
   rgbp=sr/cfg['source']['rgb_pattern'].format(frame=p['frame'])
   rgbhash=sha(rgbp);assert rgbhash==rgb_rows[p['frame']]['RGB_sha256'];locks[str(rgbp)]=rgbhash
   rgb=np.asarray(Image.open(rgbp).convert('RGB')).copy();assert rgb.shape[:2]==mask.shape
   palette=[[30,150,255],[255,135,25],[35,215,145],[210,75,220],[245,210,40],[235,80,110],[70,205,225],[170,195,50]]
   col=np.array(palette[i%len(palette)],float);image=rgb.astype(float)
   image[mask]=.58*image[mask]+.42*col
   inner=mask.copy();inner[1:]&=mask[:-1];inner[:-1]&=mask[1:];inner[:,1:]&=mask[:,:-1];inner[:,:-1]&=mask[:,1:]
   image[mask&~inner]=col
   y,x=np.nonzero(mask);margin=max(35,int(max(x.ptp(),y.ptp())*.15))
   box=[max(0,int(x.min())-margin),max(0,int(y.min())-margin),min(1200,int(x.max())+margin+1),min(680,int(y.max())+margin+1)]
   im=Image.fromarray(image.astype(np.uint8)).crop(box);im.thumbnail((1200,800))
   rel='original_images/'+scene+'_'+p['id']+'.jpg';im.save(OUT/rel,quality=92)
   thumb='original_images/'+scene+'_'+p['id']+'_thumb.jpg';im.thumbnail((300,180));im.save(OUT/thumb,quality=88)
   rows.append(dict(source='original',scene=scene,id=p['id'],group=g['id'],frame=p['frame'],area=p['area'],
       support=p['support_count'],model={0:'CropFormer 整图',1:'CropFormer 裁剪',2:'SAM'}[p['representative']['model']],
       semantic=semantics[p['id']]['top_class'],image=rel,thumb=thumb,auxiliary_ids=g['auxiliary_ids'],
       provenance=dict(manifest=str(manifest),manifest_sha256=locks[str(manifest)],mask=str(mp),mask_sha256=p['mask_sha256'],digest=p['digest'],
                       rgb=str(rgbp),rgb_sha256=rgbhash,stage='final_abc',original_track_id=g.get('original_track_id'))))
  scenes[scene]=len(selected['groups']);print(scene,scenes[scene],flush=True)
 assert scenes==dict(room0=21,office2=13)
 for p,h in locks.items():assert sha(p)==h
 write_json(OUT/'original.json',dict(status='PASS',scenes=scenes,rows=rows,source='original full-depth main final_abc repair seeds'))
 write_json(OUT/'original_validation.json',dict(status='PASS',total=len(rows),scenes=scenes,guards=guards,
     all_mask_pixels_unchanged=True,all_rgb_hashes_verified=True,grouping_performed=False,source_locks=locks))
if __name__=='__main__':build()
