"""Publish final-only seed masks for every completed Replica scene."""
from pathlib import Path
import json,hashlib,colorsys,importlib.util,shutil,subprocess,sys
import numpy as np
from PIL import Image
from scripts.two_audit_seed_filter import ROOT,OUT,read,sha,write
from instance_repair.two_audit_sources import verify_record
from instance_repair.two_audit_cache import ObservationCache
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
REVIEW=OUT/'review/replica8'
REVIEW.mkdir(exist_ok=True)
def build():
 oldscript=Path('scripts/build_two_audit_original_seeds.py').read_text().replace("OUT=ROOT/'stable_seed_filter_v1/review'","OUT=ROOT/'stable_seed_filter_v1/review/replica8'").replace("for scene in ['room0','office2']:","for scene in "+repr(SCENES)+":").replace(" assert scenes==dict(room0=21,office2=13)"," assert len(scenes)==8")
 if not (REVIEW/'original.json').exists():exec(compile(oldscript,'original_export','exec'),dict(__name__='__main__'))
 scenes={};rows=[];status={};(REVIEW/'current_images').mkdir(exist_ok=True)
 for scene in SCENES:
  o=OUT/scene
  if not (o/'complete.json').exists():
   log=ROOT/'run_logs/replica8'/(scene+'_filter.log')
   status[scene]='待确认：链式包含' if log.exists() and 'Chained qualifying containment needs user review' in log.read_text() else '处理中'
   continue
  assert read(o/'complete.json')['status']=='PASS'
  cache=ObservationCache(ROOT/'observations'/scene)
  cfg=read(Path(cache.manifest['mapping_root'])/'config.json')['source']
  metadata={x['frame']:x for x in (json.loads(line) for line in (Path(cfg['mask_root'])/cfg['mask_metadata']).read_text().splitlines() if line.strip())}
  final=read(o/'final_abc/seed_manifest.json');selected={x['id']:x for x in read(o/'chosen_masks.json')['records']}
  sem=read(o/'abc_semantics/semantics.json')['records']
  for g in final['groups']:
   p=g['primary'];x=selected[p['id']];mp=o/'v1'/p['mask'];assert sha(mp)==p['mask_sha256']
   mask=np.asarray(Image.open(mp))>0
   fr=cache.frames[p['frame']];verify_record(fr['sources']['mask'])
   labels=np.asarray(Image.open(fr['sources']['mask']['path']));assert np.array_equal(mask,labels==x['mask_local_id'])
   rgbp=Path(cfg['scene_root'])/cfg['rgb_pattern'].format(frame=p['frame'])
   assert sha(rgbp)==metadata[p['frame']]['input_sha256']
   rel='current_images/'+scene+'_'+p['id']+'.jpg'
   if not (REVIEW/rel).exists():
    rgb=np.asarray(Image.open(rgbp).convert('RGB')).astype(float)
    col=np.array(colorsys.hsv_to_rgb((int(x['track'][1:])*.61803398875)%1,.72,.96))*255
    rgb[mask]=.58*rgb[mask]+.42*col
    inner=mask.copy();inner[1:]&=mask[:-1];inner[:-1]&=mask[1:];inner[:,1:]&=mask[:,:-1];inner[:,:-1]&=mask[:,1:]
    rgb[mask&~inner]=col
    y,z=np.nonzero(mask);margin=max(35,int(max(np.ptp(y),np.ptp(z))*.15));box=[max(0,int(z.min())-margin),max(0,int(y.min())-margin),min(mask.shape[1],int(z.max())+margin+1),min(mask.shape[0],int(y.max())+margin+1)]
    im=Image.fromarray(rgb.astype(np.uint8)).crop(box);im.thumbnail((1200,800));im.save(REVIEW/rel,quality=92)
   rows.append(dict(scene=scene,id=p['id'],track=x['track'],frame=p['frame'],final=True,image=rel,thumb=rel,semantic=sem[p['id']],support=p['support_count'],mask_sha256=p['mask_sha256']))
  scenes[scene]=dict(final=len(final['groups']));status[scene]='完成'
 write(REVIEW/'data.json',dict(status='PASS',scenes=scenes,rows=rows,scene_status=status))
 code=Path('scripts/build_two_audit_combined_seeds.py').read_text().replace("REVIEW=OUT/'review'","REVIEW=OUT/'review/replica8'").replace("for scene in ['room0','office2']:","for scene in current['scenes']:").replace("OUT/'combined_seed_validation.json'","OUT/'replica8_combined_seed_validation.json'")
 exec(compile(code,'combined_export','exec'),dict(__name__='__main__'))
 h=Path('scripts/two_audit_combined_seeds.html').read_text().replace('<a href="replica8/">八场景最终种子</a>','').replace('<option>room0</option><option>office2</option>',''.join('<option>'+s+'</option>' for s in scenes))
 (REVIEW/'grouped.html').write_text(h)
 shutil.copyfile('scripts/two_audit_replica8_final.html',REVIEW/'index.html')
 print('PUBLISHED',scenes,flush=True)
if __name__=='__main__':build()
