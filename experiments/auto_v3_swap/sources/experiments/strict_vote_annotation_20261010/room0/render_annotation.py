"""Generate complete, checksummed candidates and projected conflict labels."""
from pathlib import Path
import base64, copy, hashlib, json, zipfile
import numpy as np
from PIL import Image
from prepare_inputs import ROOT,SOURCE,NAME,sha,write,read
from render_helpers import assets_for,palette,color_map,static_card,healthy,ASSETS,OUT

def boxes(x,y):return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None

def main():
 manifest=read(ROOT/'input_manifest.json');jobs={j['key']:j for j in manifest['jobs']}
 cf=read(ROOT/'cropformer/audit.json');sam=read(ROOT/'sam2_1/audit.json')
 assert cf['status']==sam['status']=='PASS' and cf['control_min_pixel_agreement']>=.999
 viewer=dict(name=NAME,cases=copy.deepcopy(manifest['cases']),crops={},scene_counts={'room0':30},total_views=90)
 projection=dict(schema='fixed_roi_projection_to_2D_masks_v1',views={},crops={});counts=[];verified_predictions=[]
 for case in viewer['cases']:
  for v in case['views']:
   full=Image.open(SOURCE/v['source_RGB']).convert('RGB');thumb=ASSETS/(case['uid']+f'_v{v["number"]}_full.jpg')
   full.save(thumb,quality=92);v['full_rgb']='assets/'+thumb.name
   a=np.asarray(Image.open(SOURCE/v['source_alarm']).convert('RGB'));rgba=np.zeros((*a.shape[:2],4),np.uint8);codes=np.zeros(a.shape[:2],np.uint8)
   for channel,color,code in [(0,(140,145,155),1),(1,(255,190,30),2),(2,(245,60,70),3)]:
    mask=a[:,:,channel]>0;rgba[mask]=[*color,220];codes[mask]=code
   ap=ASSETS/(case['uid']+f'_v{v["number"]}_alarm.png');Image.fromarray(rgba).save(ap,compress_level=2);v['alarm']='assets/'+ap.name
   state_names={1:'U',2:'T',3:'C'};full_counts={name:int((codes==code).sum()) for code,name in state_names.items()}
   projection['views'][case['uid']+'_v'+str(v['number'])]=dict(alarm_asset=v['alarm'],alarm_sha256=sha(ap),full_image_wh=[1200,680],projected_pixel_counts=full_counts)
   for key in v['variants']:
    if key in viewer['crops']:continue
    j=jobs[key];rgb=np.asarray(Image.open(j['input_file']).convert('RGB'));base=np.asarray(Image.open(j['baseline_file']),np.uint16)
    assert sha(j['input_file'])==j['RGB_sha256'] and sha(j['baseline_file'])==j['baseline_sha256']
    cp=ROOT/'cropformer'/(key+'.png');sp=ROOT/'sam2_1'/(key+'.png');cr=read(cp.with_suffix('.json'));sr=read(sp.with_suffix('.json'))
    assert cr['RGB_sha256']==sr['RGB_sha256']==j['RGB_sha256'];assert sha(cp)==cr['mask_sha256'] and sha(sp)==sr['mask_sha256']
    local=np.asarray(Image.open(cp),np.uint16);slocal=np.asarray(Image.open(sp),np.uint16);assert rgb.shape[:2]==base.shape==local.shape==slocal.shape
    oldcolors={int(i):palette(int(i)) for i in np.unique(base[base>0])};newcolors,cm=color_map(base,local,1000);samcolors,sm=color_map(base,slocal,2000)
    native=ASSETS/(key+'_rgb.webp');Image.fromarray(rgb).save(native,format='WEBP',lossless=True,method=3)
    models=[assets_for(base,oldcolors,key+'_original'),assets_for(local,newcolors,key+'_local'),assets_for(slocal,samcolors,key+'_sam')]
    for m in models:
     m['ids_file']=m['ids'];m['ids']='data:image/png;base64,'+base64.b64encode((OUT/m['ids_file']).read_bytes()).decode('ascii')
    cj=dict(key=key,strategy=j['strategy'],box=j['crop_xyxy'],wh=[rgb.shape[1],rgb.shape[0]],zoom=j['effective_zoom'],
     coverage=j['target_pixel_coverage'],complete=j['all_target_pixels_covered'],RGB='assets/'+native.name,models=models,
     network_input_wh=cr['network_input_wh'],sam_raw_proposals=sr['raw_proposals'],color_correspondence=dict(CropFormer=cm,SAM=sm))
    viewer['crops'][key]=cj
    if key==v['recommended']:
     folder=OUT/'comparisons/room0'/case['ROI'];folder.mkdir(parents=True,exist_ok=True);p=folder/f'view{v["number"]}_frame{v["frame"]:06d}.jpg'
     static_card(rgb,[base,local,slocal],[oldcolors,newcolors,samcolors],j,[m['count'] for m in models]).save(p,quality=94,subsampling=0);cj['comparison']=p.relative_to(OUT).as_posix()
     counts.append(dict(ROI=case['ROI'],view=v['number'],frame=v['frame'],original=models[0]['count'],cropformer_local=models[1]['count'],sam2_1=models[2]['count']))
    x0,y0,x1,y1=j['crop_xyxy'];c=codes[y0:y1,x0:x1];positions={name:np.nonzero(c==code) for code,name in state_names.items()}
    pc={name:int(len(y)) for name,(y,x) in positions.items()}
    pr=dict(group='CONFLICT',crop_xyxy=j['crop_xyxy'],wh=cj['wh'],counts=pc,full_counts=full_counts,state_boxes={n:boxes(x,y) for n,(y,x) in positions.items()},models=[])
    for labels in [base,local,slocal]:
     states={}
     for name,(yy,xx) in positions.items():
      values=labels[yy,xx];rows=[]
      for lid,count in zip(*np.unique(values,return_counts=True)):
       keep=values==lid;x=xx[keep];y=yy[keep];k=int(((x-x.mean())**2+(y-y.mean())**2).argmin())
       assert labels[y[k],x[k]]==lid
       rows.append(dict(id=int(lid),count=int(count),x=int(x[k]),y=int(y[k]),projection_box=boxes(x,y)))
      rows.sort(key=lambda r:(-r['count'],r['id']));assert sum(r['count'] for r in rows)==pc[name];states[name]=rows
     pr['models'].append(states)
    projection['crops'][key]=pr;verified_predictions.append(key)
  print('RENDERED',case['ROI'],len(viewer['crops']),flush=True)
 assert len(viewer['cases'])==30 and viewer['total_views']==sum(len(c['views']) for c in viewer['cases'])==90
 assert set(projection['crops'])==set(viewer['crops']) and len(projection['views'])==90
 write(OUT/'viewer_data.json',viewer);write(OUT/'projection_annotations.json',projection);write(OUT/'mask_counts.json',counts)
 setup=dict(status='READY_FOR_HUMAN_CHOICES',scene_scope='room0',source_selection=NAME,case_count=30,view_count=90,candidate_crop_count=len(viewer['crops']),
  ranking_group='CONFLICT',ranking='strict-vote absolute conflict point count descending; original component ID for ties',
  source_candidate_sha256=manifest['source_candidate_sha256'],source_surface_sha256=manifest['source_surface_sha256'],
  models=['original','cropformer_local','sam2_1_local'],human_records_created=0,tracking_started=False,map_modified=False,no_GT_used=True,
  previous_candidates_used_as_human_choices=False,cache_reused_predictions=len(manifest['cache_reuse']),
  source_data_directory=str(ROOT),isolated_storage_key='cvpr-room0-strict-vote-selection-20261010-v1',export_filename='room0_strict_vote_人工标注_20261010.json',
  old_annotations_unchanged=True,three_models_same_RGB_and_crop=True,all_label_maps_verified=True,
  crop_border_policy='allow selection and record touching IDs; never assert complete beyond crop',
  evaluation_protocol='object_observed_repair / revision 2; candidate selection only; GT not used',policy=manifest['policy'])
 write(OUT/'selection_setup.json',setup)
 write(OUT/'projection_annotation_audit.json',dict(status='PASS',views=90,crops=len(viewer['crops']),models_per_crop=3,all_histograms_and_anchors_verified=True,unit='unique projected image pixels',no_GT_used=True))
 for model in ['cropformer','sam2_1']:
  for n in ['model_provenance.json','audit.json']:p=OUT/'provenance'/model/n;p.parent.mkdir(parents=True,exist_ok=True);shutil_copy(ROOT/model/n,p)
 write(OUT/'provenance/input_manifest.json',manifest)
 for p,h in manifest['protected_source_hashes'].items():assert sha(p)==h,p
 assets={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='files_sha256.json'}
 write(OUT/'files_sha256.json',assets)
 archive=ROOT/'room0_top30_candidates.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
  for rel in sorted(assets)+['files_sha256.json']:z.write(OUT/rel,rel)
 write(ROOT/'delivery_receipt.json',dict(setup,archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(assets)+1))
 print('READY',json.dumps(dict(cases=30,views=90,crops=len(viewer['crops']),bytes=archive.stat().st_size)),flush=True)

def shutil_copy(source,target):
 import shutil
 shutil.copyfile(source,target)

if __name__=='__main__':main()
