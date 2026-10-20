"""Render verifiable seed evidence and the completed R2 report for local review."""
from pathlib import Path
import collections,hashlib,json,shutil,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion,binary_dilation
from build_pr_exemption_preview import build_preview
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019')
O=R/'review';A=O/'assets'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def jpg(arr,name):
 Image.fromarray(np.uint8(np.clip(arr,0,255))).save(A/name,quality=90)
 return 'assets/'+name

def overlay(rgb,mask,color=(20,222,192),alpha=.36):
 out=rgb.copy().astype(float);out[mask]=out[mask]*(1-alpha)+np.array(color)*alpha
 edge=binary_dilation(mask)&~binary_erosion(mask);out[edge]=color
 return out

def collection_image(rgb,rows,masks):
 out=rgb.copy().astype(float)
 for row in rows:
  mask=masks[row['group_id']];tid=row['original_track_id'];col=tuple(int(x) for x in np.random.default_rng(tid+1709).integers(65,240,3))
  out=overlay(out,mask,col,.27)
 im=Image.fromarray(np.uint8(out));draw=ImageDraw.Draw(im)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',16)
 for row in rows:
  y,x=np.nonzero(masks[row['group_id']]);mid=len(x)//2;xx,yy=int(x[mid]),int(y[mid]);text=row['group_id']
  box=draw.textbbox((xx,yy),text,font=font);draw.rectangle((box[0]-3,box[1]-3,box[2]+3,box[3]+3),fill='#102734');draw.text((xx,yy),text,font=font,fill='white')
 return np.asarray(im)

def main():
 A.mkdir(parents=True,exist_ok=True)
 done=read(R/'v2/complete.json');assert done['status']=='PASS'
 src=read(R/'v1/seed_manifest.json');selection=read(R/'v2/selection.json');measures=read(R/'v2/measurements.json');sem=read(R/'v2/semantic/classification.json')
 assert sha(R/'v2/selection.json')==done['selection_sha256']
 source={g['id']:g for g in src['groups']};sr={s['group_id']:s for s in sem['records']};mc={s['group_id']:s for s in measures['seeds']};cov={s['group_id']:s for s in measures['coverage']}
 frames={f['frame']:f for f in sem['frames']};rows=[];masks={};rgbs={};source_cache={};byframe=collections.defaultdict(list)
 for rec in selection['records']:
  gid=rec['group_id'];g=source[gid];p=g['primary'];f=p['frame'];a=p['representative'];rgb_path=frames[f]['RGB_path']
  if f not in rgbs:
   assert sha(rgb_path)==frames[f]['RGB_sha256'];rgbs[f]=np.asarray(Image.open(rgb_path).convert('RGB'));jpg(rgbs[f],f'f{f:06d}_rgb.jpg')
  rgb=rgbs[f];path=R/'v1'/p['mask'];assert sha(path)==p['mask_sha256'];mask=np.asarray(Image.open(path))>0;masks[gid]=mask
  shutil.copy2(path,A/(gid+'_mask.png'))
  images=dict(rgb=f'assets/f{f:06d}_rgb.jpg',mask=f'assets/{gid}_mask.png',overlay=jpg(overlay(rgb,mask),gid+'_overlay.jpg'))
  sp=a['source_path'];box=a['source_box'];key=(sp,tuple(box));x0,y0,x1,y1=map(int,box)
  if key not in source_cache:
   arr=np.asarray(Image.open(sp));assert arr.shape==(y1-y0,x1-x0)
   out=rgb.copy().astype(float)*.6;region=out[y0:y1,x0:x1];colors=np.random.default_rng(2201).integers(50,240,(int(arr.max())+1,3));valid=arr>0
   region[valid]=rgb[y0:y1,x0:x1][valid]*.4+colors[arr[valid]]*.6
   source_cache[key]=out
  source_view=source_cache[key].copy();edge=binary_dilation(mask)&~binary_erosion(mask);source_view[edge]=(255,255,255)
  images['source_segmentation']=jpg(source_view,gid+'_source.jpg')
  semrec=sr[gid]
  for name,rel in semrec['images'].items():
   q=R/'v2/semantic'/rel;assert q.is_file();dest=A/(gid+'_semantic_'+name+q.suffix);shutil.copy2(q,dest);images['semantic_'+name]='assets/'+dest.name
  frame_sem=R/'v2/semantic'/frames[f]['semantic_preview'];dest=A/(f'f{f:06d}_semantic.jpg');
  if not dest.exists():shutil.copy2(frame_sem,dest)
  images['whole_semantic']='assets/'+dest.name
  coverage=cov.get(gid)
  if coverage and coverage['instance_id']>0:
   with np.load(R/'overmerge'/f'frame{f:06d}_visible.npz') as z:idx=z['vertex_index'];uv=z['uv']
   if 'baseline_labels' not in locals():
    with np.load('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room0/abstain/final/instance_surface.npz') as z:baseline_labels=z['instance_id']
   take=baseline_labels[idx]==coverage['instance_id'];vis=np.zeros(mask.shape,bool);vis[uv[take,1],uv[take,0]]=True
   images['matched_instance']=jpg(overlay(overlay(rgb,binary_dilation(vis),(255,180,35),.45),mask,(20,222,192),.2),gid+'_instance.jpg')
  row=dict(**rec,area=p['area'],support_count=p['support_count'],model=['CropFormer 整图','CropFormer 裁剪','SAM 2.1'][a['model']],source_label=a['label'],source_box=box,ROIs=sorted({a['ROI'] for a in p['aliases']}),images=images,coverage=coverage,semantic={k:semrec[k] for k in ['top_class','top_class_fraction','high_excluded_fraction','high_excluded_pixels','low_confidence_fraction','decision','categories']},source_mask_sha256=p['mask_sha256'],auxiliary_count=len(g['auxiliary_ids']),source_correction=p.get('source_policy_user_override'))
  rows.append(row);byframe[f].append(row)
 frame_rows=[]
 for frame,items in sorted(byframe.items()):
  after=[r for r in items if r['decision']=='retain'];rgb=rgbs[frame]
  frame_rows.append(dict(frame=frame,before=len(items),after=len(after),group_ids=[r['group_id'] for r in items],before_image=jpg(collection_image(rgb,items,masks),f'f{frame:06d}_before.jpg'),after_image=jpg(collection_image(rgb,after,masks),f'f{frame:06d}_after.jpg')))
 evaluation=read(R/'e1_repair/evaluation_r2/comparison.json') if (R/'e1_repair/evaluation_r2/comparison.json').exists() else None
 if evaluation:assert read(R/'e1_repair/run_complete.json')['status']=='PASS'
 paired={stage:read(R/'e1_repair/evaluation_r2/paired'/stage/'paired_repair_metrics.json') for stage in ['native','final']} if evaluation else None
 data=dict(scene='room0',generated=time.time(),v1=read(R/'v1/audit.json')['stats'],selection=selection,rows=rows,frames=frame_rows,evaluation=evaluation,source_hashes={n:sha(R/n) for n in ['v1/seed_manifest.json','v1/audit.json','v2/selection.json','v2/measurements.json','v2/seed_manifest.json','v2/source_lock.json']},code_root=str(C),result_root=str(R),manual_overrides=[],manual_source_corrections=src.get('manual_source_corrections'),GT_used_for_selection=False)
 data['paired_evaluation']=paired
 if (R/'comparison_three_way/summary.json').exists():
  data['three_way']=read(R/'comparison_three_way/summary.json');assert data['three_way']['status']=='PASS'
 (O/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
 build_preview(data,src,O)
 for name in ['index.html','style.css','app.js','comparison.js','pr_exemption.html','pr_exemption.js','pr_exemption.css']:shutil.copy2(C/'review_web'/name,O/name)
 downloads=O/'reports';downloads.mkdir(exist_ok=True)
 for name in ['comparison_three_way/summary.json','comparison_three_way/consistency.json','comparison_three_way/inputs_frozen_before_scoring.json','v1/seed_manifest.json','v1/audit.json','v2/selection.json','v2/measurements.json','v2/seed_manifest.json','v2/source_lock.json','v2/complete.json','e1_repair/evaluation_r2/comparison.json','e1_repair/evaluation_r2/protocol_freeze.json','e1_repair/evaluation_r2/per_GT.json','e1_repair/selected_input_audit.json','e1_repair/run_complete.json','e1_repair/tests_complete.json','e1_repair/legacy_path_compatibility.json','e1_repair/pipeline_code_freeze.json','e1_repair/pipeline_complete.json','e1_repair/e1/E1_expanded_history/predictions_freeze.json','e1_repair/evaluation_r2/paired/native/paired_repair_metrics.json','e1_repair/evaluation_r2/paired/final/paired_repair_metrics.json']:
  p=R/name
  if p.exists():shutil.copy2(p,downloads/name.replace('/','__'))
 for name in ['README.md','adapter_provenance.json','source_policy_update_audit.json']:
  shutil.copy2(C/name,downloads/name)
 for row in rows:
  for path in row['images'].values():assert (O/path).is_file(),path
 assert len(rows)==done['input'] and sum(r['decision']=='retain' for r in rows)==done['retained']
 files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name!='review_manifest.json'}
 (O/'review_manifest.json').write_text(json.dumps(dict(status='PASS',seeds=len(rows),retained=done['retained'],evaluation_complete=bool(evaluation),files=files),indent=2))
 print('REVIEW_READY',len(rows),done['retained'],bool(evaluation),flush=True)
if __name__=='__main__':main()
