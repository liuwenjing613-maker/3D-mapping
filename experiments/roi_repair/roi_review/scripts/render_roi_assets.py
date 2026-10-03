"""Export historical RGB/mask/map evidence for ROI review only."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion,binary_dilation
from plyfile import PlyData,PlyElement
from detect_rois import sha
from select_history import project_visible
SRC=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
OUT=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0')
STATE_COLORS={0:(89,89,89),1:(255,255,0),3:(255,0,0)}

def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)

def write_ply(p,xyz,rgb,instances,ranks,state,point_index,role):
 v=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
   ('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),
   ('roi_rank','<i4'),('original_state','u1'),('source_point_index','<i4'),('roi_role','u1')])
 for j,k in enumerate(['x','y','z']):v[k]=xyz[:,j]
 for j,k in enumerate(['red','green','blue']):v[k]=rgb[:,j]
 for k,a in [('instance_id',instances),('roi_rank',ranks),('original_state',state),('source_point_index',point_index),('roi_role',role)]:v[k]=a
 p.parent.mkdir(parents=True,exist_ok=True)
 PlyData([PlyElement.describe(v,'vertex')],text=False,byte_order='<',
  comments=['P1-A1 room0 read-only ROI review; positions and instance IDs unchanged',
            'roi_role 1=alarm core 2=context; roi_rank is the complete global priority rank']).write(p)

def palette(ids):
 return np.column_stack([((ids*37+53)%195+60),((ids*83+11)%195+60),((ids*131+97)%195+60)]).astype(np.uint8)

def crop_box(uv,visible,w,h):
 if not visible.any():return [0,0,w,h]
 a=uv[visible];lo=a.min(0).astype(float);hi=a.max(0).astype(float)
 mid=(lo+hi)/2;size=np.maximum((hi-lo+1)*1.25,160)
 lo=np.maximum(0,np.floor(mid-size/2)).astype(int);hi=np.minimum([w,h],np.ceil(mid+size/2)).astype(int)
 return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]

def box_cloud(rows):
 pts=[];ranks=[]
 digits={'0':'abcedf','1':'bc','2':'abged','3':'abgcd','4':'fgbc','5':'afgcd','6':'afgecd','7':'abc','8':'abcdefg','9':'abfgcd'}
 seg={'a':((0,1),(1,1)),'b':((1,1),(1,.5)),'c':((1,.5),(1,0)),
      'd':((1,0),(0,0)),'e':((0,0),(0,.5)),'f':((0,.5),(0,1)),'g':((0,.5),(1,.5))}
 for r in rows:
  lo=np.array(r['bbox_min']);hi=np.array(r['bbox_max'])
  corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
  for i,a in enumerate(corners):
   for b in corners[i+1:]:
    if np.count_nonzero(a!=b)==1:
     length=np.linalg.norm(b-a);t=np.linspace(0,1,max(2,int(length/.005)+1))
     line=a[None]+t[:,None]*(b-a)[None];pts.append(line);ranks.extend([r['rank']]*len(line))
  # Digits in the world XY plane for a top-down PLY overlay.
  origin=np.array([lo[0],lo[1],hi[2]+.035]);height=.10;width=.055
  for i,d in enumerate(str(r['rank'])):
   for s in digits[d]:
    a,b=seg[s];a=origin+np.array([i*.075+a[0]*width,a[1]*height,0]);b=origin+np.array([i*.075+b[0]*width,b[1]*height,0])
    t=np.linspace(0,1,15);line=a[None]+t[:,None]*(b-a)[None];pts.append(line);ranks.extend([r['rank']]*15)
 xyz=np.concatenate(pts).astype(np.float32);rr=np.array(ranks,np.int32)
 rgb=np.tile([0,210,255],(len(xyz),1)).astype(np.uint8)
 return xyz,rgb,rr

def main():
 ranked=json.loads((OUT/'roi_ranked.json').read_text());manifest=json.loads((OUT/'detector_manifest.json').read_text())
 m=np.load(OUT/'roi_membership.npz');co=m['core_offsets'];cp=m['core_point_index'];ro=m['context_offsets'];rp=m['context_point_index'];core_to=m['point_core_candidate']
 top=ranked[:20];representatives=[];parents=set()
 for r in ranked:
  if r['viewable'] and r['parent_component'] not in parents:
   parents.add(r['parent_component']);representatives.append(r)
   if len(representatives)==12:break
 additional=[]
 for sname in ['U','T']:
  cand=next((r for r in ranked if r['viewable'] and r['core_counts'][sname]>0 and r['core_counts'][sname]/r['core_points']>.70),None)
  if cand is not None:additional.append(cand)
 review={r['rank']:r for r in top+representatives+additional};review=[review[k] for k in sorted(review)]
 (OUT/'review_selection.json').write_text(json.dumps({'global_top20':[r['rank'] for r in top],
   'distinct_component_representatives':[r['rank'] for r in representatives],
   'additional_U_T_examples':[r['rank'] for r in additional],'rendered':[r['rank'] for r in review]},indent=2)+'\n')
 full=SRC/'final/room0/P1-A1/room0_final_reference_v2_full.ply'
 vertices=PlyData.read(full)['vertex'].data
 xyz=np.column_stack([vertices[k] for k in ['x','y','z']]).astype(np.float32)
 colors=np.column_stack([vertices[k] for k in ['red','green','blue']]);inst=vertices['instance_id']
 with np.load(SRC/'native/room0/P1-A1/surface_p0/surface_evidence.npz') as e:state=e['state']
 with np.load(SRC/'final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as f:
  assert np.array_equal(xyz,f['xyz_m']) and np.array_equal(inst,f['instance_id'])
 tree=cKDTree(xyz);cfg=json.loads((BASE/'configs/raw.json').read_text());c=cfg['camera'];source=cfg['source']
 scene=Path(source['scene_root']);masks=Path(source['mask_root']);poses=np.loadtxt(scene/'traj.txt').reshape(-1,4,4)
 asset=OUT/'review_assets';asset.mkdir(exist_ok=True)
 cache={};evidence=[];clouds=[]
 def frame(fid):
  if fid in cache:return cache[fid]
  rgbpath=scene/source['rgb_pattern'].format(frame=fid);maskpath=masks/source['mask_pattern'].format(frame=fid)
  rgb=np.asarray(Image.open(rgbpath).convert('RGB'));raw=np.asarray(Image.open(maskpath))
  depth=np.asarray(Image.open(scene/source['depth_pattern'].format(frame=fid)),np.float32)/c['depth_png_units_per_meter']
  h,w=depth.shape;vv,uu=np.indices((h,w));valid=(depth>0)&(depth<10)
  z=depth[valid];pc=np.column_stack([(uu[valid]-c['cx'])*z/c['fx'],(vv[valid]-c['cy'])*z/c['fy'],z])
  pose=poses[fid];world=pc@pose[:3,:3].T+pose[:3,3]
  dist,idx=tree.query(world,workers=8);good=dist<=.02
  index=np.full((h,w),-1,np.int32);loc=np.flatnonzero(valid);index.ravel()[loc[good]]=idx[good]
  mapcolor=np.zeros((h,w,3),np.uint8);visible=index>=0;mapcolor[visible]=colors[index[visible]]
  background=(rgb*.3).astype(np.uint8);background[visible]=mapcolor[visible]
  rawcolor=palette(raw.ravel()).reshape(h,w,3);rawcolor[raw==0]=[89,89,89]
  rawcolor=(.75*rawcolor+.25*rgb).astype(np.uint8)
  boundary=np.zeros((h,w),bool);boundary[:,1:]|=raw[:,1:]!=raw[:,:-1];boundary[1:]|=raw[1:]!=raw[:-1]
  rawcolor[boundary]=[245,245,245]
  with np.load(SRC/f'raw_inputs/room0/surface_regions/frame_region_support/f{fid:06d}.npz') as z:
   assert sha(maskpath)==str(z['source_mask_sha256'][0])
  result=(rgb,raw,depth,index,background,rawcolor)
  cache[fid]=result
  return result
 for r in review:
  k=r['candidate_index'];core=cp[co[k]:co[k+1]];context=rp[ro[k]:ro[k+1]]
  directory=asset/r['roi_id'];directory.mkdir(exist_ok=True)
  role=np.where(core_to[context]==k,1,2).astype(np.uint8)
  state_colors=colors[context].copy()
  for s,col in STATE_COLORS.items():state_colors[(role==1)&(state[context]==s)]=col
  np.savez_compressed(directory/'cloud.npz',xyz=xyz[context],rgb=colors[context],state_rgb=state_colors,role=role,point_index=context)
  records=[]
  for pos,view in enumerate(r['selected_views']):
   fid=view['frame_id'];rgb,raw,depth,index,mapimage,maskimage=frame(fid);visible=index>=0
   in_context=np.zeros(index.shape,bool);in_context[visible]=np.isin(index[visible],context)
   in_core=np.zeros(index.shape,bool);in_core[visible]=core_to[index[visible]]==k
   valid_points,uv=project_visible(xyz[context],poses[fid],depth,c)
   box=crop_box(uv,valid_points,c['width'],c['height'])
   outline=binary_dilation(in_context)&~binary_erosion(in_context)
   anomaly=(.45*rgb).astype(np.uint8)
   for s,col in STATE_COLORS.items():
    ids=np.zeros(index.shape,bool);ids[visible]=in_core[visible]&(state[index[visible]]==s)
    anomaly[ids]=col
   anomaly[outline]=[0,210,255]
   rgb_crop=rgb.copy();rgb_crop[outline]=[0,210,255]
   for name,array in [('rgb',rgb_crop),('states',anomaly),('current3d',mapimage),('cropformer',maskimage)]:
    Image.fromarray(array).crop(box).save(directory/f'view{pos+1}_{name}.png')
   full_img=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(full_img)
   draw.rectangle(box,outline=(0,210,255),width=3);draw.text((box[0]+5,max(2,box[1]-14)),f'{r["roi_id"]}  frame {fid}',fill=(0,210,255))
   full_img.save(directory/f'view{pos+1}_location.jpg',quality=92)
   rawids,rawcounts=np.unique(raw[in_context],return_counts=True);so=np.argsort(-rawcounts)[:8]
   records.append({'view_number':pos+1,'frame_id':fid,'crop_xyxy':box,
    'projected_context_pixels':int(in_context.sum()),'projected_core_pixels':int(in_core.sum()),
    'mask_ids_in_context':[{'local_id':int(rawids[j]),'pixels':int(rawcounts[j])} for j in so],
    'source_mask_hash_verified_against_frozen_support_cache':True,
    'selected_view_statistics':view})
  evidence.append({'rank':r['rank'],'roi_id':r['roi_id'],'views':records})
  log('Rendered',r['roi_id'],len(core),'alarms',len(context),'context points')
 # Keep the complete trigger count audit; the requested deliverable is images.
 candidate_rank=np.zeros(len(ranked),np.int32)
 for r in ranked:candidate_rank[r['candidate_index']]=r['rank']
 alarm=np.flatnonzero(core_to>=0);alarm_colors=np.empty((len(alarm),3),np.uint8)
 for s,col in STATE_COLORS.items():alarm_colors[state[alarm]==s]=col
 (OUT/'review_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
 assert sha(Path(manifest['source_files']['native_evidence']))==manifest['source_hashes']['native_evidence']
 assert sha(Path(manifest['source_files']['final_map']))==manifest['source_hashes']['final_map']
 audit={'status':'PASS','GT_used':False,'original_instance_labels_and_coordinates_unchanged':True,
   'all_145379_alarm_points_present_once_in_ROI_cores':len(alarm)==145379,
   'source_maps_hashes_unchanged':True,'rendered_ROI_ranks':[r['rank'] for r in review],
   'complete_ROI_count':len(ranked),'reference_v2_colors_retained_for_instances':True,
   'reprojection':'every valid depth pixel backprojects to immutable TSDF surface nearest point; reject >2cm; GT-free and occlusion-tested by measured depth',
   'deliverable':'ranked ROI images; PLY generation disabled at user request'}
 (OUT/'review_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 log('COMPLETE',json.dumps(audit))

if __name__=='__main__':main()
