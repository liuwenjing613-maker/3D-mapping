"""Select visible historical views without gating on mask support or reading GT."""
from pathlib import Path
import json,csv,time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
OUT=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002')
SRC=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0')

def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)

def project_visible(points,pose,depth,c,tolerance=.02):
 pc=(points.astype(np.float64)-pose[:3,3])@pose[:3,:3]
 z=pc[:,2];front=(z>.05)&(z<10)
 uv=np.full((len(points),2),-1,np.int32)
 uv[front]=np.rint(np.column_stack([c['fx']*pc[front,0]/z[front]+c['cx'],c['fy']*pc[front,1]/z[front]+c['cy']])).astype(np.int32)
 inside=front&(uv[:,0]>=0)&(uv[:,0]<c['width'])&(uv[:,1]>=0)&(uv[:,1]<c['height'])
 jj=np.flatnonzero(inside);u,v=uv[jj].T;dep=depth[v,u]
 back=np.column_stack([(u-c['cx'])*dep/c['fx'],(v-c['cy'])*dep/c['fy'],dep])
 good=(dep>0)&(dep<10)&(np.linalg.norm(back-pc[jj],axis=1)<=tolerance)
 visible=np.zeros(len(points),bool);visible[jj[good]]=True
 return visible,uv

def main():
 rows=json.loads((OUT/'roi_candidates.json').read_text());nroi=len(rows)
 m=np.load(OUT/'roi_membership.npz');core_to=m['point_core_candidate'];vox=m['voxel_id_1cm']
 core=m['core_point_index'];own=core_to[core]
 # Representative of each (candidate, 1cm voxel), deterministic point order.
 order=np.lexsort((core,vox[core],own));ordered=core[order]
 starts=np.r_[True,(own[order][1:]!=own[order][:-1])|(vox[ordered][1:]!=vox[ordered][:-1])]
 rep=ordered[starts];owner=core_to[rep]
 denom=np.bincount(owner,minlength=nroi)
 np.save(OUT/'core_view_representatives.npy',rep)
 with np.load(SRC/'final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as f:xyz=f['xyz_m']
 points=xyz[rep]
 cfg=json.loads((BASE/'configs/raw.json').read_text());camera=cfg['camera'];scene=Path(cfg['source']['scene_root'])
 poses=np.loadtxt(scene/'traj.txt').reshape(-1,4,4)
 fids=np.arange(0,2000,5,dtype=np.int32)
 visible_matrix=np.zeros((len(fids),len(rep)),bool)
 counts=np.zeros((nroi,len(fids)),np.int32);pixels=np.zeros_like(counts)
 for j,fid in enumerate(fids):
  depth=np.asarray(Image.open(scene/f'results/depth{fid:06d}.png'),np.float32)/camera['depth_png_units_per_meter']
  vis,uv=project_visible(points,poses[fid],depth,camera)
  visible_matrix[j]=vis
  counts[:,j]=np.bincount(owner[vis],minlength=nroi)
  keys=owner[vis].astype(np.int64)*(camera['width']*camera['height'])+uv[vis,1]*camera['width']+uv[vis,0]
  unique=np.unique(keys)
  pixels[:,j]=np.bincount(unique//(camera['width']*camera['height']),minlength=nroi)
  if (j+1)%40==0:log('Depth visibility',j+1,'/',len(fids))
 coverage=counts/denom[:,None]
 resolution=np.minimum(1.,pixels/np.maximum(1,counts))
 quality=coverage*(.65+.35*resolution)
 by_owner=np.argsort(owner,kind='stable');offset=np.r_[0,np.cumsum(denom)]
 for k,r in enumerate(rows):
  r['view_representative_voxels']=int(denom[k]);r['visible_historical_frames']=int(np.sum(counts[k]>0))
  available=list(np.flatnonzero(counts[k]>0));selected=[];seen=np.zeros(denom[k],bool)
  ridx=by_owner[offset[k]:offset[k+1]]
  directions=poses[fids,:3,3]-np.array(r['center'])
  directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
  for view_no in range(min(3,len(available))):
   if not selected:
    chosen=max(available,key=lambda j:(quality[k,j],pixels[k,j],-int(fids[j])))
    angle=None
   else:
    angles=np.rad2deg(np.arccos(np.clip(directions@directions[selected].T,-1,1))).min(1)
    def value(j):
     novel=np.mean(visible_matrix[j,ridx]&~seen)
     return quality[k,j]*(.25+.75*min(1.,angles[j]/25.))+.20*novel
    chosen=max(available,key=lambda j:(value(j),quality[k,j],-int(fids[j])))
    angle=float(angles[chosen])
   selected.append(chosen);available.remove(chosen);seen|=visible_matrix[chosen,ridx]
  r['selected_views']=[{'frame_id':int(fids[j]),'visible_core_voxels':int(counts[k,j]),
    'core_coverage':float(coverage[k,j]),'visible_core_pixels':int(pixels[k,j]),
    'view_quality':float(quality[k,j]),
    'min_angle_to_previous_deg':None if pos==0 else float(np.rad2deg(np.arccos(np.clip(directions[j]@directions[selected[:pos]].T,-1,1))).min())}
    for pos,j in enumerate(selected)]
  r['three_view_union_coverage']=float(seen.mean())
  r['primary_coverage']=float(coverage[k,selected[0]]) if selected else 0.
  r['viewable']=bool(selected)
 ranked=sorted(rows,key=lambda r:(not r['viewable'],-r['score_priority'],-(r['conflict_frame_votes_median'] or 0),r['candidate_index']))
 for rank,r in enumerate(ranked,1):r['rank']=rank;r['roi_id']=f'ROI-{rank:04d}'
 (OUT/'roi_ranked.json').write_text(json.dumps(ranked,ensure_ascii=False,indent=2)+'\n')
 with (OUT/'roi_ranking.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=['rank','roi_id','candidate_index','score_priority','score_density','score_reference_raw','core_points','U','T','CONFLICT','context_points','stable_context_points','primary_frame','primary_coverage','three_view_union_coverage','visible_historical_frames','competition','extent_m']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in ranked:
   row={k:r[k] for k in fields if k in r}
   row.update(r['core_counts']);row['primary_frame']=r['selected_views'][0]['frame_id'] if r['selected_views'] else ''
   row['competition']=json.dumps(r['dominant_competing_ids']);row['extent_m']=json.dumps(r['extent_m'])
   w.writerow(row)
 np.savez_compressed(OUT/'historical_visibility.npz',frame_ids=fids,core_representatives=rep,
    visible=np.packbits(visible_matrix,axis=1),counts=counts,pixels=pixels)
 manifest={'frames_checked':len(fids),'frame_range':'0:2000:5','depth_visibility_tolerance_m':.02,
    'visibility':'positive depth and Euclidean surface-to-depth-backprojection distance <= 2cm; no mask gate',
    'views_per_ROI':3,'view_diversity_target_deg':25,'main_view_score':'coverage * (0.65 + 0.35 * unique_pixels / visible_core_voxels)',
    'auxiliary_selection':'quality weighted by angle diversity plus newly covered voxels',
    'viewable_ROIs':sum(r['viewable'] for r in ranked),'ROIs_without_depth_visible_views':sum(not r['viewable'] for r in ranked),
    'ranking':'viewable proposals first, then stabilized weighted abnormal density; no proposals discarded',
    'GT_used':False,'map_modified':False}
 (OUT/'frame_selector_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 log('COMPLETE',json.dumps(manifest))
 log('TOP',json.dumps([{k:r[k] for k in ['roi_id','candidate_index','score_priority','core_counts','core_points','context_points','primary_coverage','selected_views']} for r in ranked[:12]]))

if __name__=='__main__':main()
