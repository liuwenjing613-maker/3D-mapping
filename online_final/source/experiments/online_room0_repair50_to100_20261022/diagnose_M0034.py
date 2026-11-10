"""Read-only M0034 spatial/support evidence on the fixed observed100 surface."""
import sys
from pathlib import Path
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/tracking100')
sys.path.insert(0,str(C))
from bootstrap100 import *
import numpy as np,hashlib,collections
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,binary_fill_holes,label
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=TOP/'review/diagnosis_M0034'
D.mkdir(parents=True,exist_ok=True)
def counts(a):
 u,n=np.unique(a,return_counts=True);return dict(zip(u.astype(str).tolist(),n.tolist()))
def stats(p,labels,xyz,u,v,inside,dist):
 out={}
 for pid in np.unique(labels):
  sel=labels==pid;q=xyz[p[sel]]
  out[str(int(pid))]=dict(count=int(sel.sum()),fraction=float(sel.mean()),
   xyz_min=q.min(0).tolist(),xyz_max=q.max(0).tolist(),xyz_centroid=q.mean(0).tolist(),
   projected_inside_original_mask=int(inside[sel].sum()),
   projected_mask_boundary_distance_pixels=dict(minimum=float(dist[sel].min()),median=float(np.median(dist[sel])),p95=float(np.quantile(dist[sel],.95)),
    count_gt_3px=int((dist[sel]>3).sum()),count_gt_5px=int((dist[sel]>5).sum())),
   image_uv_min=[int(u[sel].min()),int(v[sel].min())],image_uv_max=[int(u[sel].max()),int(v[sel].max())])
 return out
def main():
 started=time.perf_counter()
 scene=r.Scene('room0');frame=scene.source.load_frame(285)
 src=TOP/'prefix100_new50/all_edge_20261021/v1'
 seed=next(x for x in read(TRACKROOT/'B/seed_id_map.json')['objects'] if x['mask_id']=='M0034')
 primary=seed['source'];mask_path=src/primary['mask'];mask=np.asarray(Image.open(mask_path))>0
 assert int(mask.sum())==22030 and digest(mask_path)==primary['mask_sha256']
 source_path=Path(primary['representative']['source_path'])
 sam=np.asarray(Image.open(source_path));same=np.array_equal(sam==primary['representative']['label'],mask);assert same
 z=np.load(src/'surface_supports.npz');slot=None
 for index,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
  q=z['indices'][a:b]
  if len(q)==primary['support_count'] and hashlib.sha256(q.tobytes()).hexdigest()==primary['support_sha256']:slot=index;p=q.copy();break
 assert slot is not None and len(p)==14336
 labels=scene.final[p];native=scene.original[p];assert counts(labels)=={'-1':3929,'19':1,'57':853,'89':9553}
 np.savez_compressed(D/'support_partition.npz',surface_point_index=p,xyz_m=scene.xyz[p],final_instance_id=labels,native_instance_id=native,
  non89_surface_point_index=p[labels!=89],index89=p[labels==89],index_unassigned=p[labels<0],index57=p[labels==57],index19=p[labels==19])
 # True pinhole projection using source camera pose/intrinsics; no altered masks.
 cam=(np.c_[scene.xyz[p],np.ones(len(p))]@np.linalg.inv(frame.camera_to_world).T)[:,:3]
 uf=frame.camera.fx*cam[:,0]/cam[:,2]+frame.camera.cx;vf=frame.camera.fy*cam[:,1]/cam[:,2]+frame.camera.cy
 u=np.rint(uf).astype(int);v=np.rint(vf).astype(int)
 valid=(cam[:,2]>0)&(u>=0)&(u<1200)&(v>=0)&(v<680)
 assert valid.all()
 inside=mask[v,u];dist=distance_transform_edt(mask)[v,u];depth_residual=np.abs(frame.depth_m[v,u]-cam[:,2])
 category=stats(p,labels,scene.xyz,u,v,inside,dist)
 # Connectivity is a diagnostic only, not candidate filtering or repair.
 pairs=cKDTree(scene.xyz[p]).query_pairs(.025,output_type='ndarray')
 adj=coo_matrix((np.ones(len(pairs)*2), (np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(p),len(p))).tocsr()
 ncomp,component=connected_components(adj,directed=False)
 cn=np.bincount(component)
 # Exact 2D->surface Gate seed support, separate from reverse-projection selection support.
 zz=np.load(TRACKROOT/'B/cases/room0v2-f000285-b01/seed_projected_support.npz')
 gp=np.unique(zz['surface_point_index'][zz['track_id']==3])
 samples=gp[np.linspace(0,len(gp)-1,min(len(gp),2048),dtype=int)]
 decision=next(x for x in read(TRACKROOT/'B/gate/frame_decisions.json')['frames'] if x['frame']==285)
 source_gate=dict(initial=decision['initial_objects']['3'],final=decision['objects']['3'])
 frozen=read(TRACKROOT/'B/cases/room0v2-f000285-b01/frozen_association.json')
 # Save original RGB/mask links and explicitly marked observational projections.
 for name,target in [('rgb_source.jpg',TRACKROOT/'inputs/scene/results/frame000285.jpg'),('mask_original.png',mask_path)]:
  link=D/name
  if link.exists():assert link.resolve()==target.resolve()
  else:link.symlink_to(target)
 im=Image.fromarray(frame.rgb);box=(830,0,1040,430)
 im.crop(box).save(D/'rgb_crop.png')
 Image.fromarray(mask.astype(np.uint8)*255).crop(box).save(D/'mask_crop.png')
 overlay=frame.rgb.copy();overlay[mask]=np.rint(.55*overlay[mask]+.45*np.array([35,98,255])).astype(np.uint8)
 Image.fromarray(overlay).crop(box).save(D/'original_mask_overlay.png')
 colors={89:(10,165,205),-1:(255,158,28),57:(226,44,66),19:(179,51,209)}
 proj=frame.rgb.copy()
 # Nearest depth wins when multiple 3D support points project to one pixel.
 for j in np.argsort(cam[:,2])[::-1]:proj[v[j],u[j]]=colors[int(labels[j])]
 Image.fromarray(proj).crop(box).save(D/'selection_support_projection.png')
 # Orthographic point plot in world x/z; all selected points retained.
 fig,ax=plt.subplots(figsize=(5.8,7.2),dpi=160)
 for pid in [89,-1,57,19]:
  choose=labels==pid
  ax.scatter(scene.xyz[p[choose],0],scene.xyz[p[choose],2],s=5 if pid!=19 else 24,c=np.array(colors[pid])[None,:]/255,label=f'ID {pid}: {int(choose.sum())}',linewidths=0)
 ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World z (m)');ax.set_title('M0034 selected surface points (world x-z)\nRead-only projection; same 100-frame geometry');ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
 fig.tight_layout();fig.savefig(D/'world_xz_partition.png',bbox_inches='tight');plt.close(fig)
 report=dict(status='PASS',seed_group='B/A003',mask_id='M0034',raw_frame=285,processed_frame=58,GT_used=False,map_write=False,
  geometry_sha256=digest(B/'geometry/surface.ply'),surface_count=len(scene.xyz),
  source=dict(mask_pixels=int(mask.sum()),mask_sha256=digest(mask_path),original_source_label_equals_seed=same,
   sam_label=primary['representative']['label'],source_model=primary['representative']['model'],routing=primary['representative']['selection_reason'],
   semantic_family=primary['representative']['semantic_family'],source_box=primary['representative']['source_box'],source_path=str(source_path),
   bbox=primary['bbox'],connected_2d_components=int(label(mask)[1]),internal_hole_pixels=int(binary_fill_holes(mask).sum()-mask.sum())),
  selection_support=dict(count=len(p),support_slot=slot,sha256=primary['support_sha256'],final_counts=counts(labels),native_counts=counts(native),
   non89_count=int((labels!=89).sum()),precision_against_ID89=9553/14336,by_final_ID=category,
   coordinate_plane=dict(y_range=[float(scene.xyz[p,1].min()),float(scene.xyz[p,1].max())],
    y_std_m=float(scene.xyz[p,1].std()),spatial_components_radius_m=.025,components=ncomp,largest_component_points=int(cn.max()),component_sizes_descending=sorted(cn.tolist(),reverse=True)),
   projection=dict(method='World points transformed by inverse raw285 camera_to_world, pinhole fx=fy=600,cx=599.5,cy=339.5; round nearest pixel. Nearest camera depth drawn on collision. No mask dilation, fill, or editing.',
    inside_source_mask_points=int(inside.sum()),total=len(p),depth_abs_residual_median_m=float(np.median(depth_residual)),depth_abs_residual_p95_m=float(np.quantile(depth_residual,.95)),depth_abs_residual_max_m=float(depth_residual.max())),
   outside_ID89_are_not_all_other_objects='Unassigned label -1 is not a different object label; label mismatch alone does not establish SAM oversegmentation.'),
  gate_support=dict(method='Original pixel_stride=2 depth->nearest-surface projection, max distance0.015m; uniformly sample at most2048 sorted surface indices without filtering current instance ID.',
   projected_points=len(gp),final_ID_counts=counts(scene.final[gp]),anchor_samples=len(samples),anchor_final_ID_counts=counts(scene.final[samples]),
   sampled_only_ID89=bool(np.all(scene.final[samples]==89)),anchor_non89_count=int(np.count_nonzero(scene.final[samples]!=89)),source_frame_gate=source_gate,
   source_prediction_exact=bool(np.array_equal(np.asarray(Image.open(TRACKROOT/'B/cases/room0v2-f000285-b01/tracking/f000285.png'))==3,mask))),
  source_images=dict(original_rgb='rgb_source.jpg',original_mask='mask_original.png',crop_box=list(box),
   rgb_crop='rgb_crop.png',original_mask_overlay='original_mask_overlay.png',selection_surface_projection='selection_support_projection.png',world_xz='world_xz_partition.png'),
  seconds=time.perf_counter()-started,interpretation='Selection support14336, forward-projected Gate support5499 and sampled anchors2048 are different denominators. No GT judgment is made. Read original RGB/mask and 3D partition jointly.')
 scene.verify_unchanged();dump(D/'diagnosis.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in ['gate_support','source_images']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
