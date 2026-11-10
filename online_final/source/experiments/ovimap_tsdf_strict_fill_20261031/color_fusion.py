"""GT-free RGB-D visibility fusion on the exact native final surface."""
from common import *
import cv2
DEPTH_TOLERANCE=.05 # Native TSDF truncation distance (5 * 1 cm).
def visible_samples(xyz,pose,K,depth):
 camera=(np.asarray(xyz,np.float64)-pose[:3,3])@pose[:3,:3]
 z=camera[:,2];good=np.isfinite(camera).all(1)&(z>.1)&(z<10.)
 ix=np.flatnonzero(good)
 u=np.rint(K[0,0]*camera[ix,0]/z[ix]+K[0,2]).astype(np.int64)
 v=np.rint(K[1,1]*camera[ix,1]/z[ix]+K[1,2]).astype(np.int64)
 inside=(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
 ix,u,v=ix[inside],u[inside],v[inside]
 d=depth[v,u]
 keep=np.isfinite(d)&(d>.1)&(d<10.)&(np.abs(z[ix]-d)<=DEPTH_TOLERANCE)
 return ix[keep],v[keep],u[keep]

def sampled_rgb_world(depth,mask,rgb,pose,K):
 d=depth[::2,::2];m=mask[::2,::2]
 valid=(m>0)&np.isfinite(d)&(d>0)&(d<10)
 y,x=np.nonzero(valid);z=d[valid].astype(np.float64)
 camera=np.column_stack(((x*2-K[0,2])*z/K[0,0],(y*2-K[1,2])*z/K[1,1],z))
 return camera@pose[:3,:3].T+pose[:3,3],rgb[::2,::2][valid]
def recover_supported_colors(scene,xyz,missing,sums,counts,rows,base,inputs):
 from scipy.spatial import cKDTree
 sys.path.insert(0,str(OLD_C))
 from strict_votes import nearest,RADIUS
 targets=np.flatnonzero(missing)
 if not len(targets):return 0
 nearby=cKDTree(xyz[targets]);full=cKDTree(xyz)
 lookup=np.full(len(xyz),-1,np.int32);lookup[targets]=np.arange(len(targets))
 ownroot=OLD/'ours'/f'{scene}_400_main';checkpoint=read(ownroot/'checkpoint.json')
 root=Path('/home/chenkejun/beauty/conceptgraphs/data/Replica')/scene/'results'
 maskroot=Path('/data/chenkejun/ovimap_runtime_20260908/sem_aligned')/scene
 for i,(row,audit) in enumerate(zip(rows,base)):
  fid=row['frame_id'];history=ownroot/f'history/f{fid:06d}.npz'
  assert sha(history)==checkpoint['history_sha256'][str(history.relative_to(ownroot))]
  h=load(history)
  # Bound the candidate query, then retain only the exact original surface assignment.
  selected=np.flatnonzero(np.isfinite(nearby.query(h['world'],workers=2,distance_upper_bound=RADIUS)[0]))
  if not len(selected):continue
  point,ok=nearest(full,h['world'][selected])
  ok &= missing[np.minimum(point,len(xyz)-1)]
  if not ok.any():continue
  dp=root/f'depth{fid:06d}.png';rp=root/f'frame{fid:06d}.jpg';mp=maskroot/f'frame{fid:06d}.png'
  for path in [dp,rp,mp]:
   value=sha(path);assert value==audit['input_hashes'][str(path)];inputs[str(path)]=value
  depth=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED).astype(np.float32)/6553.5
  rgb=cv2.cvtColor(cv2.imread(str(rp)),cv2.COLOR_BGR2RGB)
  mask=cv2.imread(str(mp),cv2.IMREAD_UNCHANGED)
  world,colors=sampled_rgb_world(depth,mask,rgb,np.asarray(row['pose']),np.asarray(row['intrinsics']))
  np.testing.assert_array_equal(world,h['world'])
  dest=lookup[point[ok]];value=colors[selected[ok]]
  counts[targets]+=np.bincount(dest,minlength=len(targets)).astype(np.int32)
  for j in range(3):sums[targets,j]+=np.bincount(dest,weights=value[:,j],minlength=len(targets))
  if (i+1)%50==0:print('RGB_SUPPORT',scene,i+1,'recovered',int((counts[targets]>0).sum()),flush=True)
 return int((counts[targets]>0).sum())

def colors(scene):
 dst=O/'scenes'/scene;dst.mkdir(parents=True,exist_ok=True)
 attrs=load(dst/'normals.npz');xyz=attrs['xyz_m']
 rows=[json.loads(x) for x in (OLD/'native'/f'{scene}_400_trace/prefix_audit.jsonl').read_text().splitlines()]
 base=[json.loads(x) for x in (OLD/'ours'/f'{scene}_400_main/audit.jsonl').read_text().splitlines()]
 assert len(rows)==len(base)==400
 sums=np.zeros((len(xyz),3),np.float64);counts=np.zeros(len(xyz),np.int32)
 inputs={}
 root=Path('/home/chenkejun/beauty/conceptgraphs/data/Replica')/scene/'results'
 for i,(row,own) in enumerate(zip(rows,base)):
  fid=row['frame_id'];assert fid==own['frame_id']==i*5
  dp=root/f'depth{fid:06d}.png';rp=root/f'frame{fid:06d}.jpg'
  for path in [dp,rp]:
   h=sha(path);assert h==own['input_hashes'][str(path)];inputs[str(path)]=h
  # Use the original P1 depth-unit conversion and the actual native float32 pose values.
  depth=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED).astype(np.float32)/6553.5
  rgb=cv2.cvtColor(cv2.imread(str(rp),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
  pose=np.asarray(row['pose'],np.float64);K=np.asarray(row['intrinsics'],np.float64)
  ix,v,u=visible_samples(xyz,pose,K,depth)
  sums[ix]+=rgb[v,u];counts[ix]+=1
  if (i+1)%50==0:print('RGB',scene,i+1,'covered',int((counts>0).sum()),flush=True)
 direct=counts>0
 recovered=recover_supported_colors(scene,xyz,~direct,sums,counts,rows,base,inputs)
 valid=counts>0
 rgb=np.zeros((len(xyz),3),np.float32);rgb[valid]=(sums[valid]/counts[valid,None]/255).astype(np.float32)
 save(dst/'rgb.npz',rgb=rgb,color_observation_count=counts,color_valid=valid,color_source=np.where(direct,1,np.where(valid,2,0)).astype(np.uint8))
 report=dict(status='PASS',frames=400,points=len(xyz),observed_color_points=int(valid.sum()),
  missing_color_points=int((~valid).sum()),direct_color_points=int(direct.sum()),support_recovered_color_points=recovered,depth_tolerance_m=DEPTH_TOLERANCE,
  RGB_method='visibility-gated mean RGB, then for still-uncolored points real RGB from exact historical raw-depth samples assigned by the frozen strict surface projection',
  missing_attribute_policy='exclude unsupported points from completion; preserve their original labels',
  GT_used=False,input_sha256=inputs,source_geometry_sha256=digest(xyz),output_sha256=sha(dst/'rgb.npz'))
 dump(dst/'rgb.json',report)
if __name__=='__main__':colors(sys.argv[1])
