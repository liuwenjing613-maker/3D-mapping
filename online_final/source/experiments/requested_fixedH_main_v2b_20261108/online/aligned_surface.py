"""Causal access to certified native OVIMAP surfaces and a shared attribute-only fill adapter."""
from alignment_common import *
import sys
from scipy.spatial import cKDTree as _KD
sys.path.insert(0,str(SRC))
from revisable_instance_map.offline_surface_assignment import assign_surface as _assign
class cKDTree(_KD):
 def query(self,x,k=1,*args,**kw):
  if k!=1:return super().query(x,k,*args,**kw)
  d,i=super().query(x,2,*args,**kw);single=np.asarray(x).ndim==1
  dd=np.atleast_2d(d);ii=np.atleast_2d(i);xx=np.atleast_2d(x)
  tied=np.flatnonzero(np.isfinite(dd[:,0])&(dd[:,0]==dd[:,1]))
  for j in tied:
   cs=np.asarray(self.query_ball_point(xx[j],np.nextafter(dd[j,0],np.inf)),np.int64)
   ds=np.sum((self.data[cs]-xx[j])**2,axis=1);ii[j,0]=int(cs[ds==ds.min()].min())
  if single:return dd[0,0],ii[0,0]
  return dd[:,0],ii[:,0]
def scene_name():return os.environ['CVPR_SCENE']
def attribute_root(n=None):
 return O/'attributes'/scene_name()/str(n if n is not None else int(os.environ['CVPR_CHECKPOINT']))
def assign_surface(xyz,normals,rgb,evidence,pair,labels,settings,progress=None):
 root=attribute_root();a=load(root/'normals.npz');co=load(root/'rgb.npz')
 np.testing.assert_array_equal(xyz,a['xyz_m']);np.testing.assert_array_equal(rgb,co['rgb'])
 valid=a['normal_valid']&co['color_valid']
 assert np.all(valid[labels>0]),'Published competitor lacks real RGB or native normal'
 if valid.all():return _assign(xyz,normals,rgb,evidence,pair,labels,settings,progress)
 ix=np.flatnonzero(valid);mapping=np.full(len(xyz),-1,np.int32);mapping[ix]=np.arange(len(ix),dtype=np.int32)
 keep=valid[pair['surface_point_index']]
 pp={k:(mapping[v[keep]] if k=='surface_point_index' else v[keep]) for k,v in pair.items()}
 variants,details,stats=_assign(xyz[ix],normals[ix],rgb[ix],{k:v[ix] for k,v in evidence.items()},pp,labels[ix],settings,progress)
 full={}
 for k,v in variants.items():
  full[k]=labels.copy();full[k][ix]=v
 stats=dict(stats,attribute_adapter_excluded=int((~valid).sum()),excluded_confirmed_points=0)
 return full,dict(subset_to_surface_index=ix,**details),stats
class _Cloud:
 def __init__(self,c):self.c=c
 def to_legacy(self):return self.c
class NativeGeometry:
 def __init__(self,**kwargs):self.integrated_frames=set()
 def restore(self,path):
  state=load(path);self.integrated_frames=set(map(int,state['frames']))
  n=len(self.integrated_frames);assert self.integrated_frames==set(range(0,n*5,5))
  assert str(state['scene'][0])==scene_name()
  assert str(state['geometry_sha256'][0])==read(attribute_root(n)/'complete.json')['geometry_sha256']
 def integrate(self,frame):
  assert frame.frame_id==len(self.integrated_frames)*5
  row=read(O/'frame_meta'/scene_name()/f'f{frame.frame_id:06d}.json')
  np.testing.assert_array_equal(frame.camera_to_world,np.asarray(row['pose']))
  np.testing.assert_array_equal(frame.camera.intrinsic,np.asarray(row['intrinsics']))
  assert digest(frame.depth_m)==row['depth_decoded_sha256']
  assert digest(frame.mask_local.astype(np.int32))==row['mask_decoded_sha256']
  self.integrated_frames.add(frame.frame_id)
  return dict(frame_id=frame.frame_id,native_cached_prefix=True,geometry_sha256=row['geometry_sha256'],
   tsdf_repair_independent=True,actual_integration_performed_by_original_native_producer=True)
 def extract_point_cloud(self,**kwargs):
  import open3d as o3d
  n=len(self.integrated_frames);root=attribute_root(n);m=read(root/'complete.json')
  assert m['frames']==n and m['raw_cutoff']==max(self.integrated_frames)
  for p,h in m['output_sha256'].items():assert sha(p)==h
  a=load(root/'normals.npz');co=load(root/'rgb.npz')
  assert digest(a['xyz_m'])==m['geometry_sha256']
  cloud=o3d.geometry.PointCloud()
  cloud.points=o3d.utility.Vector3dVector(a['xyz_m']);cloud.normals=o3d.utility.Vector3dVector(a['normals']);cloud.colors=o3d.utility.Vector3dVector(co['rgb'])
  return _Cloud(cloud)
 def save(self,path):
  n=len(self.integrated_frames);m=read(attribute_root(n)/'complete.json')
  save(path,frames=np.array(sorted(self.integrated_frames),np.int32),scene=np.array([scene_name()]),geometry_sha256=np.array([m['geometry_sha256']]),format=np.array(['certified_native_prefix_reference_v1']))
