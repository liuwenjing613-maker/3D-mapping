"""Frozen P1-A1 C/U/T candidate selection; no masks, repair or GT required."""
from pathlib import Path
import os,json,hashlib,time
import numpy as np

SCENES=['room0','room2','room1','office0','office1','office2','office3','office4']
GROUPS={'CONFLICT':3,'U':0,'T':1}
PREFIX={'CONFLICT':'C','U':'U','T':'T'}
CODE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('ROI_SELECTION_ROOT','/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/catalog'))
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
OLD=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_u30_dual_repair_20261004')
DEPTH_TOLERANCE=.02

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def core_sha(a):return hashlib.sha256(np.asarray(a,dtype='<i4').tobytes()).hexdigest()
def paths(scene):
 assert scene=='room0'
 return (Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/baseline/final/instance_surface.npz'),Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/baseline/surface_evidence.npz'),
         Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/geometry/review_surface_graph_n02_p08.npz'),Path('/home/chenkejun/CVPR/experiments/online_room0_prefix80_20261021/config.json'))

def project_visible(points,pose,depth,c):
 pc=(points.astype(np.float64)-pose[:3,3])@pose[:3,:3];z=pc[:,2];front=(z>.05)&(z<10)
 uv=np.full((len(points),2),-1,np.int32)
 uv[front]=np.rint(np.column_stack([c['fx']*pc[front,0]/z[front]+c['cx'],c['fy']*pc[front,1]/z[front]+c['cy']])).astype(np.int32)
 inside=front&(uv[:,0]>=0)&(uv[:,0]<c['width'])&(uv[:,1]>=0)&(uv[:,1]<c['height'])
 jj=np.flatnonzero(inside);u,v=uv[jj].T;dep=depth[v,u]
 back=np.column_stack([(u-c['cx'])*dep/c['fx'],(v-c['cy'])*dep/c['fy'],dep])
 good=(dep>0)&(dep<10)&(np.linalg.norm(back-pc[jj],axis=1)<=DEPTH_TOLERANCE)
 valid=np.zeros(len(points),bool);valid[jj[good]]=True
 return valid,uv
def crop_box(uv,w=1200,h=680):
 lo=uv.min(0);hi=uv.max(0)+1;size=np.minimum([w,h],np.maximum(2*(hi-lo),[560,380])).astype(int)
 low=np.clip(np.floor((lo+hi-size)/2).astype(int),0,np.array([w,h])-size)
 return [int(low[0]),int(low[1]),int(low[0]+size[0]),int(low[1]+size[1])]
