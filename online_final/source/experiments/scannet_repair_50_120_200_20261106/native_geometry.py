from pathlib import Path
import sys,os,json,time,hashlib,itertools,traceback
import numpy as np,cv2
sys.path.insert(0,'/experiment/build')
import consistent_gsm as native
scene,mode=sys.argv[1:3]
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
source=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106/inputs')/scene
cfg=json.loads((source/'config.json').read_text());total={'scene0011_00':200,'scene0050_00':200}[scene]
root=O/'native'/scene/mode;root.mkdir(parents=True,exist_ok=False)
def dump(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
poses=np.loadtxt(source/'input/traj.txt').reshape(-1,4,4);intr=np.loadtxt(source/'input/intrinsics.txt').reshape(-1,3,3)
def frame(fid):
 depth=cv2.imread(str(source/'input/results'/f'depth{fid:06d}.png'),-1).astype(np.float32)/1000
 rgb=cv2.cvtColor(cv2.imread(str(source/'input/results'/f'frame{fid:06d}.png')),cv2.COLOR_BGR2RGB)
 assert depth.shape==(480,640) and rgb.shape==(480,640,3)
 K=intr[fid];pose=poses[fid];assert np.isfinite(K).all() and np.isfinite(pose).all()
 assert np.allclose(pose[3],[0,0,0,1]) and abs(np.linalg.det(pose[:3,:3])-1)<.01
 pc=cv2.rgbd.depthTo3d(depth,K).reshape(-1,3);valid=np.isfinite(pc).all(1)&(pc[:,2]>0)
 return np.ascontiguousarray(pc[valid],dtype=np.float32),np.ascontiguousarray(rgb.reshape(-1,3)[valid]),np.ascontiguousarray(pose,dtype=np.float32),K,depth,valid
cv2.setNumThreads(1)
node=native.GlobalSegmentMap_py(str(root),'Nyu40',False,False,4,2,1,False,3,True,.2,.9)
start=time.time()
try:
 if mode in ['kernel_original','kernel_geometry']:
  pc,col,pose,K,depth,valid=frame(0);pc=np.ascontiguousarray(pc[::17][:24000]);col=np.full(pc.shape,200,np.uint8)
  for i,(points,colors) in enumerate(zip(np.array_split(pc,3),np.array_split(col,3))):
   if mode=='kernel_original':
    lo=points.min(0);hi=points.max(0);box=np.array([[lo[j] if b[j]==0 else hi[j] for j in range(3)] for b in itertools.product([0,1],repeat=3)],np.float32)
    node.insertSegmentsOpen(points,box,i+1,1,np.array([0.,1.],np.float32),1.,1.,pose,True,i);node.integrateFrame();node.clearTemporaryMemory()
   else:
    before=node.cvprLabelFingerprint();node.cvprIntegrateGeometryOnly(points,colors,pose);assert before==node.cvprLabelFingerprint()
   np.savez_compressed(root/f'state{i}.npz',**node.cvprTsdfState())
  dump(root/'complete.json',dict(status='PASS',mode=mode));sys.exit(0)
 count=3 if mode=='pilot' else total
 checks=[1,3,50,120,total]
 baseline_label=node.cvprLabelFingerprint()
 for n,fid in enumerate(range(0,count*5,5),1):
  pc,col,pose,K,depth,valid=frame(fid)
  node.cvprIntegrateGeometryOnly(pc,col,pose)
  row=dict(frame_id=fid,processed_frames=n,depth_decoded_sha256=digest(depth),pose=pose.tolist(),intrinsics=K.tolist(),valid_depth_pixels=int(valid.sum()),submitted_points=len(pc),union_equals_all_valid_depth=True,mask_used=False,GT_used=False,seconds=time.time()-start)
  assert len(pc)==valid.sum()
  with (root/'frame_input_audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if n in checks:
   assert node.cvprLabelFingerprint()==baseline_label
   state=node.cvprTsdfState();raw=node.cvprSurface();xyz,inv=np.unique(raw['xyz'],axis=0,return_inverse=True)
   out=root/str(n);out.mkdir();np.savez_compressed(out/'surface.npz',xyz_m=xyz);np.savez_compressed(out/'tsdf.npz',**state)
   dump(out/'audit.json',dict(status='PASS',frames=n,raw_cutoff=fid,points=len(xyz),geometry_sha256=digest(xyz),GT_used=False,mask_independent=True,label_storage_unchanged=True))
   print('CHECKPOINT',scene,n,len(xyz),round(time.time()-start,2),flush=True)
  dump(root/'progress.json',dict(status='RUNNING',frames=n,total=count,seconds=time.time()-start,pid=os.getpid()))
 dump(root/'complete.json',dict(status='PASS',frames=count,seconds=time.time()-start,GT_used=False,geometry_kernel='native OVIMAP geometry-only',submission='all valid depth samples once per frame, mask-independent',per_frame_intrinsics=True))
except BaseException as e:
 if isinstance(e,SystemExit):raise
 dump(root/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
