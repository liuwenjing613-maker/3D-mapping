from pathlib import Path
import sys,os,json,hashlib
import numpy as np,cv2
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results')/C.name
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/p1a1_scannet_20261008/snapshot/revisable_instance_map/src')
from revisable_instance_map.scannet_frame_io import color_coordinates
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for scene in ['scene0011_00','scene0050_00']:
 root=Path('/data/chenkejun/CVPR/datasets/MaskClustering/scannet/processed')/scene
 dst=O/'inputs'/scene;dst.mkdir(parents=True,exist_ok=True)
 if (dst/'selection.json').exists():continue
 valid=[int(p.stem) for p in sorted((root/'depth').glob('*.png'),key=lambda p:int(p.stem)) if np.isfinite(np.loadtxt(root/'pose'/(p.stem+'.txt'))).all()]
 selected=np.asarray(valid)[np.linspace(0,len(valid)-1,200,dtype=int)].tolist()
 K=np.loadtxt(root/'intrinsic/intrinsic_depth.txt')[:3,:3]
 cam=dict(width=640,height=480,fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],depth_png_units_per_meter=1000)
 infer=dict(dataset='ScanNet',scene=scene,frame_ids=selected,camera=cam,source=dict(scene_root=str(root),mask_root=str(dst/'raw_masks')))
 dump(dst/'inference_config.json',infer)
 records=[];poses=np.tile(np.eye(4,dtype=np.float32),(996,1,1));intr=np.tile(K,(996,1,1))
 results=dst/'input/results';results.mkdir(parents=True,exist_ok=True)
 mx,my=color_coordinates(root,640,480)
 for j,raw in enumerate(selected):
  fid=j*5;dp=root/'depth'/f'{raw}.png';rp=root/'color'/f'{raw}.jpg';pp=root/'pose'/f'{raw}.txt'
  depth=cv2.imread(str(dp),-1);assert depth.shape==(480,640)
  rgb=cv2.remap(cv2.imread(str(rp)),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  png=results/f'frame{fid:06d}.png';assert cv2.imwrite(str(png),rgb)
  (results/f'frame{fid:06d}.jpg').symlink_to(png)
  (results/f'depth{fid:06d}.png').symlink_to(dp)
  poses[fid]=np.loadtxt(pp).astype(np.float32)
  records.append(dict(frame=fid,original_frame=raw,rgb_sha256=sha(rp),depth_sha256=sha(dp),pose_sha256=sha(pp),aligned_rgb_sha256=sha(png)))
 np.savetxt(dst/'input/traj.txt',poses.reshape(-1,16),fmt='%.17g');np.savetxt(dst/'input/intrinsics.txt',intr.reshape(-1,9),fmt='%.17g')
 cfg=json.loads(Path('/data/chenkejun/CVPR/results/scannetpp_v2_full_stride5_20261030/5ee7c22ba0/config.json').read_text())
 cfg.update(scene=scene,source_dataset='ScanNet v2 / ScanNet200 labels; Replica-layout adapter only')
 cfg['camera'].update(cam);cfg['frame_selection']=dict(start=0,stop_exclusive=996,stride=5)
 cfg['source'].update(scene_root=str(dst/'input'),mask_root=str(dst/'masks'))
 dump(dst/'config.json',cfg)
 dump(dst/'cam_params.json',dict(camera=dict(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],h=480,w=640,scale=1000)))
 dump(dst/'selection.json',dict(status='PASS',scene=scene,original_frames=selected,rows=records,frame_alias_note='Virtual 0:5:1000 IDs; original ScanNet frame IDs preserved in each row; no temporal interpolation',color_depth_alignment='Existing calibrated intrinsics reprojection; sensor extrinsics checked by original helper'))
 print('INPUT_READY',scene,len(selected),flush=True)
