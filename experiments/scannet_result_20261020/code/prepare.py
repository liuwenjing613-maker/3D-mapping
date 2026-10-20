"""Calibrate cached ScanNet RGB; retain source IDs, masks, poses and fixed TSDF."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import cv2
from PIL import Image
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
OLD=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_scannet_engineering_fix_20261008')
CFG=Path('/home/chenkejun/CVPR/experiments/p1a1_scannet_engineering_fix_20261008/configs')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False))
def link(p,target):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.is_symlink():assert p.resolve()==target.resolve()
    elif p.exists():raise FileExistsError(p)
    else:p.symlink_to(target.resolve(),target_is_directory=target.is_dir())
def main(scene):
    out=R/'inputs'/scene;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    original=CFG/(scene+'_engineering.json');cfg=json.loads(original.read_text())
    source=Path(cfg['source']['scene_root']);fids=cfg['frame_ids']
    assert fids==sorted(int(p.stem) for p in (source/'color').glob('*.jpg'))
    intrinsic=source/'intrinsic'
    kd=np.loadtxt(intrinsic/'intrinsic_depth.txt')[:3,:3]
    kc=np.loadtxt(intrinsic/'intrinsic_color.txt')[:3,:3]
    for sensor in ['depth','color']:assert np.allclose(np.loadtxt(intrinsic/f'extrinsic_{sensor}.txt'),np.eye(4))
    h,w=cfg['camera']['height'],cfg['camera']['width'];v,u=np.indices((h,w),dtype=np.float64)
    rays=np.stack((u,v,np.ones_like(u)),axis=-1)@(kc@np.linalg.inv(kd)).T
    mx=(rays[:,:,0]/rays[:,:,2]).astype(np.float32);my=(rays[:,:,1]/rays[:,:,2]).astype(np.float32)
    poses=np.full((max(fids)+1,4,4),np.nan)
    hashes={str(original):sha(original)}
    for p in intrinsic.glob('*.txt'):hashes[str(p)]=sha(p)
    frames=[]
    for fid in fids:
        rgbp=source/cfg['source']['rgb_pattern'].format(frame=fid)
        depp=source/cfg['source']['depth_pattern'].format(frame=fid)
        maskp=Path(cfg['source']['mask_root'])/cfg['source']['mask_pattern'].format(frame=fid)
        posep=source/'pose'/f'{fid}.txt'
        for p in [rgbp,depp,maskp,posep]:hashes[str(p)]=sha(p)
        poses[fid]=np.loadtxt(posep);assert np.isfinite(poses[fid]).all()
        rgb=cv2.cvtColor(cv2.remap(cv2.imread(str(rgbp)),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),cv2.COLOR_BGR2RGB)
        target=out/'results'/f'frame{fid:06d}.png';target.parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(target)
        assert np.array_equal(np.asarray(Image.open(target)),rgb)
        link(out/'results'/f'depth{fid:06d}.png',depp)
        link(out/'masks'/f'frame{fid:06d}.png',maskp)
        frames.append(dict(frame=fid,input_sha256=sha(target),RGB_sha256=sha(target),mask_sha256=sha(maskp),
                           original_RGB_sha256=sha(rgbp),depth_sha256=sha(depp),pose_sha256=sha(posep)))
    np.savetxt(out/'traj.txt',poses.reshape(-1,4),fmt='%.17g')
    (out/'masks/frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in frames))
    cfg['source']=dict(scene_root=str(out),mask_root=str(out/'masks'),rgb_pattern='results/frame{frame:06d}.png',
                       depth_pattern='results/depth{frame:06d}.png',mask_pattern='frame{frame:06d}.png',
                       mask_metadata='frames.jsonl',trajectory='traj.txt')
    cfg['adaptation']=dict(original_config=str(original),original_config_sha256=sha(original),RGB='exact calibrated cv2.remap, lossless PNG',
        tracking_frame_ids=fids,missing_frame_policy='never interpolate or invent missing frames',original_frame_ids_preserved=True)
    save(C/'configs'/(scene+'.json'),cfg)
    link(out/'configs/raw.json',C/'configs'/(scene+'.json'))
    link(out/'geometry/surface.ply',OLD/scene/'shared_geometry/surface.ply')
    link(R/'compat/native'/scene/'P1-A1/association',OLD/scene/'combined/association')
    save(out/'complete.json',dict(status='PASS',scene=scene,frame_count=len(fids),frame_ids=fids,
        source_hashes=hashes,aligned_rgb_hashes={str(out/'results'/f'frame{x["frame"]:06d}.png'):x['RGB_sha256'] for x in frames},
        fixed_geometry_sha256=sha(out/'geometry/surface.ply'),mapping_GT_used=False))
    print('PREPARED',scene,len(fids),flush=True)
if __name__=='__main__':main(sys.argv[1])
