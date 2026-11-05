"""Depth-checked backward mask transport; unobserved labels remain unknown."""
import numpy as np

DEPTH_TOLERANCE_M = .02
ROUNDTRIP_PIXELS = 1.5


def source_cloud(mask, depth, pose, camera):
    y,x=np.nonzero(mask);z=depth[y,x].astype(np.float64)
    p=np.stack(((x-camera['cx'])*z/camera['fx'],(y-camera['cy'])*z/camera['fy'],z),axis=1)
    return dict(world=p@pose[:3,:3].T+pose[:3,3], x=x,y=y,valid=np.isfinite(z)&(z>0),pose=pose)


def cloud_visibility(cloud, target_depth, target_pose, camera):
    h,w=target_depth.shape;inv=np.linalg.inv(target_pose)
    p=cloud['world']@inv[:3,:3].T+inv[:3,3];z=p[:,2];safe=np.where(z>0,z,1.)
    u=np.rint(camera['fx']*p[:,0]/safe+camera['cx']).astype(np.int64)
    v=np.rint(camera['fy']*p[:,1]/safe+camera['cy']).astype(np.int64)
    valid=cloud['valid']&(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
    uc,vc=np.clip(u,0,w-1),np.clip(v,0,h-1);dz=target_depth[vc,uc]
    valid &= np.isfinite(dz)&(dz>0)&(np.abs(dz-z)<=DEPTH_TOLERANCE_M)
    back=np.stack(((uc-camera['cx'])*dz/camera['fx'],(vc-camera['cy'])*dz/camera['fy'],dz),axis=1)
    tf=np.linalg.inv(cloud['pose'])@target_pose;q=back@tf[:3,:3].T+tf[:3,3];qz=np.where(q[:,2]>0,q[:,2],1.)
    bu=camera['fx']*q[:,0]/qz+camera['cx'];bv=camera['fy']*q[:,1]/qz+camera['cy']
    valid &= (q[:,2]>0)&((bu-cloud['x'])**2+(bv-cloud['y'])**2<=ROUNDTRIP_PIXELS**2)
    return int(valid.sum())/len(valid)


def rank_views(poses, arrived, source_frames, primary_frame):
    """Minimax pose distance, before any mask relations or outcomes are examined."""
    sources = sorted(set(source_frames))
    rows = []
    for frame in arrived:
        t = poses[frame]
        ds = []
        for f in sources:
            s = poses[f]
            translation = float(np.linalg.norm(t[:3, 3] - s[:3, 3]))
            angle = float(np.degrees(np.arccos(np.clip((np.trace(t[:3, :3].T @ s[:3, :3]) - 1) / 2, -1, 1))))
            score = translation / .1 + angle / 10
            ds.append(dict(frame=f, translation_m=translation, rotation_deg=angle, score=score))
        rows.append(dict(frame=int(frame), max_score=max(d['score'] for d in ds),
                         mean_score=float(np.mean([d['score'] for d in ds])), sources=ds))
    rows.sort(key=lambda r: (round(r['max_score'], 10), round(r['mean_score'], 10),
                             r['frame'] != primary_frame, r['frame']))
    return rows


def correspondence(depth_from, depth_to, pose_from, pose_to, camera):
    """For every from-pixel, locate its mutually depth-consistent to-pixel."""
    h, w = depth_from.shape
    if depth_to.shape != (h, w):
        raise ValueError('Depth shape mismatch')
    yy, xx = np.indices((h, w), dtype=np.float64)
    z = depth_from.ravel()
    points = np.stack(((xx.ravel() - camera['cx']) * z / camera['fx'],
                       (yy.ravel() - camera['cy']) * z / camera['fy'], z), axis=1)
    tf = np.linalg.inv(pose_to) @ pose_from
    p = points @ tf[:3, :3].T + tf[:3, 3]
    pz = p[:, 2]
    safe = np.where(pz > 0, pz, 1.)
    u = np.rint(camera['fx'] * p[:, 0] / safe + camera['cx']).astype(np.int64)
    v = np.rint(camera['fy'] * p[:, 1] / safe + camera['cy']).astype(np.int64)
    valid = np.isfinite(z) & (z > 0) & (pz > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uc, vc = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    dz = depth_to[vc, uc]
    valid &= np.isfinite(dz) & (dz > 0) & (np.abs(dz - pz) <= DEPTH_TOLERANCE_M)
    # Measured target depth must also return to the original pixel neighborhood.
    back = np.stack(((uc - camera['cx']) * dz / camera['fx'],
                     (vc - camera['cy']) * dz / camera['fy'], dz), axis=1)
    inv = np.linalg.inv(tf)
    q = back @ inv[:3, :3].T + inv[:3, 3]
    qz = np.where(q[:, 2] > 0, q[:, 2], 1.)
    bu = camera['fx'] * q[:, 0] / qz + camera['cx']
    bv = camera['fy'] * q[:, 1] / qz + camera['cy']
    valid &= (q[:, 2] > 0) & ((bu - xx.ravel()) ** 2 + (bv - yy.ravel()) ** 2 <= ROUNDTRIP_PIXELS ** 2)
    return (vc * w + uc).reshape(h, w), valid.reshape(h, w)
