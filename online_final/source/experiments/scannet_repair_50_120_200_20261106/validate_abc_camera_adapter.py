from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
from PIL import Image
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
S=C.parent/'unified_full_depth_20261101'
def module(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
reports=[]
for method in ['main','online']:
    old=module(S/'policies'/method/'instance_repair/seed_abc_projection.py','original_'+method)
    new=module(C/'policies'/method/'instance_repair/seed_abc_projection.py','adapted_'+method)
    cam=dict(fx=9.,fy=10.,cx=4.,cy=3.);pose=np.eye(4);target=pose.copy();target[0,3]=.01
    depth=np.ones((8,12),np.float64);depth[2,2]=0;mask=depth>0
    for dst in [pose,target]:
        a,av=old.correspondence(depth,depth,pose,dst,cam);b,bv=new.correspondence(depth,depth,pose,dst,cam)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(av,bv)
        assert old.cloud_visibility(old.source_cloud(mask,depth,pose,cam),depth,dst,cam)==new.cloud_visibility(new.source_cloud(mask,depth,pose,cam),depth,dst,cam)
    # Analytic zoom and principal-point shift on a flat plane: u=2*x-2, v=2*y-1.
    a=dict(fx=8.,fy=9.,cx=3.,cy=3.);b=dict(fx=16.,fy=18.,cx=4.,cy=5.)
    depth=np.ones((8,12),np.float64);ix,known=new.correspondence(depth,depth,pose,pose,a,b)
    yy,xx=np.indices(depth.shape);u=2*xx-2;v=2*yy-1;inside=(u>=0)&(u<12)&(v>=0)&(v<8)
    np.testing.assert_array_equal(known,inside);np.testing.assert_array_equal(ix[inside],(v*12+u)[inside])
    real=[]
    for scene in ['scene0011_00','scene0050_00']:
        root=O/'online'/scene/'50/B/mapping/inputs/scene'
        poses=np.loadtxt(root/'traj.txt').reshape(-1,4,4);ks=np.loadtxt(root/'intrinsics.txt').reshape(-1,3,3)
        for sf,tf in [(0,245),(245,0)]:
            ds=np.asarray(Image.open(root/f'results/depth{sf:06d}.png'),np.float64)/1000
            dt=np.asarray(Image.open(root/f'results/depth{tf:06d}.png'),np.float64)/1000
            def camera(k):return dict(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2])
            cs,ct=camera(ks[sf]),camera(ks[tf]);mask=np.zeros(ds.shape,bool)
            pixels=np.flatnonzero(ds>0)[::7919];mask.ravel()[pixels]=True
            cloud=new.source_cloud(mask,ds,poses[sf],cs);fraction=new.cloud_visibility(cloud,dt,poses[tf],ct)
            inv_to=np.linalg.inv(poses[tf]);inv_from=np.linalg.inv(poses[sf]);valid=0
            for flat in pixels:
                y,x=divmod(int(flat),640);z=ds[y,x]
                world=poses[sf]@np.array([(x-cs['cx'])*z/cs['fx'],(y-cs['cy'])*z/cs['fy'],z,1.])
                q=inv_to@world
                if q[2]<=0:continue
                u,v=np.rint([ct['fx']*q[0]/q[2]+ct['cx'],ct['fy']*q[1]/q[2]+ct['cy']]).astype(int)
                if not (0<=u<640 and 0<=v<480):continue
                zt=dt[v,u]
                if not (np.isfinite(zt) and zt>0 and abs(zt-q[2])<=.02):continue
                back=inv_from@(poses[tf]@np.array([(u-ct['cx'])*zt/ct['fx'],(v-ct['cy'])*zt/ct['fy'],zt,1.]))
                if back[2]<=0:continue
                uv=np.array([cs['fx']*back[0]/back[2]+cs['cx'],cs['fy']*back[1]/back[2]+cs['cy']])
                valid+=bool(np.sum((uv-[x,y])**2)<=1.5**2)
            assert fraction==valid/len(pixels),(scene,sf,tf,fraction,valid,len(pixels))
            real.append(dict(scene=scene,source=sf,target=tf,checked_pixels=len(pixels),visible_fraction=fraction))
    reports.append(dict(method=method,equal_intrinsics_bit_exact=True,analytic_unequal_intrinsics_exact=True,real_two_scene_bidirectional_scalar_checks=real))
result=dict(status='PASS',time=time.time(),reports=reports,GT_used=False,thresholds_unchanged=True,
    limitation='Projection and API validation only; full candidate-to-repair handoff still requires runtime validation')
p=C/'adapter_audits/abc_camera/validation.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
