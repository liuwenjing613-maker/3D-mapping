from pathlib import Path
import json,hashlib,shutil,ast
import numpy as np
C=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix80_20261021/3d')
W=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/review/3d')
C.mkdir(parents=True,exist_ok=True);(W/'data').mkdir(parents=True,exist_ok=True);(W/'vendor').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
for n in [80]:
 R=Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021')
 original=json.loads((R/'ply/manifest.json').read_text())
 source=R/'baseline/final/instance_surface.npz'
 with np.load(source) as z:xyz=z['xyz_m'];rgb=z['rgb'];labels=z['instance_id']
 with np.load(R/'baseline/surface_evidence.npz') as z:
  np.testing.assert_array_equal(xyz,z['xyz_m']);state=z['state']
 a=np.empty(len(xyz),dtype=dtype)
 for j,k in enumerate(['x','y','z']):a[k]=xyz[:,j]
 rgb8=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
 for j,k in enumerate(['red','green','blue']):a[k]=rgb8[:,j]
 a['instance_id']=labels;a['state']=state;a['surface_index']=np.arange(len(xyz),dtype=np.int32)
 name=f'room0_prefix{n}_full_rgb.ply';p=R/'ply'/name
 header=('ply\nformat binary_little_endian 1.0\ncomment Actual causal prefix TSDF surface; RGB rounded to uint8; coordinates and IDs preserved\n'
  +f'element vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nproperty uchar state\nproperty int surface_index\nend_header\n').encode()
 with p.open('wb') as f:f.write(header);a.tofile(f)
 check=np.memmap(p,dtype=dtype,mode='r',offset=len(header),shape=(len(a),))
 np.testing.assert_array_equal(check,a)
 covered=[]
 for item in original['exports']:
  pp=R/'ply'/item['file'];assert sha(pp)==item['sha256']
  with pp.open('rb') as f:
   offset=0
   while True:
    line=f.readline();offset+=len(line)
    if line==b'end_header\n':break
  old=np.memmap(pp,dtype=dtype,mode='r',offset=offset,shape=(item['points'],))
  idx=old['surface_index'];covered.append(idx)
  for k in ['x','y','z','instance_id','state','surface_index']:np.testing.assert_array_equal(old[k],a[k][idx])
  link=W/'data'/pp.name
  if not link.exists():link.symlink_to(pp)
 np.testing.assert_array_equal(np.sort(np.concatenate(covered)),np.arange(len(a)))
 link=W/'data'/name
 if not link.exists():link.symlink_to(p)
 cfg=json.loads(Path(f'/home/chenkejun/CVPR/experiments/online_room0_prefix{n}_20261021/config.json').read_text())
 poses=np.loadtxt(Path(cfg['source']['scene_root'])/cfg['source']['trajectory']).reshape(-1,4,4)
 fids=list(range(0,n*5,5));cameras=[]
 for fid in fids:
  t=poses[fid];cameras.append(dict(frame=fid,eye=t[:3,3].tolist(),forward=t[:3,2].tolist(),up=(-t[:3,1]).tolist()))
 ids,counts=np.unique(labels[labels>0],return_counts=True)
 row=dict(frames=n,last_frame=fids[-1],points=len(a),assigned=int(np.sum(labels>0)),instances=len(ids),
  unassigned=int(np.sum(labels<=0)),states={name:int(np.sum((labels<=0)&(state==k))) for name,k in [('U',0),('T',1),('C',3)]},
  bbox=[xyz.min(0).tolist(),xyz.max(0).tolist()],instance_counts={str(int(i)):int(c) for i,c in zip(ids,counts)},
  ply='data/'+name,bytes=p.stat().st_size,sha256=sha(p),cameras=cameras,
  downloads=[dict(title='完整 RGB 点云',file='data/'+name),dict(title='实例着色 PLY',file=f'data/room0_prefix{n}_instances.ply'),dict(title='剩余 U/T/C PLY',file=f'data/room0_prefix{n}_three_states.ply')],
  source_sha256=sha(source),original_PLY_geometry_and_ID_equality=True,all_surface_indices_present_exactly_once=True)
 rows.append(row)
 (R/'ply/full_rgb_audit.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
vendor=Path('/home/chenkejun/CVPR/experiments/e1_process_player_20261016/player/vendor')
for name in ['three.module.js','three.core.js','OrbitControls.js']:
 assert (vendor/name).is_file()
 target=W/'vendor'/name
 if not target.exists():target.symlink_to(vendor/name)
manifest=dict(scene='room0',maps=rows,coordinate_unit='meter',up_axis='Z',repair_applied=False,GT_used=False,downsampled=False,verified=True)
(W/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(frames=r['frames'],points=r['points'],bytes=r['bytes'],states=r['states']) for r in rows]))
