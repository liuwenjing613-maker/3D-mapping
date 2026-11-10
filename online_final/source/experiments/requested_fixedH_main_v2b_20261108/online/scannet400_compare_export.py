from pathlib import Path
import numpy as np,json,time,traceback,colorsys,hashlib
R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/scannet400/main/scene0050_00/400/A');O=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/direct_post_sparse/viewer/scannet400');start=time.time()
def dump(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def status(x,**kw):dump(O/'status.json',dict(status=x,elapsed=time.time()-start,**kw))
try:
 status('RUNNING',stage='读取完整后处理点云')
 paths=[Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/scannet400/baseline_postprocess/instance_surface.npz'),R/'postprocess_final/instance_surface.npz'];data=[]
 for p in paths:
  with np.load(p) as z:data.append({k:z[k] for k in ['xyz_m','rgb','instance_id']})
 a,b=data;xyz=a['xyz_m'];np.testing.assert_array_equal(xyz,b['xyz_m']);np.testing.assert_array_equal(a['rgb'],b['rgb']);ids=[a['instance_id'].astype(np.int32),b['instance_id'].astype(np.int32)];n=len(xyz)
 allids=np.unique(np.concatenate(ids));allids=allids[allids>0];adj={int(i):set() for i in allids};ix=np.arange(0,n,8);vox=np.floor(xyz[ix]/.1).astype(np.int32);_,vi=np.unique(vox,axis=0,return_inverse=True)
 for lab in ids:
  pairs=np.unique(np.column_stack((vi,lab[ix])),axis=0);pairs=pairs[pairs[:,1]>0];starts=np.r_[0,np.flatnonzero(np.diff(pairs[:,0]))+1,len(pairs)]
  for l,h in zip(starts[:-1],starts[1:]):
   group=pairs[l:h,1].tolist()
   for x in group:
    if len(group)>1:adj[int(x)].update(int(y) for y in group if y!=x)
 colors=np.unique(np.rint(np.array([colorsys.hsv_to_rgb(h/144,s,v) for h in range(144) for s in [.6,.8,1.] for v in [.75,.9,1.]])*255).astype(np.uint8),axis=0);palette={};used=set()
 # Reuse colors for IDs present in the 200-frame no-repair baseline; new IDs stay distinct.
 ref=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/direct_post_sparse/viewer/scannet200/before.ply')
 with ref.open('rb') as f:hdr=f.read(1024);end=hdr.index(b'end_header\n')+len(b'end_header\n');count=int(next(l.split()[-1] for l in hdr[:end].decode().splitlines() if l.startswith('element vertex ')))
 dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')]);m=np.memmap(ref,dtype=dtype,mode='r',offset=end,shape=(count,));u,ix=np.unique(m['instance_id'],return_index=True)
 for iid,i in zip(u,ix):
  if iid>0 and iid in adj:
   color=np.array([m['red'][i],m['green'][i],m['blue'][i]],dtype=np.uint8);palette[int(iid)]=color;found=np.flatnonzero(np.all(colors==color,axis=1));used.update(map(int,found))
 del m
 for iid in sorted(map(int,allids),key=lambda i:(-len(adj[i]),i)):
  if iid in palette:continue
  near=[palette[j] for j in adj[iid] if j in palette]
  score=((((colors[:,None,:].astype(float)-np.asarray(near)[None,:,:])**2)*[.3,.59,.11]).sum(2).min(1) if near else np.sum((colors.astype(float)-110)**2,axis=1))
  if used:score[list(used)]=-1
  k=int(np.argmax(score));palette[iid]=colors[k];used.add(k)
 rgb=[]
 for lab in ids:
  uniq,inv=np.unique(lab,return_inverse=True);lut=np.asarray([palette[int(x)] if x>0 else [92,102,112] for x in uniq],np.uint8);rgb.append(lut[inv])
 status('RUNNING',stage='导出全密度 PLY',points=n)
 dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')]);views=[]
 for i,arm in enumerate(['before','after']):
  ar=np.empty(n,dtype=dt)
  for j,k in enumerate(['x','y','z']):ar[k]=xyz[:,j]
  for j,k in enumerate(['red','green','blue']):ar[k]=rgb[i][:,j]
  ar['instance_id']=ids[i];header=('ply\nformat binary_little_endian 1.0\ncomment ScanNet scene0050 400 frame full depth final B guarded DBSCAN '+arm+'\nelement vertex '+str(n)+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n').encode();tmp=O/(arm+'.tmp.ply')
  with tmp.open('wb') as f:f.write(header);ar.tofile(f)
  mm=np.memmap(tmp,dtype=dt,mode='r',offset=len(header),shape=(n,));np.testing.assert_array_equal(mm['instance_id'],ids[i]);np.testing.assert_array_equal(mm['x'],xyz[:,0]);del mm;tmp.replace(O/(arm+'.ply'));del ar
  views.append({'instances':int((np.unique(ids[i])>0).sum()),'assigned_points':int((ids[i]>0).sum()),'unassigned_points':int((ids[i]<=0).sum()),'ply':arm+'.ply'})
 tmp=O/'surface.tmp.bin'
 with tmp.open('wb') as f:
  for x in [np.asarray([n],dtype='<u4'),xyz.astype('<f4',copy=False),rgb[0],rgb[1]]:f.write(x.tobytes())
 assert tmp.stat().st_size==4+n*18;tmp.replace(O/'surface.bin')
 metrics=json.loads((R/'evaluation_current_annotation/diagnostic_iou040/result.json').read_text());dump(O/'scene.json',{'status':'PASS','scene':'scene0050_00','count':n,'bbox':[xyz.min(0).tolist(),xyz.max(0).tolist()],'views':views,'full_resolution':True,'geometry_same':True,'colors':'same prediction ID same color; distinct new IDs distinguished; gray unassigned','postprocessing':'final B + guarded DBSCAN','metrics':{arm:{t:{k:v for k,v in metrics['results'][arm][t].items() if k!='matched_gt_ids'} for t in ['0.4','0.5']} for arm in ['before','after']},'GT':'rev153 temporary hybrid point-splat, provisional development evaluation','source_paths':list(map(str,paths))})
 status('PASS',points=n,views=views,full_resolution=True)
except BaseException:status('FAILED',error=traceback.format_exc());raise
