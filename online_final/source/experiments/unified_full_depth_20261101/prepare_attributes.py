"""Identical causal native attributes for every compared method."""
from alignment_common import *
import sys,cv2,subprocess,concurrent.futures,traceback
from scipy.spatial import cKDTree
sys.path.insert(0,str(OLD_C))
from strict_votes import nearest,RADIUS
import importlib.util
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
runner=module('_filled_native_runner',FILL_C/'native_runner.py')
# This function does not load data or mutate any existing output.
sys.path.insert(0,str(FILL_C))
cf=module('_original_colors',FILL_C/'color_fusion.py')
sys.path.remove(str(FILL_C))
def attributes(scene,n):
 dst=O/'attributes'/scene/str(n);dst.mkdir(parents=True,exist_ok=True)
 if (dst/'complete.json').exists():
  done=read(dst/'complete.json')
  for p,h in done['output_sha256'].items():assert sha(p)==h
  return scene,n
 native=O/'native'/scene/str(n)
 start=time.time()
 while n!=400 and not (native/'verified.json').exists():
  if (O/'native_failure.json').exists():raise RuntimeError('Native producer failed')
  if time.time()-start>3600:raise TimeoutError(str(native))
  time.sleep(5)
 assert n==400 or read(native/'verified.json')['status']=='PASS'
 state=native/'tsdf.npz';surface=native/'surface.npz'
 if not (dst/'normals.json').exists():
  with (dst/'normals.log').open('w') as f:
   subprocess.run(runner.container(['/usr/bin/python3','-B','/experiment/restore_normals.py',str(state),str(surface),str(dst/'normals.npz')]),stdout=f,stderr=subprocess.STDOUT,check=True)
 a=load(dst/'normals.npz');xyz=a['xyz_m'];assert a['normal_valid'].shape==(len(xyz),)
 rows=[json.loads(x) for x in (OLD/'native'/f'{scene}_400_trace/prefix_audit.jsonl').read_text().splitlines()][:n]
 baseline=[json.loads(x) for x in (OLD/'ours'/f'{scene}_400_main/audit.jsonl').read_text().splitlines()][:n]
 assert [r['frame_id'] for r in rows]==list(range(0,n*5,5))
 root=Path('/home/chenkejun/beauty/conceptgraphs/data/Replica')/scene/'results'
 sums=np.zeros((len(xyz),3),np.float64);counts=np.zeros(len(xyz),np.int32);source={}
 def frame(row,old):
  fid=row['frame_id'];rp=root/f'frame{fid:06d}.jpg';dp=root/f'depth{fid:06d}.png'
  for p in [rp,dp]:
   h=sha(p);assert h==old['input_hashes'][str(p)];source[str(p)]=h
  depth=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED).astype(np.float32)/6553.5
  rgb=cv2.cvtColor(cv2.imread(str(rp)),cv2.COLOR_BGR2RGB)
  return depth,rgb,np.asarray(row['pose']),np.asarray(row['intrinsics'])
 for row,old in zip(rows,baseline):
  depth,rgb,pose,K=frame(row,old);ix,v,u=cf.visible_samples(xyz,pose,K,depth)
  sums[ix]+=rgb[v,u];counts[ix]+=1
 direct=counts>0;missing=~direct;targets=np.flatnonzero(missing)
 if len(targets):
  nearby=cKDTree(xyz[targets]);full=cKDTree(xyz)
  lookup=np.full(len(xyz),-1,np.int32);lookup[targets]=np.arange(len(targets))
  for row,old in zip(rows,baseline):
   depth,rgb,pose,K=frame(row,old)
   # Every valid original depth sample is physical evidence, independent of any predicted mask.
   world,colors=cf.sampled_rgb_world(depth,np.ones(depth.shape,np.uint8),rgb,pose,K)
   selected=np.flatnonzero(np.isfinite(nearby.query(world,workers=2,distance_upper_bound=RADIUS)[0]))
   if not len(selected):continue
   point,ok=nearest(full,world[selected]);ok &= missing[np.minimum(point,len(xyz)-1)]
   dest=lookup[point[ok]];value=colors[selected[ok]]
   counts[targets]+=np.bincount(dest,minlength=len(targets)).astype(np.int32)
   for j in range(3):sums[targets,j]+=np.bincount(dest,weights=value[:,j],minlength=len(targets))
 valid=counts>0;rgb=np.zeros((len(xyz),3),np.float32);rgb[valid]=(sums[valid]/counts[valid,None]/255).astype(np.float32)
 save(dst/'rgb.npz',rgb=rgb,color_observation_count=counts,color_valid=valid,color_source=np.where(direct,1,np.where(valid,2,0)).astype(np.uint8))
 dump(dst/'inputs.json',dict(status='PASS',scene=scene,frame_ids=[x['frame_id'] for x in rows],source_sha256=source))
 dump(dst/'complete.json',dict(status='PASS',scene=scene,frames=n,raw_cutoff=(n-1)*5,points=len(xyz),
  geometry_sha256=digest(xyz),GT_used=False,direct_points=int(direct.sum()),support_recovered_points=int((valid&~direct).sum()),
  missing_color_points=int((~valid).sum()),valid_normals=int(a['normal_valid'].sum()),
  source_policy='Original visibility mean; fallback actual RGB from every valid stride-2 depth sample, no mask or ID dependency',
  output_sha256={str(p):sha(p) for p in [dst/'normals.npz',dst/'rgb.npz',dst/'inputs.json']},seconds=time.time()-start))
 print('ATTRIBUTES_READY',scene,n,len(xyz),int((~valid).sum()),flush=True);return scene,n
def main():
 try:
  jobs=[(s,n) for n in [50,100,200,300,400] for s in SCENES]
  count=len(jobs)
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   running={}
   while jobs or running:
    for job in list(jobs):
     scene,n=job
     if len(running)>=6:break
     if n==400 or (O/'native'/scene/str(n)/'verified.json').exists():
      running[pool.submit(attributes,*job)]=job;jobs.remove(job)
    if running:
     done,_=concurrent.futures.wait(running,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
     for f in done:f.result();del running[f]
    else:time.sleep(5)
    if (O/'native_failure.json').exists():raise RuntimeError('Native producer failed')
  dump(O/'attributes_complete.json',dict(status='PASS',jobs=count))
 except Exception:dump(O/'attributes_failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
