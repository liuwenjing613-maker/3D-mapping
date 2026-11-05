"""Apply only archived trusted-partition hooks to cloned unified online sources."""
from pathlib import Path
import ast, hashlib, json
C=Path(__file__).resolve().parent
OLD=Path('/home/chenkejun/CVPR/experiments/online_v2_trusted_partition_20261027/full_run/room2')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(text,old,new,name):
 n=text.count(old)
 if n!=1:raise RuntimeError(f'{name}: expected exactly one source anchor, got {n}: {old[:120]}')
 return text.replace(old,new,1)
report={}
for scene in SCENES:
 root=C/'online'/scene
 if not root.is_dir():raise RuntimeError(f'Clone not ready: {root}')
 edits={}
 name='map_continue.py';p=root/name;s=p.read_text();before=sha(p)
 s=once(s,'from one_vote import effective_keys','from one_vote import effective_keys\nfrom trusted_online import TrustedPartition',name)
 s=once(s,"'inputs/scene/results','inputs/masks']:","'inputs/scene/results','inputs/masks','partition_layers']:",name)
 old="""            points,local,stats=project_frame_regions(frame,tree,pixel_stride=2,max_distance_m=.015,workers=4)
            table=np.full(int(frame.mask_local.max())+1,-1,np.int64)
            for lid in np.unique(local):table[lid]=lookup[(frame.frame_id,int(lid))]"""
 new="""            layer=np.asarray(Image.open(R/'partition_layers'/f'f{frame.frame_id:06d}.png'),np.uint16)
            assert layer.shape==frame.mask_local.shape
            assert int(layer.max())<base
            points,local,stats=project_frame_regions(replace(frame,mask_local=layer),tree,pixel_stride=2,max_distance_m=.015,workers=4)
            table=np.arange(base,dtype=np.int64)"""
 s=once(s,old,new,name)
 # Image is needed inside strict_surface as well as main.
 s=once(s,'from scipy.sparse import save_npz','from scipy.sparse import save_npz\nfrom PIL import Image',name)
 old="""  geometry_duplicate_points=len(old_xyz)-len(np.unique(old_xyz,axis=0))
  grid_restore_error=float(dist.max());assert grid_restore_error==0."""
 new="""  geometry_duplicate_points=len(old_xyz)-len(np.unique(old_xyz,axis=0))
  region_base=R.parents[2]
  regions=[]
  for stage in (x for x in (50,100,200,300) if x<=PRIOR_COUNT):
   rp=region_base/str(stage)/GROUP/'repair/e1/region.npz'
   sp=region_base/str(stage)/GROUP/'mapping/baseline/native/instance_surface.npz'
   fixed_hashes[str(rp)]=sha(rp);fixed_hashes[str(sp)]=sha(sp)
   with np.load(rp) as z: owners=z['owner'].copy()
   with np.load(sp) as z: region_xyz=z['xyz_m'].copy()
   assert len(owners)==len(region_xyz)
   assert np.all(np.isfinite(region_xyz))
   assert int(owners.max(initial=0))<=REPAIR_MAX_ID
   regions.append((region_xyz,owners))
  partition=TrustedPartition(restored_xyz,regions)
  grid_restore_error=float(dist.max());assert grid_restore_error==0."""
 s=once(s,old,new,name)
 s=once(s,"  grid_restore_error=0.;geometry_duplicate_points=0","  grid_restore_error=0.;geometry_duplicate_points=0;partition=TrustedPartition(np.empty((0,3)),[])",name)
 s=once(s,' continuation=[];frame_times=[];surface_rebuild_frames=[]',' continuation=[];frame_times=[];surface_rebuild_frames=[];partition_audit=[]',name)
 s=once(s,'  current=associator.process_frame(frame,newobs);assert len(current)==len(newobs)','''  current=associator.process_frame(frame,newobs);assert len(current)==len(newobs)
  layer,pa=partition.apply(frame,newobs,current,associator)
  assert layer.shape==frame.mask_local.shape and layer.dtype==np.uint16
  assert int(layer.max())<associator.next_instance_id
  Image.fromarray(layer).save(R/'partition_layers'/f'f{fid:06d}.png')
  partition_audit.extend(dict(frame_id=fid,**a) for a in pa)''',name)
 s=once(s," association_export_seconds=time.perf_counter()-tick"," write(R/'partition_audit.json',partition_audit)\n association_export_seconds=time.perf_counter()-tick",name)
 s=once(s,"  strict_same_frame_point_at_most_one_vote=True,all_prior_layers_retained=True,","  trusted_partition_enabled=True,trusted_partition_source='archived_20261027',trusted_region_cutoff=PRIOR_COUNT,\n  trusted_association_feedback=True,partition_layers_persisted=True,\n  strict_same_frame_point_at_most_one_vote=True,all_prior_layers_retained=True,",name)
 edits[name]=(s,before)
 name='map_support.py';p=root/name;s=p.read_text();before=sha(p)
 anchor=" cache=[str(previous.resolve()),str(Path(prior['checkpoint']).parent.parent),str(previous.resolve())]"
 replacement=anchor+"""
 # All trusted regions belong to committed prefixes of this branch. The first
 # checkpoint may be an immutable baseline symlink; authorize its resolved path.
 for n in (50,100,200,300):
  if n<=PRIOR_COUNT:
   prefix=ROOT.parents[1]/str(n)/GROUP
   cache.extend([str(prefix),str(prefix.resolve())])"""
 s=once(s,anchor,replacement,name)
 s=once(s,"guard['allowed_source_files'] += [str(p) for p in","guard['allowed_source_files'] += [str(p.resolve(strict=True)) for p in",name)
 edits[name]=(s,before)
 name='scene_continue.py';p=root/name;s=p.read_text();before=sha(p)
 old="""    table=np.zeros(int(frame.mask_local.max())+1,np.uint16)
    for mid in np.unique(frame.mask_local):
     if mid:
      pid=self.lookup[(fid,int(mid))];assert 0<pid<=65535
      table[mid]=pid
    self._layers[fid]=[table[frame.mask_local]]"""
 new="""    path=B/'partition_layers'/f'f{fid:06d}.png'
    image=np.asarray(Image.open(path),np.uint16).copy()
    assert image.shape==frame.mask_local.shape
    self.source_hashes[str(path)]=digest(path)
    self._layers[fid]=[image]"""
 s=once(s,old,new,name);edits[name]=(s,before)
 name='commit_continue.py';p=root/name;s=p.read_text();before=sha(p)
 s=once(s,"                assert old.observation_id in original_observations\n                observation = original_observations[old.observation_id]","                original_id = old.observation_id.split('/partition-id',1)[0]\n                assert original_id in original_observations\n                observation = original_observations[original_id]",name)
 s=once(s,"                pixels, voxels = sample_pixels(frame, observation, associator)\n                layer_index = 0","                pixels, voxels = sample_pixels(frame, observation, associator)\n                if '/partition-id' in old.observation_id:\n                    select = old_layers[0].ravel()[pixels] == old.instance_id\n                    pixels, voxels = pixels[select], voxels[select]\n                layer_index = 0",name)
 edits[name]=(s,before)
 # Validate every result before publishing this scene's changes.
 for name,(s,before) in edits.items():ast.parse(s,filename=str(root/name))
 report[scene]={}
 for name,(s,before) in edits.items():
  p=root/name;p.write_text(s);report[scene][name]={'before_sha256':before,'after_sha256':sha(p)}
 for name in ['trusted_online.py','trusted_partition.py']:
  p=root/name;p.write_bytes((OLD/name).read_bytes());ast.parse(p.read_text(),filename=str(p))
  assert sha(p)==sha(OLD/name)
  report[scene][name]={'archived_source':str(OLD/name),'sha256':sha(p)}
(C/'policy_patch_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print('PATCHED',','.join(SCENES))
