"""Export only real, complete 200/300 A/B instance maps for web comparison."""
from pathlib import Path
import argparse,hashlib,json,sys,time,traceback
import numpy as np
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
sys.path.insert(0,str(BASE))
from spatial_instance_palette import build_palette
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
ROOT=TOP/'continuation300';OUT=TOP/'review/compare200_300'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def verify_complete(path,checkpoint,group):
 q=read(path);assert q['status']=='PASS',str(path)
 for key in ['mapping_frames','processed_frames','target_processed_frames','frame_count']:
  if key in q:assert q[key]==checkpoint,(str(path),key,q[key])
 for key in ['raw_cutoff','last_raw_frame','last_frame_id']:
  if key in q:assert q[key]==(checkpoint-1)*5,(str(path),key,q[key])
 if 'group' in q:assert q['group']==group
 return q
def export():
 start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
 maps={};sources={};completed={};pair_keys=[]
 for checkpoint in [200,300]:
  for group in ['A','B']:
   root=ROOT/str(checkpoint)/group;pair=f'{checkpoint}_{group}';pair_keys.append(pair)
   for stage,p in [('mapping',root/'mapping/mapping_complete.json'),('repair',root/'repair/repair_complete.json')]:
    verify_complete(p,checkpoint,group);completed[pair+'_'+stage]=dict(path=str(p),sha256=digest(p))
   before=root/'mapping/baseline';after=root/'repair/partial/whole_to_partial/room0'
   for suffix,mapfile,evfile in [
    ('before',before/'final/instance_surface.npz',before/'surface_evidence.npz'),
    ('after',after/'final/instance_surface.npz',after/'native/surface_evidence.npz')]:
    key=pair+'_'+suffix
    with np.load(mapfile) as z:xyz=z['xyz_m'].copy();labels=z['instance_id'].copy();rgb=z['rgb'].copy()
    with np.load(evfile) as z:np.testing.assert_array_equal(xyz,z['xyz_m']);states=z['state'].copy()
    assert np.isfinite(xyz).all() and len(xyz)==len(labels)==len(states)
    maps[key]=dict(xyz=xyz,labels=labels,rgb=rgb,states=states,path=mapfile,statepath=evfile,
      checkpoint=checkpoint,group=group,when=suffix,pair=pair)
    for name,p in [('map',mapfile),('evidence',evfile)]:sources[key+'_'+name]=dict(path=str(p),sha256=digest(p))
   for field in ['xyz','rgb']:np.testing.assert_array_equal(maps[pair+'_before'][field],maps[pair+'_after'][field])
   trajectory=root/'mapping/inputs/scene/traj.txt'
   poses=np.loadtxt(trajectory).reshape(-1,4,4);cutoff=(checkpoint-1)*5
   assert len(poses)==cutoff+1,(str(trajectory),len(poses),cutoff+1)
   cameras=[dict(frame=f,eye=poses[f,:3,3].tolist(),forward=poses[f,:3,2].tolist(),up=(-poses[f,:3,1]).tolist()) for f in range(0,cutoff+1,5)]
   for suffix in ['before','after']:maps[pair+'_'+suffix]['cameras']=cameras
   sources[pair+'_trajectory']=dict(path=str(trajectory),sha256=digest(trajectory))
 # Independent geometries are concatenated for display adjacency only; no map correspondence is inferred.
 offsets={};cursor=0;geometries=[]
 for pair in pair_keys:
  points=maps[pair+'_before']['xyz'];offsets[pair]=(cursor,cursor+len(points));cursor+=len(points);geometries.append(points)
 xyz_joint=np.vstack(geometries);variants=[]
 for m in maps.values():
  a=np.full(cursor,-1,np.int32);lo,hi=offsets[m['pair']];a[lo:hi]=m['labels'];variants.append(a)
 dump(OUT/'export_status.json',dict(status='RUNNING',stage='joint_spatial_palette',message='八个实际实例状态已就绪，正在联合选色与导出。'))
 palette,audit=build_palette(xyz_joint,variants,radius_m=.25)
 audit.update(display_only=True,source_states=list(maps),source_geometries={p:len(maps[p+'_before']['xyz']) for p in pair_keys},
  palette_uses_all_eight_completed_states=True,display_palette_can_differ_from_previous50_100_page=True,
  source_map_coordinates_and_labels_unchanged=True,no_optimization_or_repair_from_future_observations=True)
 dump(OUT/'palette.json',audit)
 dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
 rows=[];field_audits={};seen_colors={}
 for key,m in maps.items():
  xyz=m['xyz'];labels=m['labels'];states=m['states'];n=len(xyz);a=np.empty(n,dtype)
  for i,name in enumerate(['x','y','z']):a[name]=xyz[:,i]
  color=np.zeros((n,3),np.uint8)
  for pid in np.unique(labels[labels>0]):color[labels==pid]=palette[str(int(pid))]
  color[(labels<=0)&(states==1)]=[255,255,0];color[(labels<=0)&(states==3)]=[255,0,0]
  for i,name in enumerate(['red','green','blue']):a[name]=color[:,i]
  a['instance_id']=labels;a['state']=states;a['surface_index']=np.arange(n)
  p=OUT/'data'/f'room0_{key}_instances_joint.ply'
  header=('ply\nformat binary_little_endian 1.0\ncomment Instance colors, not RGB; joint display palette200/300/A/B; original labels and geometry unchanged\n'
   +f'element vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nproperty uchar state\nproperty int surface_index\nend_header\n').encode()
  tmp=p.with_suffix('.ply.tmp')
  with tmp.open('wb') as f:f.write(header);a.tofile(f)
  z=np.memmap(tmp,dtype=dtype,mode='r',offset=len(header),shape=(n,))
  np.testing.assert_array_equal(z,a)
  np.testing.assert_array_equal(np.column_stack([z[x] for x in ['x','y','z']]),xyz)
  np.testing.assert_array_equal(z['instance_id'],labels);np.testing.assert_array_equal(z['state'],states)
  np.testing.assert_array_equal(z['surface_index'],np.arange(n))
  written_color=np.column_stack([z[x] for x in ['red','green','blue']]);np.testing.assert_array_equal(written_color,color)
  for pid in np.unique(labels[labels>0]):
   c=np.unique(written_color[labels==pid],axis=0);assert c.shape==(1,3)
   np.testing.assert_array_equal(c[0],palette[str(int(pid))])
   if int(pid) in seen_colors:assert seen_colors[int(pid)]==c[0].tolist()
   seen_colors[int(pid)]=c[0].tolist()
  del z;tmp.replace(p)
  field_audits[key]=dict(status='PASS',points=n,coordinates_exact=True,instance_id_exact=True,native_state_exact=True,
   surface_index_exact=True,no_downsampling=True,PLY_instance_color_exact_palette=True,
   source_coordinate_sha256=ah(xyz),source_instance_id_sha256=ah(labels),source_native_state_sha256=ah(states),
   source_map=str(m['path']),source_native_evidence=str(m['statepath']))
  ids,counts=np.unique(labels[labels>0],return_counts=True);frames=m['checkpoint']
  rows.append(dict(key=key,title=f"{frames} 帧 · {m['group']} 分支 · "+('修复前' if m['when']=='before' else '修复后'),
   frames=frames,last_frame=(frames-1)*5,group=m['group'],stage=m['when'],points=n,assigned=int((labels>0).sum()),
   unassigned=int((labels<=0).sum()),instances=len(ids),states={name:int(((labels<=0)&(states==s)).sum()) for name,s in [('U',0),('T',1),('C',3)]},
   bbox=[xyz.min(0).tolist(),xyz.max(0).tolist()],instance_counts={str(int(i)):int(c) for i,c in zip(ids,counts)},
   ply=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=digest(p),source_map=str(m['path']),
   source_sha256=sources[key+'_map']['sha256'],source_labels_sha256=ah(labels),cameras=m['cameras'],
   tag='承接本分支上次提交后续建' if m['when']=='before' else 'partial＋核心＋E1完成',
   downloads=[dict(title='实例 PLY（全量）',file=str(p.relative_to(OUT)))],
   instance_colors_not_RGB=True,all_surface_indices_present_exactly_once=True))
  print('EXPORTED',key,n,p.stat().st_size,flush=True)
 for v in sources.values():assert digest(v['path'])==v['sha256']
 for v in completed.values():assert digest(v['path'])==v['sha256']
 prior=read(OUT/'prior_pages_untouched.json')
 for path,expected in prior['sha256'].items():assert digest(path)==expected
 dump(OUT/'field_audit.json',dict(status='PASS',maps=field_audits,same_numeric_ID_same_RGB_across_all_maps=True,
  positive_ID_colors=seen_colors,within_pair_geometry_exact=True,cross_pair_surface_index_not_assumed=True,
  previous50_100_pages_unchanged=True))
 result=dict(status='PASS',scene='room0',maps=rows,source_counts={p:len(maps[p+'_before']['xyz']) for p in pair_keys},
  source_fingerprints=sources,completion_fingerprints=completed,branch_choices=['A','B'],default_branch='A',
  switch_branch_changes_all_four_maps=True,point_geometry_within_each_pair_equal=True,
  cross_branch_geometry_or_point_order_assumed_equal=False,surface_indices_are_checkpoint_local=True,
  source_map_labels_and_geometry_unchanged=True,new_numeric_ID_is_branch_local_not_semantic_equivalence=True,
  same_numeric_ID_same_color=True,downsampled=False,instance_colors_not_RGB=True,GT_used=False,
  maximum_source_raw_frame=1495,palette='palette.json',field_audit='field_audit.json',
  changed_points={p:int((maps[p+'_before']['labels']!=maps[p+'_after']['labels']).sum()) for p in pair_keys},
  palette_metrics={k:audit[k] for k in ['instance_count','neighbor_pairs','minimum_neighbor_deltaE76','p10_neighbor_deltaE76','minimum_all_ID_deltaE76']},
  visualization_export_seconds=time.perf_counter()-start,previous50_100_pages_unchanged=True)
 dump(OUT/'manifest.ready.json',result)
 dump(OUT/'export_ready.json',dict(status='PASS',seconds=result['visualization_export_seconds'],maps=8,
  total_PLY_bytes=sum(r['bytes'] for r in rows),palette_metrics=result['palette_metrics']))
 dump(OUT/'export_status.json',dict(status='PASS',stage='export_verified',message='八份实际实例PLY已导出；等待全部提交核验后发布。'))
 print(json.dumps(read(OUT/'export_ready.json')),flush=True)
def publish():
 q=read(OUT/'manifest.ready.json');assert q['status']=='PASS';commits={};audits={}
 for checkpoint in [200,300]:
  for group in ['A','B']:
   r=ROOT/str(checkpoint)/group/'repair'
   verify_complete(r/'repair_complete.json',checkpoint,group)
   p=r/'commit_complete.json';verify_complete(p,checkpoint,group)
   commits[f'{checkpoint}_{group}']=dict(path=str(p),sha256=digest(p))
   p=ROOT/str(checkpoint)/group/'final_audit.json';verify_complete(p,checkpoint,group)
   audits[f'{checkpoint}_{group}']=dict(path=str(p),sha256=digest(p))
 for v in list(q['source_fingerprints'].values())+list(q['completion_fingerprints'].values()):assert digest(v['path'])==v['sha256']
 for row in q['maps']:assert digest(OUT/row['ply'])==row['sha256']
 for p,h in read(OUT/'prior_pages_untouched.json')['sha256'].items():assert digest(p)==h
 q['commit_verification']=commits;q['independent_final_audits']=audits;q['published_unix']=time.time();dump(OUT/'manifest.json',q)
 dump(OUT/'status.json',dict(status='PASS',message='200/300 帧 A/B 八份实际实例 PLY 全部核验完成，可切换分支同步查看四图。'))
 print('PUBLISHED',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');args=ap.parse_args()
 try:publish() if args.publish else export()
 except Exception:
  dump(OUT/'export_failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
