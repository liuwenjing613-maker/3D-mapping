"""Display-only instance PLY export for50/100 and independent100 A/B repairs."""
from pathlib import Path
import argparse,hashlib,json,sys,time,traceback
import numpy as np
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
sys.path.insert(0,str(BASE))
from spatial_instance_palette import build_palette
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
OUT=TOP/'review/compare50_100'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def publish():
 ready=read(OUT/'manifest.ready.json');assert ready['status']=='PASS'
 commits={}
 for g in ['A','B']:
  repair=TOP/'repair100'/g/'repair_complete.json';assert read(repair)['status']=='PASS'
  p=TOP/'repair100'/g/'commit_complete.json';assert p.exists(),str(p)
  q=read(p);assert q['status']=='PASS',q;commits[g]=dict(path=str(p),sha256=digest(p))
 for key,row in ready['source_fingerprints'].items():assert digest(row['path'])==row['sha256'],key
 for row in ready['maps']:assert digest(OUT/row['ply'])==row['sha256']
 ready['commit_verification']=commits;ready['published_unix']=time.time()
 dump(OUT/'manifest.json',ready)
 dump(OUT/'status.json',dict(status='PASS',message='50前后与100前后实例PLY已校验；A/B独立修复可切换，所有视图共用一套实例色。',page='http://127.0.0.1:8798/online-repair/compare50_100/'))
 print('PUBLISHED',flush=True)
def export():
 started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
 old=read(TOP/'review/50/manifest.json');assert old['status']=='PASS'
 oldrows={x['key']:x for x in old['maps']}
 old_ply_hashes={str(TOP/'review/50'/row['ply']):digest(TOP/'review/50'/row['ply']) for row in old['maps']}
 for g in ['A','B']:
  p=TOP/'repair100'/g/'repair_complete.json'
  assert p.exists() and read(p)['status']=='PASS',str(p)
 definitions=[
  ('50_before','50 帧 · 修复前',50,Path(oldrows['before']['source_map']),Path('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021/baseline/surface_evidence.npz')),
  ('50_after','50 帧 · 第一次修复后',50,Path(oldrows['after']['source_map']),Path(oldrows['after']['source_map']).parent.parent/'native/surface_evidence.npz'),
  ('100_before','100 帧 · 第二次修复前',100,TOP/'prefix100/baseline/final/instance_surface.npz',TOP/'prefix100/baseline/surface_evidence.npz')]
 for g in ['A','B']:
  r=TOP/'repair100'/g/'partial/whole_to_partial/room0'
  definitions.append(('100_'+g,'100 帧 · '+g+'组修复后',100,r/'final/instance_surface.npz',r/'native/surface_evidence.npz'))
 sources={};maps={}
 for key,title,frames,path,statepath in definitions:
  with np.load(path) as z:xyz=z['xyz_m'].copy();labels=z['instance_id'].copy();rgb=z['rgb'].copy()
  with np.load(statepath) as z:np.testing.assert_array_equal(xyz,z['xyz_m']);states=z['state'].copy()
  assert np.isfinite(xyz).all() and len(labels)==len(states)==len(xyz)
  maps[key]=dict(xyz=xyz,labels=labels,rgb=rgb,states=states,title=title,frames=frames,path=path,statepath=statepath)
  for suffix,p in [('map',path),('states',statepath)]:sources[key+'_'+suffix]=dict(path=str(p),sha256=digest(p))
 for key,reference in [('50_after','50_before'),('100_A','100_before'),('100_B','100_before')]:
  np.testing.assert_array_equal(maps[key]['xyz'],maps[reference]['xyz'])
  np.testing.assert_array_equal(maps[key]['rgb'],maps[reference]['rgb'])
 n50=len(maps['50_before']['xyz']);n100=len(maps['100_before']['xyz'])
 assert n50==686681 and n100==697680
 # Both geometries are retained separately. Padding only forms a visualization color graph.
 xyz=np.vstack([maps['50_before']['xyz'],maps['100_before']['xyz']])
 label_variants=[]
 for key in ['50_before','50_after','100_before','100_A','100_B']:
  a=maps[key]['labels']
  label_variants.append(np.r_[a,np.full(n100,-1,np.int32)] if key.startswith('50_') else np.r_[np.full(n50,-1,np.int32),a])
 dump(OUT/'status.json',dict(status='PREPARING',message='实际A/B修复结果已完成，正在为五个实例状态联合生成空间高色差配色。'))
 palette,audit=build_palette(xyz,label_variants,radius_m=.25)
 audit.update(method='joint50/100/A/B spatial adjacency Lab maximin; numeric ID shared across all displayed states',
  palette_only=True,source_geometry_counts=[n50,n100],source_states=list(maps),palette_raw_cutoff=495,
  future_frames_used=False,historical_map_labels_unchanged=True,new_comparison_palette_may_differ_from_previous50_page=True)
 dump(OUT/'palette.json',audit)
 trajectories=np.loadtxt(TOP/'prefix100/inputs/scene/traj.txt').reshape(-1,4,4);assert len(trajectories)==496
 cameras=[dict(frame=fid,eye=trajectories[fid,:3,3].tolist(),forward=trajectories[fid,:3,2].tolist(),up=(-trajectories[fid,:3,1]).tolist()) for fid in range(0,496,5)]
 np.testing.assert_allclose(cameras[0]['eye'],oldrows['before']['cameras'][0]['eye'])
 dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
 rows=[];field_audits={};seen_colors={}
 for key,m in maps.items():
  xyz=m['xyz'];labels=m['labels'];states=m['states']
  a=np.empty(len(xyz),dtype)
  for i,name in enumerate(['x','y','z']):a[name]=xyz[:,i]
  color=np.zeros((len(xyz),3),np.uint8)
  for pid in np.unique(labels[labels>0]):color[labels==pid]=palette[str(int(pid))]
  color[(labels<=0)&(states==1)]=[255,255,0];color[(labels<=0)&(states==3)]=[255,0,0]
  for i,name in enumerate(['red','green','blue']):a[name]=color[:,i]
  a['instance_id']=labels;a['state']=states;a['surface_index']=np.arange(len(xyz))
  file=OUT/'data'/('room0_'+key+'_instances_joint.ply')
  header=('ply\nformat binary_little_endian 1.0\ncomment Instance colors, not RGB; joint display palette50/100/A/B; all source labels and geometry unchanged\n'
   +f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nproperty uchar state\nproperty int surface_index\nend_header\n').encode()
  with file.open('wb') as f:f.write(header);a.tofile(f)
  check=np.memmap(file,dtype=dtype,mode='r',offset=len(header),shape=(len(a),))
  np.testing.assert_array_equal(check,a)
  # Independently compare the written fields with their immutable NPZ sources.
  written_xyz=np.column_stack([check[name] for name in ['x','y','z']])
  np.testing.assert_array_equal(written_xyz,xyz)
  np.testing.assert_array_equal(check['instance_id'],labels)
  np.testing.assert_array_equal(check['state'],states)
  np.testing.assert_array_equal(check['surface_index'],np.arange(len(xyz)))
  written_rgb=np.column_stack([check[name] for name in ['red','green','blue']])
  np.testing.assert_array_equal(written_rgb,color)
  for pid in np.unique(labels[labels>0]):
   value=np.unique(written_rgb[labels==pid],axis=0)
   assert value.shape==(1,3)
   np.testing.assert_array_equal(value[0],palette[str(int(pid))])
   if int(pid) in seen_colors:assert seen_colors[int(pid)]==value[0].tolist()
   seen_colors[int(pid)]=value[0].tolist()
  field_audits[key]=dict(status='PASS',points=len(xyz),coordinates_exact=True,
   instance_id_exact=True,native_state_exact=True,surface_index_exact=True,
   no_downsampling=True,source_coordinate_dtype=str(xyz.dtype),
   PLY_instance_color_exact_palette=True,PLY_RGB_is_not_source_RGB=True,
   source_coordinate_sha256=array_hash(xyz),source_instance_id_sha256=array_hash(labels),
   source_native_state_sha256=array_hash(states),source_map=str(m['path']),
   source_native_evidence=str(m['statepath']))
  ids,counts=np.unique(labels[labels>0],return_counts=True)
  rows.append(dict(key=key,title=m['title'],frames=m['frames'],last_frame=(m['frames']-1)*5,points=len(xyz),
   assigned=int((labels>0).sum()),unassigned=int((labels<=0).sum()),instances=len(ids),
   states={name:int(((labels<=0)&(states==val)).sum()) for name,val in [('U',0),('T',1),('C',3)]},
   bbox=[xyz.min(0).tolist(),xyz.max(0).tolist()],instance_counts={str(int(i)):int(n) for i,n in zip(ids,counts)},
   ply=str(file.relative_to(OUT)),bytes=file.stat().st_size,sha256=digest(file),
   source_map=str(m['path']),source_sha256=sources[key+'_map']['sha256'],source_labels_sha256=array_hash(labels),
   cameras=cameras[:m['frames']],all_surface_indices_present_exactly_once=True,instance_colors_not_RGB=True,
   tag='原严格一票' if key=='50_before' else '50修复后续建至100' if key=='100_before' else 'partial＋E1完成',
   downloads=[dict(title='实例 PLY（全量）',file=str(file.relative_to(OUT)))]))
  print('EXPORTED',key,len(a),file.stat().st_size,flush=True)
 dump(OUT/'field_audit.json',dict(status='PASS',maps=field_audits,same_numeric_ID_same_RGB_across_all_maps=True,positive_ID_colors=seen_colors))
 for row in sources.values():assert digest(row['path'])==row['sha256']
 for path,h in old_ply_hashes.items():assert digest(path)==h
 manifest=dict(status='PASS',scene='room0',maps=rows,
  point_geometry_within_each_checkpoint_equal=True,cross_checkpoint_geometry_same=False,
  surface_indices_are_checkpoint_local=True,same_numeric_ID_same_color=True,downsampled=False,GT_used=False,
  instance_colors_not_RGB=True,palette='palette.json',palette_only_changes=True,
  old50_existing_PLY_hashes_unchanged=old_ply_hashes,source_fingerprints=sources,default_branch='A',
  branch_choices=['A','B'],palette_rule='25cm spatial adjacency over union of five displayed states; deterministic Lab contrast',
  source_candidate_groups=dict(A=13,B=12),repair100_branch_state_independent=True,
  new_numeric_ID_is_branch_local_not_cross_branch_semantic_equivalence=True,field_audit='field_audit.json',
  changed_points=dict(repair50=int(np.count_nonzero(maps['50_before']['labels']!=maps['50_after']['labels'])),
   repair100_A=int(np.count_nonzero(maps['100_before']['labels']!=maps['100_A']['labels'])),
   repair100_B=int(np.count_nonzero(maps['100_before']['labels']!=maps['100_B']['labels']))),
  source_counts=dict(points50=n50,points100=n100),visualization_export_seconds=time.perf_counter()-started,
  palette_metrics={k:audit[k] for k in ['instance_count','neighbor_pairs','minimum_neighbor_deltaE76','p10_neighbor_deltaE76','minimum_all_ID_deltaE76']},
  algorithm_outputs_unmodified=True,only_known_raw_frames_0_to495=True)
 dump(OUT/'manifest.ready.json',manifest)
 dump(OUT/'export_ready.json',dict(status='PASS',seconds=manifest['visualization_export_seconds'],maps=5,total_PLY_bytes=sum(row['bytes'] for row in rows),palette_metrics=manifest['palette_metrics'],ready_manifest=str(OUT/'manifest.ready.json')))
 dump(OUT/'status.json',dict(status='AWAITING_COMMIT_VERIFICATION',message='五份真实实例PLY已导出并核验，等待A/B分支状态提交校验后发布。'))
 print(json.dumps(read(OUT/'export_ready.json')),flush=True)
if __name__=='__main__':
 args=argparse.ArgumentParser();args.add_argument('--publish',action='store_true');args=args.parse_args()
 try:
  if args.publish:publish()
  else:export()
 except Exception:
  dump(OUT/'export_failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
