from common import *
import numpy as np,colorsys,time,traceback
def main():
 while not (R/'repair_complete.json').exists():time.sleep(5)
 assert read(R/'repair_complete.json')['status']=='PASS'
 out=TOP/'review/50';data=out/'data'
 original=read(B/'review/3d/manifest.json');before_meta=next(x for x in original['maps'] if x['frames']==50)
 before_paths=(B/'baseline/final/instance_surface.npz',B/'baseline/surface_evidence.npz')
 after_root=R/'partial/whole_to_partial/room0'
 after_paths=(after_root/'final/instance_surface.npz',after_root/'native/surface_evidence.npz')
 dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
 rows=[];reference=None;before_labels=None
 for key,paths in [('before',before_paths),('after',after_paths)]:
  with np.load(paths[0]) as z:xyz=z['xyz_m'];labels=z['instance_id'];rgb=z['rgb']
  with np.load(paths[1]) as z:
   np.testing.assert_array_equal(xyz,z['xyz_m']);states=z['state']
  if reference is None:reference=(xyz.copy(),rgb.copy());before_labels=labels.copy()
  else:
   np.testing.assert_array_equal(xyz,reference[0]);np.testing.assert_array_equal(rgb,reference[1])
  a=np.empty(len(xyz),dtype)
  for i,name in enumerate(('x','y','z')):a[name]=xyz[:,i]
  ids,inverse=np.unique(labels[labels>0],return_inverse=True)
  palette=np.asarray([np.array(colorsys.hsv_to_rgb((int(k)*.61803398875)%1,.67,.94))*255 for k in ids],np.uint8)
  color=np.zeros((len(xyz),3),np.uint8);color[labels>0]=palette[inverse]
  color[(labels<=0)&(states==1)]=[255,255,0];color[(labels<=0)&(states==3)]=[255,0,0]
  for i,name in enumerate(('red','green','blue')):a[name]=color[:,i]
  a['instance_id']=labels;a['state']=states;a['surface_index']=np.arange(len(xyz))
  fname='room0_50_'+key+'_instances.ply';p=data/fname
  header=('ply\nformat binary_little_endian 1.0\ncomment Instance colors, not RGB; causal raw frame cutoff 245; identical geometry before and after repair\n'
   +f'element vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nproperty uchar state\nproperty int surface_index\nend_header\n').encode()
  with p.open('wb') as f:f.write(header);a.tofile(f)
  check=np.memmap(p,dtype=dtype,mode='r',offset=len(header),shape=(len(a),));np.testing.assert_array_equal(check,a)
  ids,counts=np.unique(labels[labels>0],return_counts=True)
  rows.append(dict(key=key,title='50 帧 · '+('未修复' if key=='before' else '修复后'),tag='严格一票基线' if key=='before' else 'partial + E1 已完成',
   frames=50,last_frame=245,points=len(a),assigned=int((labels>0).sum()),instances=len(ids),unassigned=int((labels<=0).sum()),
   states={n:int(((labels<=0)&(states==k)).sum()) for n,k in [('U',0),('T',1),('C',3)]},
   bbox=[xyz.min(0).tolist(),xyz.max(0).tolist()],instance_counts={str(i):int(n) for i,n in zip(ids,counts)},
   ply='data/'+fname,bytes=p.stat().st_size,sha256=digest(p),cameras=before_meta['cameras'],
   downloads=[dict(title=('未修复' if key=='before' else '修复后')+' · 实例 PLY',file='data/'+fname)],
   source_map=str(paths[0]),source_sha256=digest(paths[0]),all_surface_indices_present_exactly_once=True,instance_colors_not_RGB=True))
  if key=='after':changed=int((labels!=before_labels).sum())
 manifest=dict(status='PASS',scene='room0',maps=rows,identical_xyz_rgb_and_point_order=True,instance_label_changed_points=changed,downsampled=False,
   raw_frame_cutoff=245,processed_frames=50,same_ID_same_color=True,GT_used=False,color_mode='instance',second_repair_performed=False)
 dump(out/'manifest.json',manifest)
 web=B/'review/online-repair'
 if not web.exists():web.symlink_to(TOP/'review',target_is_directory=True)
 dump(TOP/'ply50_published.json',dict(status='PASS',page='http://127.0.0.1:8798/online-repair/50/',changed_points=changed,
   before='http://127.0.0.1:8798/online-repair/50/data/room0_50_before_instances.ply',after='http://127.0.0.1:8798/online-repair/50/data/room0_50_after_instances.ply'))
 print(json.dumps(dict(status='PASS',points=len(reference[0]),changed_points=changed)),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(TOP/'ply50_failure.json',dict(traceback=traceback.format_exc()));raise
