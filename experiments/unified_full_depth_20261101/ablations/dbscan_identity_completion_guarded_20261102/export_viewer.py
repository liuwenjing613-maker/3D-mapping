from pathlib import Path
import json,gzip,colorsys,hashlib
import numpy as np
O=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
T=O/'ablations/dbscan_identity_completion_guarded_20261102';V=O/'instance_comparison_viewer';W=T/'viewer'
W.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')])
summary=read(T/'comparison.json')
Q=O/'ablations/dbscan_identity_completion_conservative_20261102';conservative=read(Q/'comparison.json')
for v in summary['pooled']:summary['pooled'][v]['conservative']=conservative['pooled'][v]['merge_fill']
manifest=dict(status='PASS',scenes={},methods=['main','online_B'],labels={'main':'main','online_B':'v2-B'},
 arms=['fill_only','merge_only','merge_fill','conservative'],arm_labels={'fill_only':'仅 DBSCAN 补灰','merge_only':'仅合并＋原规则重投票','merge_fill':'合并＋补灰（覆盖更多）','conservative':'合并＋补灰（保守）'},
 pooled=summary['pooled'],GT=summary['GT'],settings=read(T/'plan.json')['settings'])
pairs=0
for s in summary['scenes']:
 manifest['scenes'][s]={}
 for v in manifest['methods']:
  manifest['scenes'][s][v]={}
  info=read(T/v/s/'complete.json');families=info['families']
  for arm in manifest['arms']:
   r=read((Q/v/s/'merge_fill' if arm=='conservative' else T/v/s/arm)/'complete.json');before=load(r['original_prediction']);after=load(r['temporary_prediction'])
   xyz=before['xyz_m'];old=before['instance_id'];new=after['instance_id'];n=len(old)
   np.testing.assert_array_equal(xyz,after['xyz_m']);np.testing.assert_array_equal(before['rgb'],after['rgb'])
   original=V/s/(v+'.ply')
   with original.open('rb') as f:head=f.read(4096);offset=head.index(b'end_header\n')+11
   orig=np.memmap(original,dtype=dtype,mode='r',offset=offset,shape=(n,));np.testing.assert_array_equal(orig['instance_id'],old)
   oldrgb=np.column_stack([orig[k] for k in ['red','green','blue']]);u,ix=np.unique(old,return_index=True)
   palette={int(i):oldrgb[j].copy() for i,j in zip(u,ix) if i>0}
   for iid in np.unique(new[new>0]):
    if int(iid) not in palette:
     # Canonical members normally exist in original; this handles restored historical IDs.
     col=np.rint(np.array(colorsys.hsv_to_rgb((int(iid)*.61803398875)%1,.8,.95))*255).astype(np.uint8)
     palette[int(iid)]=col
   un,inv=np.unique(new,return_inverse=True);rgb=np.array([palette[int(i)] if i>0 else [95,104,116] for i in un],np.uint8)[inv]
   folder=W/s/v/arm;folder.mkdir(parents=True,exist_ok=True)
   link=folder/'original.ply'
   if not link.exists():link.symlink_to(original)
   arr=np.empty(n,dtype=dtype)
   for i,k in enumerate(['x','y','z']):arr[k]=xyz[:,i]
   for i,k in enumerate(['red','green','blue']):arr[k]=rgb[:,i]
   arr['instance_id']=new
   header=('ply\nformat binary_little_endian 1.0\ncomment temporary DBSCAN identity/completion; native votes >=2 and >=67%; fixed TSDF\nelement vertex '+str(n)+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n').encode()
   ply=folder/'temporary.ply'
   if not ply.exists():
    with ply.open('wb') as f:f.write(header);arr.tofile(f)
   ck=np.memmap(ply,dtype=dtype,mode='r',offset=len(header),shape=(n,));np.testing.assert_array_equal(ck['instance_id'],new)
   for i,k in enumerate(['x','y','z']):np.testing.assert_array_equal(ck[k],xyz[:,i])
   if not (folder/'surface.bin.gz').exists():
    with gzip.open(folder/'surface.bin.gz','wb',compresslevel=1) as f:
     for a in [np.array([n],'<u4'),xyz.astype('<f4'),oldrgb,rgb,old.astype('<i4'),new.astype('<i4')]:f.write(a.tobytes())
   manifest['scenes'][s][v][arm]=dict(count=n,bbox=[xyz.min(0).tolist(),xyz.max(0).tolist()],binary=f'{s}/{v}/{arm}/surface.bin',
    ply_original=f'{s}/{v}/{arm}/original.ply',ply_temporary=f'{s}/{v}/{arm}/temporary.ply',ply_sha256=sha(ply),
    old_instances=int((np.unique(old)>0).sum()),new_instances=int((un>0).sum()),old_gray=r['original_gray'],new_gray=r['temporary_gray'],
    old_metrics=r['original_metrics'],new_metrics=r['temporary_metrics'],families=[] if arm=='fill_only' else families,
    changes={k:r[k] for k in ['gray_to_assigned','assigned_to_gray','assigned_changed_ID']},geometry_GT_protocol_equal=True,same_ID_same_color=True)
   pairs+=1;print('PLY_PASS',s,v,arm,flush=True)
(W/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for name,target in [('vendor',V/'vendor'),('comparison.json',T/'comparison.json'),('first_trial.json',O/'ablations/dbscan_identity_completion_20261102/comparison.json'),('conservative_trial.json',Q/'comparison.json'),('final_audit.json',T/'final_audit.json')]:
 p=W/name
 if not p.exists():p.symlink_to(target,target_is_directory=name=='vendor')
link=V/'dbscan'
if not link.exists():link.symlink_to(W,target_is_directory=True)
(T/'viewer_complete.json').write_text(json.dumps(dict(status='PASS',pairs=pairs,coordinates_and_ids_verified=True,same_ID_same_color=True)))
print('VIEWER_EXPORT_PASS',flush=True)
