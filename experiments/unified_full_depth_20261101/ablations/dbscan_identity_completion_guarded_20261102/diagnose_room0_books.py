import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json, collections
import numpy as np
from scipy.spatial import cKDTree

O=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
T=O/'ablations/dbscan_identity_completion_guarded_20261102'
D=O/'diagnostics/room0_books_20261102'; D.mkdir(parents=True,exist_ok=True)
G=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009/mesh/room0')
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def hist(x):
 ids,n=np.unique(x,return_counts=True);order=np.argsort(-n)
 return {str(ids[k]):int(n[k]) for k in order}
def dump(p,x):p.write_text(json.dumps(x,indent=2))
g=load(G/'gt.npz');scope=json.loads((G/'gt_scope.json').read_text())
corr=load('/data/chenkejun/CVPR/results/ovimap_full_depth_baseline_20261101/scenes/room0/evaluation/r2_full_depth_correspondence.npz')
surf=load(O/'published/online_B/room0/final_B/instance_surface.npz');xyz=surf['xyz_m']
gtree=cKDTree(g['xyz_ref']);rois={};fixed={};books=[]
for ob in scope['objects']:
 if ob['semantic_class']!='book':continue
 gid=ob['raw_gt_id'];ref=g['raw_instance_id']==gid;bb=g['xyz_ref'][ref]
 ix=np.flatnonzero(np.all((xyz>=bb.min(0)-.01)&(xyz<=bb.max(0)+.01),axis=1))
 dd,j=gtree.query(xyz[ix],workers=2);ix=ix[(dd<=.01)&(g['raw_instance_id'][j]==gid)]
 rois[gid]=ix;fixed[gid]=np.unique(corr['nearest_native_index'][ref&(corr['distances_m']<=.01)&(corr['nearest_native_index']>=0)])
 books.append(dict(gt_id=gid,points=len(ix),fixed_correspondence_points=len(fixed[gid]),bbox=[bb.min(0).tolist(),bb.max(0).tolist()]))
np.savez_compressed(D/'book_regions.npz',**{'book_'+str(k):v for k,v in rois.items()},**{'fixed_'+str(k):v for k,v in fixed.items()})
norm=load(O/'attributes/room0/400/normals.npz');rgb=load(O/'attributes/room0/400/rgb.npz')
valid=norm['normal_valid']&rgb['color_valid'];normals=norm['normals']
result=dict(region_method='nearest reference book surface <=1cm, diagnosis only; fixed correspondence separately audited',books=books,variants={})
for v in ['no_repair','main','online_B']:
 labels=load(O/'published'/v/'room0/final_B/instance_surface.npz')['instance_id']
 manifest=json.loads((O/'published'/v/'room0/scene_complete.json').read_text())
 native=load(manifest['inputs']['native'])['instance_id'];ev=load(manifest['inputs']['evidence'])
 vr={};result['variants'][v]=vr
 for gid,ix in rois.items():
  gray=ix[labels[ix]<0];row=dict(labels=hist(labels[ix]),fixed_labels=hist(labels[fixed[gid]]),native_labels=hist(native[ix]),state_gray=hist(ev['state'][gray]),gray_top1=hist(ev['top1_instance_id'][gray]),gray_top2=hist(ev['top2_instance_id'][gray]))
  row['gray_vote_quantiles']={k:np.quantile(ev[k][gray],[0,.25,.5,.75,1]).tolist() if len(gray) else [] for k in ['top1_votes','total_frame_votes','confidence']}
  row['competing_pairs']=[dict(ids=list(k),n=n) for k,n in collections.Counter(tuple(sorted([int(a),int(b)])) for a,b in zip(ev['top1_instance_id'][gray],ev['top2_instance_id'][gray])).most_common(8)]
  vr[str(gid)]=row
 if v=='no_repair':continue
 after=load(T/v/'room0/merge_fill/instance_surface.npz')['instance_id']
 before=load(T/v/'room0/merge_only/instance_surface.npz')['instance_id']
 ev=load(T/v/'room0/merge_only/evidence.npz')
 seed=np.flatnonzero((ev['state']==2)&(ev['top1_votes']>=4)&(ev['confidence']>=.8)&valid&(before==ev['top1_instance_id']))
 cells=np.rint(xyz[seed]/.01).astype(np.int32);_,rep=np.unique(cells,axis=0,return_index=True);seed=seed[rep];tree=cKDTree(xyz[seed])
 di=load(T/v/'room0/merge_fill/fill_diagnostics.npz')
 for gid,ix in rois.items():
  row=vr[str(gid)];row['dbscan_labels']=hist(after[ix]);row['fixed_dbscan_labels']=hist(after[fixed[gid]])
  q=ix[before[ix]<0];d,j=tree.query(xyz[q],k=32,distance_upper_bound=.05,workers=2);near=np.isfinite(d)&(j<len(seed));a=seed[np.minimum(j,len(seed)-1)]
  delta=xyz[a]-xyz[q,None];dot=np.einsum('qkc,qc->qk',normals[a],normals[q]);p1=np.abs(np.einsum('qkc,qc->qk',delta,normals[q]));p2=np.abs(np.sum(delta*normals[a],axis=2));ok=near&(dot>=.85)&(p1<=.0075)&(p2<=.0075)
  labs=before[a];w=np.where(ok,np.exp(-d/.02),0);eq=labs[:,:,None]==labs[:,None,:];score=np.sum(eq*w[:,None,:],axis=2);score[~ok]=0;b=score.argmax(1);rr=np.arange(len(q));cid=labs[rr,b]
  frac=np.divide(score[rr,b],w.sum(1),out=np.zeros(len(q)),where=w.sum(1)>0);same=(eq[rr,b]&ok).sum(1)
  first=np.full(len(q),'passed_spatial',dtype='<U55')
  first[frac<.9]='mixed_neighbor_IDs_dominance_below_90percent';first[same<8]='fewer_than_8_compatible_same_ID_anchor_cells';first[ok.sum(1)==0]='no_normal_plane_compatible_anchor';first[near.sum(1)==0]='no_anchor_within_5cm';first[~valid[q]]='invalid_normal_or_color'
  pos=np.searchsorted(di['target_point_index'],q);inbounds=pos<len(di['target_point_index']);hit=np.zeros(len(q),bool);hit[inbounds]=di['target_point_index'][pos[inbounds]]==q[inbounds]
  prop=np.full(len(q),-1);prop[hit]=di['proposed_id'][pos[hit]]
  first[(first=='passed_spatial')&(prop<0)]='historical_vote_constraint'
  first[(first=='passed_spatial')&(prop>0)&(after[q]<0)]='DBSCAN_anchor_cluster_constraint'
  first[after[q]>0]='filled'
  row['fill_stage_first_failure']=hist(first);row['remaining_gray_reason']=hist(first[after[q]<0])
  row['book_anchor_count']=int(np.isin(seed,ix).sum());row['remaining_gray_votes']=hist(ev['state'][q[after[q]<0]])
  row['samples']=[]
  if len(q):
   remaining=np.flatnonzero(after[q]<0);chosen=remaining[np.argsort(-ev['total_frame_votes'][q[remaining]])[:3]]
   for z in chosen:
    i=int(q[z]);row['samples'].append(dict(point=i,xyz=xyz[i].tolist(),reason=str(first[z]),state=int(ev['state'][i]),top1=int(ev['top1_instance_id'][i]),top1_votes=int(ev['top1_votes'][i]),top2=int(ev['top2_instance_id'][i]),top2_votes=int(ev['top2_votes'][i]),total=int(ev['total_frame_votes'][i]),confidence=float(ev['confidence'][i]),near=int(near[z].sum()),compatible=int(ok[z].sum()),same=int(same[z]),candidate=int(cid[z]),dominance=float(frac[z])))
  np.savez_compressed(D/f'{v}_book{gid}_fill.npz',point_index=q,reason=first,nearest_distance=d[:,0],compatible_count=ok.sum(1),same_count=same,candidate_id=cid,dominance=frac)
 dump(D/'summary.json',result)
 print(v,json.dumps({k:{x:y for x,y in r.items() if x in ['labels','dbscan_labels','state_gray','competing_pairs','remaining_gray_reason','book_anchor_count','samples']} for k,r in vr.items() if k in ['12','59','92']}),flush=True)
dump(D/'summary.json',result)
print('BOOK_AUDIT_PASS',flush=True)
