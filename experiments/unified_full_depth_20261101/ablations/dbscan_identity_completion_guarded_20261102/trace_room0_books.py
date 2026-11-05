import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,collections
import numpy as np
from scipy.spatial import cKDTree
O=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101');D=O/'diagnostics/room0_books_20261102';T=O/'ablations/dbscan_identity_completion_guarded_20261102'
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def hist(x):
 u,n=np.unique(x,return_counts=True);return {str(k):int(v) for k,v in zip(u,n)}
def dump(p,x):p.write_text(json.dumps(x,indent=2))
g=load('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009/mesh/room0/gt.npz');gtree=cKDTree(g['xyz_ref'])
final=load(O/'published/online_B/room0/final_B/instance_surface.npz');ev=load(T/'online_B/room0/merge_only/evidence.npz');regions=load(D/'book_regions.npz')
ix=regions['book_92'];q=ix[(final['instance_id'][ix]<0)&(ev['top1_instance_id'][ix]==235)&(ev['top2_instance_id'][ix]==605)&(ev['total_frame_votes'][ix]>40)]
center=g['xyz_ref'][g['raw_instance_id']==92].mean(0);central=int(q[np.argmin(np.linalg.norm(final['xyz_m'][q]-center,axis=1))]);points=[central,1628063,1182712]
res=dict(points=[dict(index=i,xyz=final['xyz_m'][i].tolist(),final_evidence={k:float(ev[k][i]) for k in ['top1_instance_id','top1_votes','top2_instance_id','top2_votes','total_frame_votes','confidence']}) for i in points],checkpoints=[],seeds=[],gate={},merge_rejections=[])
for cp in [200,300,400]:
 r=O/'online/room0'/str(cp)/'B';be=load(r/'mapping/baseline/surface_evidence.npz');xyz=be['xyz_m'];pdist,pidx=cKDTree(xyz).query(final['xyz_m'][points]);native=load(r/'repair/partial/whole_to_partial/room0/native/surface_evidence.npz');reg=load(r/'repair/e1/region.npz')
 gtbook=g['xyz_ref'][g['raw_instance_id']==92];z=np.flatnonzero(np.all((xyz>=gtbook.min(0)-.01)&(xyz<=gtbook.max(0)+.01),axis=1));dd,gi=gtree.query(xyz[z]);z=z[(dd<=.01)&(g['raw_instance_id'][gi]==92)]
 record=dict(checkpoint=cp,book_points=len(z),before_state=hist(be['state'][z]),before_topID=hist(be['top1_instance_id'][z]),after_state=hist(native['state'][z]),after_topID=hist(native['top1_instance_id'][z]),E1_region_owners=hist(reg['owner'][z]),E1_core_owners=hist(reg['core_owner'][z]),points=[])
 for old,i,dist in zip(points,pidx,pdist):
  record['points'].append(dict(final_index=old,index=int(i),distance=float(dist),before={k:float(be[k][i]) for k in ['top1_instance_id','top1_votes','top2_instance_id','top2_votes','total_frame_votes','confidence']},after={k:float(native[k][i]) for k in ['top1_instance_id','top1_votes','top2_instance_id','top2_votes','total_frame_votes','confidence']},E1_owner=int(reg['owner'][i]),E1_core=int(reg['core_owner'][i])))
 res['checkpoints'].append(record)
 wanted={200:[27],300:[17,20],400:[27,31]}[cp]
 diagnostics=json.loads((r/'repair/seed_diagnostics.json').read_text());assoc=json.loads((r/'repair/global_association.json').read_text())
 for case in diagnostics['cases']:
  for ob in case['objects']:
   if ob['track_id'] not in wanted:continue
   s=load(r/'repair/cases'/case['case_uid']/'seed_projected_support.npz');pts=s['surface_point_index'][s['track_id']==ob['track_id']];d,j=gtree.query(xyz[pts]);classes=np.where(d<=.02,g['raw_instance_id'][j],-999)
   res['seeds'].append(dict(checkpoint=cp,case=case['case_uid'],track=ob['track_id'],frame=case['source_choice']['frame'],mask_id=ob['mask_id'],GT_composition_2cm=hist(classes),seed=ob,association=[a for a in assoc['objects'] if a['canonical_track_id']==ob['track_id']],scores=[a for a in assoc['source_view_scores'] if a['track_id']==ob['track_id']]))
 if cp==300:
  gates=json.loads((r/'tracking/gate/frame_decisions.json').read_text())['frames'];rr=[(f['frame'],f['objects']['20']) for f in gates if '20' in f['objects']];vis=[(f,x) for f,x in rr if x['visible_seed_samples']>0]
  res['gate']=dict(total=len(rr),seed_visible_frames=len(vis),accepted=sum(x['accepted'] for _,x in rr),rejected_visible=sum(not x['accepted'] for _,x in vis),rejection_reasons_visible=dict(collections.Counter(reason for _,x in vis if not x['accepted'] for reason in x['reasons'])),frame_details=[dict(frame=f,**x) for f,x in rr])
 dump(D/'trace.json',res);print('CHECKPOINT',json.dumps(record),flush=True)
r=O/'online/room0/400/B';base=json.loads((r/'repair/global_association.json').read_text())['instance_base']
for row in res['points']:row['frames']=[]
for f in sorted((r/'repair/partial/transactions').glob('f*.npz')):
 fid=int(f.stem[1:])
 with np.load(f) as z:
  arrs={k:z[k] for k in ['baseline_frame_keys','whole_to_partial_frame_keys','whole_to_partial_candidate_keys']}
 for row in res['points']:
  i=row['index'];values={}
  for k,arr in arrs.items():
   lo=np.searchsorted(arr,i*base);hi=np.searchsorted(arr,(i+1)*base);values[k]=[int(x%base) for x in arr[lo:hi]]
  if any(values.values()):row['frames'].append(dict(frame=fid,**values))
for row in res['points']:
 row['totals']={};row['by_interval']={}
 for key in ['baseline_frame_keys','whole_to_partial_frame_keys']:
  counter=collections.Counter(x for f in row['frames'] for x in f[key]);row['totals'][key]=dict(counter)
  for low,high in [(0,1000),(1000,1500),(1500,2000)]:row['by_interval'].setdefault(str(low)+'-'+str(high-1),{})[key]=dict(collections.Counter(x for f in row['frames'] if low<=f['frame']<high for x in f[key]))
 row['final_count_verified']=sum(row['totals']['whole_to_partial_frame_keys'].values())==row['final_evidence']['total_frame_votes']
dec=json.loads((T/'online_B/room0/merge_decisions.json').read_text());res['merge_keys']=list(dec)
for k,v in dec.items():
 if isinstance(v,list):
  for x in v:
   if isinstance(x,dict) and set(x.get('ids',[]))&{235,605,890}:res['merge_rejections'].append(x)
dump(D/'trace.json',res)
print('POINTS',json.dumps([{k:v for k,v in r.items() if k!='frames'} for r in res['points']]),flush=True)
print('SEEDS',json.dumps([{k:v for k,v in r.items() if k not in ['seed','scores']} for r in res['seeds']]),flush=True)
print('MERGES',json.dumps(res['merge_rejections']),flush=True)
print('TRACE_PASS',flush=True)
