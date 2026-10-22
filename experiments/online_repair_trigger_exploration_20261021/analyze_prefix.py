"""Read-only, prefix-scoped diagnostics. Does not change maps, masks, or selection."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from PIL import Image
n=int(sys.argv[1]);assert n in [20,50]
R=Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021')
C=Path(f'/home/chenkejun/CVPR/experiments/online_room0_prefix{n}_20261021')
O=R/'diagnostics/trigger_exploration_20261021';O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
locked={str(R/p):sha(R/p) for p in ['baseline/final/instance_surface.npz','baseline/surface_evidence.npz','v1/grouping.json','v1/seed_manifest.json','v1/surface_supports.npz','v1/pair_metrics.npz','v2/selection.json','v2/measurements.json']}
cfg=read(C/'config.json');fids=list(range(0,5*n,5))
poses=np.loadtxt(Path(cfg['source']['scene_root'])/cfg['source']['trajectory']).reshape(-1,4,4)
sys.path.insert(0,str(C/'catalog'));from common import project_visible
with np.load(R/'baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id']
with np.load(R/'baseline/surface_evidence.npz') as z:states=z['state'];votes=z['total_frame_votes'];confidence=z['confidence']
with np.load(R/'v1/surface_supports.npz') as z:allidx=z['indices'];offset=z['offsets']
with np.load(R/'v1/pair_metrics.npz') as z:pairiou=z['iou']
g=read(R/'v1/grouping.json');manifest=read(R/'v1/seed_manifest.json');kept=set(read(R/'v2/selection.json')['retained_group_ids'])
sem={x['group_id']:x for x in read(R/'v2/semantic/classification.json')['records']}
coverage={x['group_id']:x for x in read(R/'v2/measurements.json')['coverage']}
first=np.full(len(xyz),n+1,np.int16);last=np.zeros(len(xyz),np.int16);observed=np.zeros(len(xyz),np.int16)
for k,fid in enumerate(fids,1):
 cache=R/f'overmerge/frame{fid:06d}_visible.npz'
 if cache.exists():
  with np.load(cache) as z:idx=z['vertex_index']
 else:
  depth=np.asarray(Image.open(Path(cfg['source']['scene_root'])/cfg['source']['depth_pattern'].format(frame=fid)),dtype=float)/cfg['camera']['depth_png_units_per_meter']
  valid,_=project_visible(xyz,poses[fid],depth,cfg['camera']);idx=np.flatnonzero(valid)
 first[idx]=np.minimum(first[idx],k);last[idx]=k;observed[idx]+=1
 if k%10==0:print('VISIBILITY',n,k,flush=True)
def support(i):return allidx[offset[i]:offset[i+1]]
def angular(centroid,frames):
 rays=np.array([poses[f,:3,3]-centroid for f in frames]);rays/=np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-12)
 angles=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1)))
 bins={}
 for threshold in [3,5,8,10,15]:
  chosen=[]
  for i in range(len(frames)):
   if not chosen or np.min(angles[i,chosen])>=threshold:chosen.append(i)
  bins[str(threshold)]=len(chosen)
 return float(angles.max()) if len(frames) else 0.,bins
stable_pending=(labels<=0)&(observed>=3)&(first<=n-5)
rows=[];union_kept=np.zeros(len(xyz),bool)
source_groups={x['id']:x for x in g['groups']}
for entry in manifest['groups']:
 gid=entry['id'];p=entry['primary'];grp=source_groups[gid];members=grp['members'];primary=support(p['index'])
 union=np.unique(np.concatenate([support(i) for i in members]))
 old=[support(i) for i in members if g['seeds'][i]['frame']<=5*(n-6)]
 oldunion=np.unique(np.concatenate(old)) if old else np.array([],dtype=np.int32)
 frames=sorted(set(grp['frames']));angle,bins=angular(xyz[union].mean(0),frames)
 cleanframes=sorted(set(g['seeds'][i]['frame'] for i in members if not g['seeds'][i]['representative'].get('touches_frame') and not g['seeds'][i]['representative'].get('touches_crop')))
 ciou=[float(pairiou[p['index'],i]) for i in members if i!=p['index'] and g['seeds'][i]['frame']!=p['frame']]
 counts={name:int(np.sum((labels[primary]<=0)&(states[primary]==k))) for name,k in [('U',0),('T',1),('C',3)]}
 pos=labels[primary];idvals,idcounts=np.unique(pos[pos>0],return_counts=True);order=np.argsort(-idcounts);ids=[dict(id=int(idvals[i]),count=int(idcounts[i])) for i in order[:5]]
 native_confirmed=states[primary]==2
 row=dict(gid=gid,mask_id=p['id'],kept=gid in kept,frame=p['frame'],processed=p['frame']//5+1,model=p['representative']['model'],semantic=sem[gid]['top_class'],
  points=len(primary),union_points=len(union),frames=frames,supporting_views=len(frames),clean_views=len(cleanframes),clean_frames=cleanframes,
  primary_touches_frame=bool(p['representative'].get('touches_frame')),primary_touches_crop=bool(p['representative'].get('touches_crop')),
  max_angle_deg=angle,angular_bins=bins,median_primary_iou=float(np.median(ciou)) if ciou else None,min_primary_iou=min(ciou) if ciou else None,
  newest_support_processed=max(frames)//5+1,oldest_support_processed=min(frames)//5+1,
  recently_first_visible_fraction=float(np.mean(first[union]>n-5)),not_depth_visible_fraction=float(np.mean(observed[union]==0)),
  union_new_since_member_window_fraction=1-len(np.intersect1d(oldunion,union,assume_unique=True))/len(union),
  median_depth_observations=float(np.median(observed[union])),unassigned_counts=counts,unassigned=sum(counts.values()),
  stable_pending=int(np.sum(stable_pending[primary])),stable_pending_fraction=float(np.mean(stable_pending[primary])),
  median_raw_votes=float(np.median(votes[primary])),median_vote_confidence=float(np.median(confidence[primary])),
  assigned_ids=ids,second_instance_fraction=float(idcounts[order[1]]/len(primary)) if len(order)>1 else 0.,
  p=coverage.get(gid,{}).get('p'),r=coverage.get(gid,{}).get('r'),coverage=coverage.get(gid))
 rows.append(row)
 if gid in kept:union_kept[primary]=True
with np.load(R/'catalog/room0/membership.npz') as z:roi_core=z['core_point_index']
union_roi=np.zeros(len(xyz),bool);union_roi[roi_core]=True
summary=dict(prefix=n,cutoff=fids[-1],geometry_points=len(xyz),kept=len(kept),
 one_view_kept=[r['gid'] for r in rows if r['kept'] and r['supporting_views']==1],
 under3views_kept=[r['gid'] for r in rows if r['kept'] and r['supporting_views']<3],
 primary_frame_edge_kept=[r['gid'] for r in rows if r['kept'] and r['primary_touches_frame']],
 fewer2cleanviews_kept=[r['gid'] for r in rows if r['kept'] and r['clean_views']<2],
 under5degrees_kept=[r['gid'] for r in rows if r['kept'] and r['max_angle_deg']<5],
 stable_pending_total=int(stable_pending.sum()),stable_pending_in_retained_supports=int(np.sum(stable_pending&union_kept)),
 stable_pending_in_ranked_ROIs=int(np.sum(stable_pending&union_roi)),
 kept_support_union=int(union_kept.sum()),unassigned_in_retained_supports=int(np.sum((labels<=0)&union_kept)),
 unassigned_in_ranked_ROIs=int(np.sum((labels<=0)&union_roi)),unassigned_total=int(np.sum(labels<=0)))
output=dict(status='PASS',purpose='Read-only diagnostic; thresholds are experimental proxies, not validated repair decisions',causal_prefix_only=True,frame_ids=fids,no_GT=True,repair_applied=False,summary=summary,rows=rows,source_sha256=locked)
for p,h in locked.items():assert sha(p)==h,p
(O/'metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
np.savez_compressed(O/'visibility_age.npz',first_processed=first,last_processed=last,observed_frames=observed,stable_pending=stable_pending)
print('METRICS_COMPLETE',json.dumps(summary),flush=True)
