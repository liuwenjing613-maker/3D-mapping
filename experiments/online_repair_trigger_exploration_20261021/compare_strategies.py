"""Compare diagnostic gates without applying repairs. Cross-prefix results are retrospective only."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from scipy.spatial import cKDTree
O=Path('/data/chenkejun/CVPR/results/online_repair_trigger_exploration_20261021');O.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
def load(n):
 R=Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021')
 d=read(R/'diagnostics/trigger_exploration_20261021/metrics.json');g=read(R/'v1/grouping.json');man=read(R/'v1/seed_manifest.json');cfg=read(Path(f'/home/chenkejun/CVPR/experiments/online_room0_prefix{n}_20261021/config.json'))
 with np.load(R/'v1/surface_supports.npz') as z:idx=z['indices'];off=z['offsets']
 with np.load(R/'v1/pair_metrics.npz') as z:iou=z['iou']
 with np.load(R/'baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id']
 with np.load(R/'baseline/surface_evidence.npz') as z:state=z['state']
 with np.load(R/'diagnostics/trigger_exploration_20261021/visibility_age.npz') as z:stable=z['stable_pending']
 poses=np.loadtxt(Path(cfg['source']['scene_root'])/cfg['source']['trajectory']).reshape(-1,4,4)
 sup={x['id']:idx[off[x['primary']['index']]:off[x['primary']['index']+1]] for x in man['groups']}
 return dict(R=R,data=d,groups={x['id']:x for x in g['groups']},seeds=g['seeds'],manifest={x['id']:x for x in man['groups']},indices=idx,offset=off,iou=iou,xyz=xyz,labels=labels,state=state,stable=stable,poses=poses,supports=sup)
def quorum(v,row,iou_threshold=.7,angle_threshold=5):
 g=v['groups'][row['gid']];p=v['manifest'][row['gid']]['primary'];anchor=p['index']
 clean=[i for i in g['members'] if not v['seeds'][i]['representative'].get('touches_frame') and not v['seeds'][i]['representative'].get('touches_crop')]
 if anchor not in clean:return None
 center=v['xyz'][v['supports'][row['gid']]].mean(0)
 others=[i for i in clean if i!=anchor and v['seeds'][i]['frame']!=p['frame']]
 best=None
 for a,b in itertools.combinations(others,2):
  selected=[anchor,a,b];frames=[v['seeds'][i]['frame'] for i in selected]
  if len(set(frames))<3:continue
  overlap=min(v['iou'][x,y] for x,y in itertools.combinations(selected,2))
  if overlap<iou_threshold:continue
  rays=np.array([v['poses'][f,:3,3]-center for f in frames]);rays/=np.linalg.norm(rays,axis=1,keepdims=True)
  angle=float(np.degrees(np.arccos(np.clip(rays@rays.T,-1,1))).max())
  if angle<angle_threshold:continue
  rec=dict(frames=frames,mask_ids=[v['seeds'][i]['id'] for i in selected],min_pair_iou=float(overlap),max_angle_deg=angle)
  if best is None or (overlap,angle)>(best['min_pair_iou'],best['max_angle_deg']):best=rec
 return best
loaded={n:load(n) for n in [20,50]}
out=dict(kind='diagnostic_comparison_not_repair_evaluation',no_GT=True,repair_applied=False,thresholds_not_calibrated=True,prefixes={})
for n,v in loaded.items():
 rows=[r for r in v['data']['rows'] if r['kept']]
 shared={}
 for r in rows:shared.setdefault(r['coverage']['instance_id'],[]).append(r['gid'])
 ambiguous=set(g for ids in shared.values() if len(ids)>1 for g in ids)
 for r in rows:
  r['quorum']=quorum(v,r);r['possible_same_old_ID_whole_part_or_merge_ambiguity']=r['gid'] in ambiguous
  r['need_proxy']=r['stable_pending']>=max(20,.02*r['points'])
  r['age_proxy_ready']=r['recently_first_visible_fraction']<=.1
  r['maturity_proxy_ready']=bool(r['quorum']) and r['age_proxy_ready']
  r['recommend_action_proxy']='WAIT_VIEW' if not r['maturity_proxy_ready'] else 'REVIEW_IDENTITY' if r['gid'] in ambiguous else 'SHADOW_CANDIDATE' if r['need_proxy'] else 'LOW_GAIN_MONITOR'
 ids={r['gid'] for r in rows}
 policies={
 'fixed_frame_all':ids,
 'count_3_views':{r['gid'] for r in rows if r['supporting_views']>=3},
 'quorum_maturity':{r['gid'] for r in rows if r['maturity_proxy_ready']},
 'maturity_need_unambiguous':{r['gid'] for r in rows if r['recommend_action_proxy']=='SHADOW_CANDIDATE'}}
 summaries={}
 for name,selected in policies.items():
  mask=np.zeros(len(v['xyz']),bool)
  for gid in selected:mask[v['supports'][gid]]=True
  summaries[name]=dict(count=len(selected),ids=sorted(selected),stable_pending_union=int(np.sum(mask&v['stable'])),
   single_view=sum(r['gid'] in selected and r['supporting_views']==1 for r in rows),
   under5degrees=sum(r['gid'] in selected and r['max_angle_deg']<5 for r in rows),
   primary_touches_frame=sum(r['gid'] in selected and r['primary_touches_frame'] for r in rows))
 sweep=[]
 for iou in [.6,.7,.8]:
  for angle in [3,5,10]:
   selected=[r['gid'] for r in rows if quorum(v,r,iou,angle) and r['age_proxy_ready']]
   sweep.append(dict(iou=iou,angle=angle,count=len(selected),ids=selected))
 # Diagnostic example: global 90% coverage drops small independent repair targets.
 eligible=policies['maturity_need_unambiguous'];target=np.zeros(len(v['xyz']),bool)
 for gid in eligible:target[v['supports'][gid]]=True
 target &= v['stable'];covered=np.zeros(len(target),bool);chosen=[]
 while target.sum() and np.sum(covered&target)<.9*target.sum():
  options=[(int(np.sum(target[v['supports'][gid]]&~covered[v['supports'][gid]])),gid) for gid in eligible-set(chosen)]
  if not options:break
  gain,gid=max(options)
  if not gain:break
  chosen.append(gid);covered[v['supports'][gid]]=True
 out['prefixes'][str(n)]=dict(summary=v['data']['summary'],policies=summaries,rows=rows,sensitivity=sweep,
  actions={name:[r['gid'] for r in rows if r['recommend_action_proxy']==name] for name in ['WAIT_VIEW','REVIEW_IDENTITY','SHADOW_CANDIDATE','LOW_GAIN_MONITOR']},
  global90_example=dict(chosen=chosen,omitted=sorted(eligible-set(chosen)),target_points=int(target.sum()),covered_points=int(np.sum(target&covered))),
  decision_features_sha256=hashlib.sha256((v['R']/'diagnostics/trigger_exploration_20261021/metrics.json').read_bytes()).hexdigest())
# This block is retrospective checking only; no result is used by the prefix policies above.
a,b=loaded[20],loaded[50];tree=cKDTree(b['xyz']);distance,nn=tree.query(a['xyz'],workers=2)
checks=[]
for ra in out['prefixes']['20']['rows']:
 gid=ra['gid'];sa=a['supports'][gid];mapped=np.unique(nn[sa][distance[sa]<=.015]);rank=[]
 for rb in out['prefixes']['50']['rows']:
  sb=b['supports'][rb['gid']];common=len(np.intersect1d(mapped,sb,assume_unique=True));rank.append(dict(gid=rb['gid'],intersection=common,old_fraction=common/max(1,len(mapped)),new_fraction=common/len(sb),iou=common/max(1,len(mapped)+len(sb)-common)))
 rank.sort(key=lambda x:-x['intersection'])
 checks.append(dict(prefix20_gid=gid,mapped_points=len(mapped),original_points=len(sa),nearest_surface_coverage=len(mapped)/len(sa),best=rank[:3]))
out['retrospective_20_to_50']=dict(tolerance_m=.015,not_used_for_any_trigger=True,not_ground_truth=True,rows=checks)
(O/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({n:dict(policies=d['policies'],actions=d['actions'],global90_example=d['global90_example']) for n,d in out['prefixes'].items()},ensure_ascii=False))
print('RETRO_CABINET',next(x for x in checks if x['prefix20_gid']=='G004'))
