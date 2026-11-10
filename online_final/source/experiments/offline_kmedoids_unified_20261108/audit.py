import sys,json
from pathlib import Path
import numpy as np
from replay import fit
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/offline_mask_filters_196_20261023')
from appdata import scene_data
from common import SCENES,sha
OUT=Path('/data/chenkejun/CVPR/results/offline_kmedoids_unified_20261108')
OLD=Path('/data/chenkejun/CVPR/results/offline_kmedoids_equal_20261108')

# Even distinct two-member groups follow the explicit undefined-score fallback.
for distance in [0,.1,.8,1]:
 f=fit(np.array([[0,distance],[distance,0]]))
 assert f['centers']==[0] and f['one_ties']==[0,1]
 assert f['silhouette'] is None and f['trial_values']==[None,None]
 assert f['reason']=='undefined_silhouette'
 assert f['two_objective']==0 and f['one_objective']==distance
assert fit(np.zeros((1,1)))['centers']==[0]
assert len(fit(np.zeros((4,4)))['centers'])==1
D=np.ones((5,5))*.9;D[:3,:3]=.1;D[3:,3:]=.1;np.fill_diagonal(D,0)
assert len(fit(D)['centers'])==2

checks=[];changed=[]
for scene in SCENES:
 data=scene_data(scene);source=data['source']
 result=json.loads((OUT/(scene+'.json')).read_text())
 old=json.loads((OLD/(scene+'.json')).read_text())
 assert set(result['groups'])==set(source['groups'])
 for gid,g in result['groups'].items():
  ids=sorted(source['groups'][gid]['members']);n=len(ids)
  assert sorted(x for c in g['clusters'] for x in c['members'])==ids
  assert set(g['decisions'])==set(ids)
  def distance(a,b):
   i,j=source['seeds'][a]['index'],source['seeds'][b]['index']
   return 1-float(source['inter'][i,j])/float(source['union'][i,j])
  one_costs={x:sum(distance(x,y) for y in ids) for x in ids}
  best=min(one_costs.values());ties=[x for x in ids if abs(one_costs[x]-best)<=1e-12]
  m=g['model_comparison']
  assert m['one_representative']==min(ties) and m['one_tied_representatives']==ties
  assert abs(m['one_mean_distance']-best/n)<1e-12
  reps=[c['representative'] for c in g['clusters']]
  assert sum(d['keep'] for d in g['decisions'].values())==g['after']==len(reps)
  for c in g['clusters']:
   members=c['members'];rep=c['representative'];assert rep in members
   costs={x:sum(distance(x,y) for y in members) for x in members}
   assert abs(costs[rep]-min(costs.values()))<1e-9
   for x in members:
    d=g['decisions'][x];assert d['keep']==(x==rep) and d['representative']==rep
    assert distance(x,rep)<=min(distance(x,r) for r in reps)+1e-12
    for p in d['comparisons']:assert abs(p['iou']-(1-distance(x,p['id'])))<1e-12
  trial=m['trial_two_representatives']
  if trial:
   labels={x:min(range(2),key=lambda k:distance(x,trial[k])) for x in ids}
   for k,r in enumerate(trial):labels[r]=k
   sizes=[sum(k==v for v in labels.values()) for k in range(2)]
   assert m['trial_two_sizes']==sizes
   assert abs(m['two_mean_distance']-sum(distance(x,trial[labels[x]]) for x in ids)/n)<1e-12
   if 2<=len(set(labels.values()))<=n-1:
    values=[]
    for x in ids:
     own=[y for y in ids if labels[y]==labels[x] and y!=x]
     other=[y for y in ids if labels[y]!=labels[x]]
     if not own:s=0
     else:
      a=sum(distance(x,y) for y in own)/len(own);b=sum(distance(x,y) for y in other)/len(other)
      s=(b-a)/max(a,b) if max(a,b) else 0
     values.append(s);assert abs(g['decisions'][x]['silhouette']-s)<1e-12
    score=sum(values)/n;assert abs(g['silhouette']-score)<1e-12
    assert g['after']==(2 if score>0 else 1)
   else:
    assert g['silhouette'] is None and not m['silhouette_valid'] and g['after']==1
    assert all(d['silhouette'] is None for d in g['decisions'].values())
  if g['after']==1:assert reps==[min(ties)]
  if n==2:
   assert g['after']==1 and old['groups'][gid]['after']==2
   changed.append(dict(scene=scene,group=gid,representative=reps[0]))
  else:
   assert g['clusters']==old['groups'][gid]['clusters']
  trace=g['objective_trace'];assert all(b<=a+1e-9 for a,b in zip(trace,trace[1:]))
 s=result['summary'];assert s['kept']==sum(g['after'] for g in result['groups'].values())
 assert s['candidates']==sum(g['before'] for g in result['groups'].values())
 checks.append(dict(scene=scene,status='PASS',summary=s));print(scene,'PASS',s,flush=True)
audit=json.loads((OUT/'audit.json').read_text())
assert all(sha(p)==h for p,h in audit['source_locks'].items())
audit.update(independent_validation=checks,edge_cases='PASS',changed_groups=changed,stage='offline screening replay only; no tracking, repair or evaluation')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
print('TOTAL_CHANGED',len(changed),'TOTAL_REPRESENTATIVES',sum(c['summary']['kept'] for c in checks))
