"""Equal-weight two-medoids on ORIGINAL frozen groups; no dedup input."""
import sys, json, hashlib
from pathlib import Path
from itertools import combinations
import numpy as np
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/offline_mask_filters_196_20261023')
from appdata import scene_data
from common import SCENES, sha
OUT=Path('/data/chenkejun/CVPR/results/offline_kmedoids_unified_20261108')
VERSION='equal-kmedoids-v2-unified-20261108'

def medoid(D, inds, incumbent=None):
 costs=D[np.ix_(inds,inds)].sum(axis=1)
 best=[inds[i] for i,c in enumerate(costs) if abs(c-float(costs.min()))<=1e-12]
 return incumbent if incumbent in best else min(best)

def assign(D,centers):
 labels=np.argmin(D[:,centers],axis=1)
 # Each medoid owns itself even when two candidates have identical supports.
 for k,c in enumerate(centers): labels[c]=k
 return labels

def silhouette(D,labels):
 values=[]
 for i,k in enumerate(labels):
  same=np.where(labels==k)[0]; same=same[same!=i]; other=np.where(labels!=k)[0]
  if len(same)==0 or len(other)==0: values.append(0.0);continue
  a=float(D[i,same].mean());b=float(D[i,other].mean());values.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
 return values

def fit(D):
 n=len(D);inds=list(range(n));one=medoid(D,inds)
 one_costs=D.sum(axis=1);one_objective=float(one_costs[one])
 tied=[i for i,cost in enumerate(one_costs) if abs(float(cost)-one_objective)<=1e-12]
 initial=[];iterations=0;trial_s=None;trial_values=[None]*n;trace=[];trial_centers=[];trial_sizes=[];two_objective=None
 if n==1:
  centers=[one];labels=np.zeros(n,dtype=int);reason='singleton'
 else:
  initial=list(min(combinations(range(n),2),key=lambda p:(-D[p],p)));centers=sorted(initial)
  for iterations in range(1,101):
   labels=assign(D,centers);objective=float(D[np.arange(n),np.array(centers)[labels]].sum());trace.append(objective)
   updated=sorted(medoid(D,np.where(labels==k)[0].tolist(),c) for k,c in enumerate(centers))
   if updated==centers:break
   centers=updated
  else:raise AssertionError('K-medoids did not converge')
  labels=assign(D,centers);trial_centers=centers.copy();trial_sizes=[int(np.sum(labels==k)) for k in range(len(centers))]
  two_objective=float(D[np.arange(n),np.array(centers)[labels]].sum())
  # A silhouette exists only for 2 <= number of clusters <= N-1.
  # When no valid score exists, choose the simpler model; never manufacture S=0.
  if 2<=len(set(labels.tolist()))<=n-1:
   trial_values=silhouette(D,labels);trial_s=float(np.mean(trial_values))
  if trial_s is not None and trial_s>0:reason='positive_silhouette'
  else:
   reason='undefined_silhouette' if trial_s is None else 'nonpositive_silhouette'
   centers=[one];labels=np.zeros(n,dtype=int)
 return dict(centers=centers,labels=labels,initial=initial,iterations=iterations,silhouette=trial_s,trial_values=trial_values,reason=reason,trace=trace,
  one=one,one_ties=tied,one_objective=one_objective,two_objective=two_objective,trial_centers=trial_centers,trial_sizes=trial_sizes)

def main():
 OUT.mkdir(parents=True,exist_ok=True); summaries=[];locks={};checks=[]
 for scene in SCENES:
  data=scene_data(scene);s=data['source'];groups={};locks.update(data['locks']);total=kept=two=original_kept=0
  for gid,g in s['groups'].items():
   ids=sorted(g['members']);indices=[s['seeds'][x]['index'] for x in ids];inter=s['inter'][np.ix_(indices,indices)];union=s['union'][np.ix_(indices,indices)]
   support=np.array([s['seeds'][x]['support_count'] for x in ids]);assert np.all(support>0) and np.all(union>0)
   I=inter/union;D=1-I;assert np.allclose(D,D.T) and np.allclose(np.diag(D),0) and np.all((D>=0)&(D<=1))
   f=fit(D);centers=f['centers'];labels=f['labels'];clusters=[];decisions={}
   for k,c in enumerate(centers):
    inds=np.where(labels==k)[0].tolist();assert c in inds
    costs=D[np.ix_(inds,inds)].sum(axis=1);assert abs(float(D[c,inds].sum())-float(costs.min()))<1e-9
    clusters.append(dict(number=k+1,representative=ids[c],members=[ids[i] for i in inds],mean_iou=float(I[c,inds].mean()),singleton=len(inds)==1))
    for i in inds:
     other=[j for j in inds if j!=i]; comps=[]
     for rc in centers:
      lo=int(min(support[i],support[rc]));hi=int(max(support[i],support[rc]));intersection=int(inter[i,rc]);u=int(union[i,rc])
      comps.append(dict(id=ids[rc],iou=float(I[i,rc]),intersection=intersection,union=u,size_ratio=lo/hi,difference=(int(support[i])+int(support[rc])-2*intersection)/lo,self=(i==rc)))
     decisions[ids[i]]=dict(keep=(i==c),cluster=k+1,representative=ids[c],comparisons=comps,mean_other_iou=float(I[i,other].mean()) if other else None,silhouette=f['trial_values'][i],frame=int(data['raw'][ids[i]]['frame']),support=int(support[i]))
   pair=None
   if len(centers)==2:
    a,b=centers;pair=dict(iou=float(I[a,b]),coverage_a=int(inter[a,b])/int(support[a]),coverage_b=int(inter[a,b])/int(support[b]))
   groups[gid]=dict(original_primary=g['primary'],before=len(ids),after=len(centers),clusters=clusters,decisions=decisions,silhouette=f['silhouette'],reason=f['reason'],initial=[ids[i] for i in f['initial']],iterations=f['iterations'],objective_trace=f['trace'],representative_pair=pair,model_comparison=dict(one_representative=ids[f['one']],one_tied_representatives=[ids[i] for i in f['one_ties']],one_mean_distance=f['one_objective']/len(ids),two_mean_distance=f['two_objective']/len(ids) if f['two_objective'] is not None else None,trial_two_representatives=[ids[i] for i in f['trial_centers']],trial_two_sizes=f['trial_sizes'],silhouette_valid=f['silhouette'] is not None,selection='S>0 only when defined; otherwise prefer one cluster; cost is diagnostic only'))
   total+=len(ids);kept+=len(centers);two+=int(len(centers)==2);original_kept+=int(decisions[g['primary']]['keep'])
  summary=dict(scene=scene,groups=len(groups),candidates=total,kept=kept,not_selected=total-kept,single_groups=len(groups)-two,two_groups=two,original_primary_kept=original_kept)
  result=dict(version=VERSION,scene=scene,summary=summary,groups=groups,rules=dict(input='all members of original v1/grouping.json',previous_dedup_used=False,weights='all 1',distance='1 - full-support 3D IoU',initialization='farthest pair; ties by mask ID',update='equal-weight medoid; retain incumbent on ties, else mask ID',silhouette='For valid partitions: unweighted mean, singleton-member s=0; S>0 keeps 2. S<=0 or undefined selects global medoid; undefined scores stay null.',small_groups='All N>=2 fit K=1 and K=2 and follow the same selection rule; N=1 has only K=1',tie_break='Global medoid ties: smallest mask ID; no quality advantage implied',iou_cutoff=None,selection_limit='Conservative fallback selects K=1 whenever silhouette is undefined; thus all N=2 groups select one regardless of pair distance; this is a policy, not evidence that the masks are equivalent.',stage='offline screening replay; no tracking, repair, evaluation'))
  tmp=OUT/(scene+'.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(OUT/(scene+'.json'));summaries.append(summary);print(json.dumps(summary),flush=True)
 assert all(sha(p)==h for p,h in locks.items()),'source changed'
 audit=dict(status='PASS',version=VERSION,source_locks=locks,code_sha256=sha(__file__),groups=sum(x['groups'] for x in summaries),candidates=sum(x['candidates'] for x in summaries),input_unchanged=True)
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');(OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
if __name__=='__main__':main()
