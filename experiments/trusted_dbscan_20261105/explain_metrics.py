from pathlib import Path
import json,numpy as np,sys
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
from unified_eval.metrics import _matching
R=Path('/data/chenkejun/CVPR/results/trusted_dbscan_20261105');D=json.loads((R/'comparison/comparison.json').read_text())
B=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/ablations/dbscan_history_covisible_20261102/online_B')
S=[json.loads((R/'comparison/pooled'/a/'summary.json').read_text()) for a in ['old_dbscan','trusted_dbscan']]
rows=[];sceneRows=[];predRows=[]
for scene in [x['scene'] for x in D['per_scene']]:
 zs=[np.load(B/scene/'r2/overlap_matrix.npz'),np.load(R/'candidate'/scene/'r2/overlap_matrix.npz')]
 matches=[{c:r for r,c in _matching(z['iou'],.5,True)} for z in zs]
 own=[]
 for z in zs:
  a,b=linear_sum_assignment(-z['intersection']);own.append({int(c):int(z['intersection'][r,c]) for r,c in zip(a,b) if z['intersection'][r,c]>0})
 for j,g in enumerate(zs[0]['gt_ids']):
  source=next(r for r in D['per_gt'] if r['scene']==scene and r['raw_gt_id']==int(g))
  r=dict(source);r['size']=int(zs[0]['gt_size'][j]);r['TP_before']=j in matches[0];r['TP_after']=j in matches[1];r['correct_before']=own[0].get(j,0);r['correct_after']=own[1].get(j,0);r['correct_delta']=r['correct_after']-r['correct_before']
  r['best_ids']=[str(z['pred_uids'][np.argmax(z['iou'][:,j])]) for z in zs]
  r['significant_parts']=[int(np.sum((z['intersection'][:,j]>=10)&(z['intersection'][:,j]/z['gt_size'][j]>=.05)&(~z['unmatched_ignore_mask']))) for z in zs]
  rows.append(r)
 for arm,z in enumerate(zs):
  mr=set(matches[arm].values())
  for i,uid in enumerate(z['pred_uids']):
   if i in mr or bool(z['unmatched_ignore_mask'][i]):continue
   j=int(np.argmax(z['intersection'][i]));inter=int(z['intersection'][i,j]);size=int(z['pred_size'][i])
   predRows.append(dict(scene=scene,arm=arm,uid=str(uid),size=size,target_points=int(z['intersection'][i].sum()),best_iou=float(z['iou'][i].max()),dominant_gt=int(z['gt_ids'][j]),dominant_intersection=inter,dominant_fraction=inter/size if size else 0))
 for z in zs:z.close()
deltas=np.array([r['policy_dbscan_delta'] for r in rows])
counts={}
for t in [0,.001,.01,.02,.05,.1]:
 counts[str(t)]={'gain':int(np.sum(deltas>t+1e-12)),'loss':int(np.sum(deltas<-t-1e-12))}
pr=[s['CA_PRF1_0_5'] for s in S];n=[p['TP']+p['FP'] for p in pr];t=[p['TP'] for p in pr]
f=lambda tp,n:2*tp/(350+n)
shap={'TP_count_effect_pp':100*((f(t[1],n[0])-f(t[0],n[0]))+(f(t[1],n[1])-f(t[0],n[1])))/2,'prediction_count_effect_pp':100*((f(t[0],n[1])-f(t[0],n[0]))+(f(t[1],n[1])-f(t[1],n[0])))/2}
groups={}
for name,keep in [('without_room2',lambda s:s!='room2'),('without_room2_office3_office4',lambda s:s not in ['room2','office3','office4'])]:
 out=[]
 for s in S:
  pp=[r['CA_PRF1_0_5'] for r in s['per_scene'] if keep(r['scene_id'])];tp=sum(r['TP'] for r in pp);fp=sum(r['FP'] for r in pp);fn=sum(r['FN'] for r in pp);out.append(dict(TP=tp,FP=fp,FN=fn,F1=2*tp/(2*tp+fp+fn)))
 groups[name]=out
fps=[]
for arm in [0,1]:
 rr=[r for r in predRows if r['arm']==arm];fps.append({'count':len(rr),'tiny_lt100':sum(r['size']<100 for r in rr),'size_median':float(np.median([r['size'] for r in rr])),'pure_gt_fragment_fraction_gt80':sum(r['dominant_fraction']>.8 for r in rr),'ignored':S[arm]['CA_PRF1_0_5']['ignored_prediction_count']})
result=dict(status='PASS',scope='same_DBSCAN_existing_outputs_read_only',F1_count_shapley=shap,delta_counts=counts,positive_IoU_sum=float(deltas[deltas>0].sum()),negative_IoU_sum=float(deltas[deltas<0].sum()),median_delta=float(np.median(deltas)),TP_gains=[r for r in rows if not r['TP_before'] and r['TP_after']],TP_losses=[r for r in rows if r['TP_before'] and not r['TP_after']],severe_still_TP=[r for r in rows if r['severe_vs_old_dbscan'] and r['TP_before'] and r['TP_after']],counts=pr,PQ=[s['CA_PQ'] for s in S],owner=[s['owner_surface'] for s in S],structure=[s['structure'] for s in S],AP_by_threshold=[s['AP_by_threshold'] for s in S],macro=[s['macro_per_scene'] for s in S],subsets=groups,FP_shape=fps,per_gt=rows,unmatched_predictions=predRows)
out=R/'diagnostics/metric_paradox';out.mkdir(exist_ok=True);(out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
compact={k:v for k,v in result.items() if k not in ['per_gt','unmatched_predictions','owner']}
for k in ['TP_gains','TP_losses','severe_still_TP']:compact[k]=[{a:r[a] for a in ['scene','raw_gt_id','semantic_class_zh','size','old_dbscan','trusted_dbscan','correct_delta','significant_parts']} for r in result[k]]
compact['owner']=[{k:v for k,v in s['owner_surface'].items() if not isinstance(v,list)} for s in S]
print(json.dumps(compact,ensure_ascii=False))
