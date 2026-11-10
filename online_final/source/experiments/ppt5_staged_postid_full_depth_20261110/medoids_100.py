from pathlib import Path
import sys,time,json,hashlib,importlib.util,traceback
import numpy as np
scene=sys.argv[1];root=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110/online')/scene/'100/B/candidates';out=root/'medoids';out.mkdir(exist_ok=True);state=out/'status.json'
def dump(x):
 q=state.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(state)
try:
 p=root/'group_runtime/driver_status.json'
 while True:
  st=json.loads(p.read_text())['status'] if p.exists() else 'WAITING'
  if st=='PASS':break
  if st=='FAILED':raise RuntimeError('original grouping failed')
  dump(dict(status='WAITING_GROUPS'));time.sleep(20)
 started=time.time();dump(dict(status='RUNNING',stage='equal_weight_medoids'))
 spec=importlib.util.spec_from_file_location('medoids_v2','/home/chenkejun/CVPR/experiments/offline_kmedoids_unified_20261108/replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 source=root/'v1/grouping.json';data=json.loads(source.read_text());seeds=data['seeds'];model_file=Path('/data/chenkejun/CVPR/results/offline_kmedoids_learned_20261108/models.json');model=json.loads(model_file.read_text())[scene];assert scene not in model['training_scenes']
 with np.load(root/'v1/pair_metrics.npz') as z:inter=z['intersection'];union=z['union']
 assert inter.shape==union.shape==(len(seeds),len(seeds));groups=[];allids=set();counts={'one':0,'two':0,'input':0,'kept':0,'learned_pairs':0}
 for g in data['groups']:
  indices=sorted(g['members'],key=lambda i:seeds[i]['id']);ids=[seeds[i]['id'] for i in indices];assert not(allids&set(ids));allids.update(ids)
  I=inter[np.ix_(indices,indices)]/union[np.ix_(indices,indices)];D=1-I;assert np.allclose(D,D.T) and np.allclose(np.diag(D),0) and np.all(np.isfinite(D))
  f=module.fit(D);centers=f['centers'];labels=f['labels'];reason=f['reason'];learning=None
  if len(ids)==2:
   z=model['coef']*float(D[0,1])+model['intercept'];p_diff=1/(1+np.exp(-z));two=p_diff>.5;learning=dict(iou=float(I[0,1]),different_score=float(p_diff),threshold=.5,model_target_scene=scene,training_scenes=model['training_scenes'],trained_before_target_checkpoint=True)
   if two:centers=[0,1];labels=np.arange(2);reason='learned_different'
   else:reason='learned_same'
   counts['learned_pairs']+=1
  clusters=[]
  for k,c in enumerate(centers):
   inds=np.flatnonzero(labels==k).tolist();assert c in inds;clusters.append(dict(number=k+1,representative=ids[c],members=[ids[i] for i in inds],mean_iou=float(I[c,inds].mean())))
  assert sum(len(q['members']) for q in clusters)==len(ids)
  groups.append(dict(group_id=g['id'],original_primary=seeds[g['primary']]['id'],clusters=clusters,before=len(ids),after=len(centers),reason=reason,silhouette=f['silhouette'],pair_learning=learning))
  counts['input']+=len(ids);counts['kept']+=len(centers);counts['one' if len(centers)==1 else 'two']+=1
 assert len(allids)==len(seeds) and counts['kept']<=2*len(groups)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();result=dict(status='PASS',scene=scene,checkpoint=100,group='B',method='equal weighted K-medoids with previously frozen leave-one-scene-out two-mask classifier',source_grouping=str(source),source_grouping_sha256=sha(source),pair_metrics_sha256=sha(root/'v1/pair_metrics.npz'),classifier_sha256=sha(model_file),groups=groups,summary=counts,seconds=time.time()-started,GT_used=False,tracking_or_repair=False)
 (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');dump(dict(status='PASS',summary=counts,seconds=result['seconds'],tracking_or_repair=False))
except Exception:dump(dict(status='FAILED',traceback=traceback.format_exc()));raise
