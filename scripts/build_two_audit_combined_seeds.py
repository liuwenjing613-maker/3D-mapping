"""Combine original/current final repair seeds under the original fixed-primary rule."""
import json,hashlib,importlib.util
from pathlib import Path
from itertools import combinations
import numpy as np
ROOT=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104')
OUT=ROOT/'stable_seed_filter_v1'
REVIEW=OUT/'review'
OLD=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/main')
CODE=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/main')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
locks={}
def lock(p):locks[str(p)]=sha(p)
original=read(REVIEW/'original.json');current=read(REVIEW/'data.json')
for n in ['original.json','data.json']:lock(REVIEW/n)
result=dict(status='PASS',rule='ordinary 3D IoU > 0.25; fixed primary; descending support, frame, model, digest; no chaining',scenes={},groups=[])
for scene in ['room0','office2']:
 rulepath=CODE/scene/'selection_code/400/A/v1/grouping_rules.py';lock(rulepath)
 spec=importlib.util.spec_from_file_location('rule',rulepath);rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
 rows=[];supports={};geometry=OLD/scene/'400/A/candidates/baseline/final/instance_surface.npz'
 gh=sha(geometry);lock(geometry)
 for source,base,views in [('original',OLD/scene/'400/A/candidates',original['rows']),('current',OUT/scene,current['rows'])]:
  receipt=read(base/'final_abc/receipt.json');context=receipt['context']
  manifest=read(base/'final_abc/seed_manifest.json');pool=read(context['grouping'])
  for p in [base/'final_abc/seed_manifest.json',base/'final_abc/receipt.json',Path(context['grouping']),Path(context['supports'])]:lock(p)
  assert pool['source_sha256'][str(geometry)]==gh
  assert pool['visibility_rule']['depth_tolerance']==.02
  seeds={s['id']:s for s in pool['seeds']}
  with np.load(context['supports']) as z:indices=z['indices'];offsets=z['offsets']
  for g in manifest['groups']:
   p=g['primary'];s=seeds[p['id']];a=indices[offsets[s['index']]:offsets[s['index']+1]]
   assert len(a)==s['support_count'] and sha(Path(context['mask_root'])/s['mask'])==s['mask_sha256']
   assert hashlib.sha256(a.tobytes()).hexdigest()==s['support_sha256']
   assert len(np.unique(a))==len(a)
   v=next(v for v in views if v['scene']==scene and v['id']==p['id'])
   if source=='current':assert v['final']
   key=source+':'+p['id'];supports[key]=a
   rows.append(dict(key=key,source=source,id=p['id'],label=v.get('track',p['id']),frame=s['frame'],support=len(a),model=s['representative']['model'],digest=s['digest'],image=v['image'],thumb=v.get('thumb',v['image']+'&thumb=1'),semantic=v['semantic'] if source=='original' else (v.get('semantic') or {}).get('top_class',''),old_group=g['id']))
  assert sum(r['source']==source for r in rows)==(original['scenes'][scene] if source=='original' else current['scenes'][scene]['final'])
 rows.sort(key=lambda r:(-r['support'],r['frame'],r['model'],r['digest'],r['key']))
 by={r['key']:r for r in rows};pairs={}
 for a,b in combinations(by,2):
  n=int(np.intersect1d(supports[a],supports[b],assume_unique=True).size)
  u=by[a]['support']+by[b]['support']-n
  pairs[tuple(sorted([a,b]))]=dict(a=a,b=b,intersection=n,union=u,iou=n/u if u else 0,passes=bool(rule.same_instance(n,by[a]['support'],by[b]['support'])))
 grouped=rule.group_seeds(list(by),lambda a,b:pairs[tuple(sorted([a,b]))]['passes'])
 assert sorted(x for g in grouped for x in g)==sorted(by)
 scene_groups=[]
 for i,keys in enumerate(grouped,1):
  members=[]
  for k in keys:
   r=dict(by[k]);r['primary']=k==keys[0];r['evidence']=None if r['primary'] else pairs[tuple(sorted([keys[0],k]))]
   if r['evidence']:assert r['evidence']['passes']
   members.append(r)
  group=dict(scene=scene,id='G%03d'%i,primary=keys[0],members=members,mixed=len({r['source'] for r in members})==2)
  scene_groups.append(group)
 # Independently replay every assignment against preceding fixed primaries.
 for i,g in enumerate(grouped):
  for k in g:
   assert all(not pairs[tuple(sorted([k,prev[0]]))]['passes'] for prev in grouped[:i])
 result['groups']+=scene_groups
 result['scenes'][scene]=dict(original=sum(r['source']=='original' for r in rows),current=sum(r['source']=='current' for r in rows),groups=len(grouped),mixed=sum(g['mixed'] for g in scene_groups),singletons=sum(len(g)==1 for g in grouped))
 write(OUT/scene/'combined_seed_pair_metrics.json',list(pairs.values()))
 print(scene,result['scenes'][scene],flush=True)
for p,h in locks.items():assert sha(p)==h
write(REVIEW/'combined.json',result)
write(OUT/'combined_seed_validation.json',dict(status='PASS',all_inputs_assigned_once=True,fixed_primary_replayed=True,original_rule_reused=True,geometry_and_support_hashes_verified=True,map_modified=False,source_locks=locks,summary=result['scenes']))
