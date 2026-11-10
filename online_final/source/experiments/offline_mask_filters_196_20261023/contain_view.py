from functools import lru_cache
from common import *
ROOT=Path('/data/chenkejun/CVPR/results/offline_medoids_containment_groups040_20261108')
@lru_cache(maxsize=8)
def result(scene):
 assert scene in SCENES
 return read(ROOT/scene/'result.json')
def metadata():
 s=read(ROOT/'summary.json')
 return dict(scenes={x['scene']:x for x in s},input=sum(x['input'] for x in s),retained=sum(x['retained'] for x in s),excluded=0,groups=sum(x['groups'] for x in s),mode='grouping_only')
def rows(scene,page=0,query='',status='all'):
 assert status in ('all','related','singleton','chain','new')
 r=result(scene);byid={x['mask_id']:x for x in r['rows']};q=query.strip().lower();groups=[]
 for group in r['groups']:
  members=[byid[m] for m in group['members']]
  if status=='related' and not group['edges']:continue
  if status=='singleton' and group['edges']:continue
  if status=='new' and not group['new_edges']:continue
  if status=='chain' and not group['chain']:continue
  if q and q!=group['id'].lower() and not any(q in ' '.join(str(x[k]) for k in ('mask_id','group_id','frame','top_class')).lower() for x in members):continue
  clusters=[dict(original_group=c['original_group'],cluster=c['cluster'],representatives=[byid[m] for m in c['representatives']]) for c in group['medoids_clusters']]
  groups.append(dict(id=group['id'],members=members,medoids_clusters=clusters,edges=group['edges'],chain=group['chain'],previous_group_ids=group['previous_group_ids'],new_edges=group['new_edges']))
 pages=max(1,(len(groups)+3)//4);page=max(0,min(page,pages-1))
 return dict(groups=groups[page*4:(page+1)*4],summary=r['summary'],total=len(groups),total_masks=sum(len(g['members']) for g in groups),page=page,pages=pages)
def detail(scene,mid):
 r=result(scene);row=next(x for x in r['rows'] if x['mask_id']==mid);pairs=[p for p in r['pairs'] if mid in (p['a'],p['b'])];pairs.sort(key=lambda p:(not p['passes'],-p['ratio']))
 return dict(row=row,pairs=pairs,summary=r['summary'])
