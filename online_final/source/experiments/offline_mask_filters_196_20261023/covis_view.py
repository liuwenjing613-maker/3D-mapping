from functools import lru_cache
from common import *
ROOT=Path('/data/chenkejun/CVPR/results/offline_medoids_covis080_20261108')
@lru_cache(maxsize=8)
def result(scene):
 assert scene in SCENES
 return read(ROOT/scene/'result.json')
def metadata():
 s=read(ROOT/'summary.json')
 return dict(scenes={x['scene']:x for x in s},input=sum(x['input'] for x in s),retained=sum(x['retained'] for x in s),excluded=sum(x['excluded'] for x in s))
def rows(scene,page=0,query='',status='all'):
 assert status in ('all','retain','exclude')
 r=result(scene);q=query.lower().strip()
 byid={x['mask_id']:x for x in r['rows']};chosen=[]
 for i,members in enumerate(r['groups'],1):
  group=dict(id=f'C{i:03d}',representative=members[0],members=[byid[m] for m in members])
  if status=='exclude' and len(members)==1:continue
  if status=='retain' and len(members)!=1:continue
  if q and not any(q in ' '.join(str(x[k]) for k in ('mask_id','group_id','frame','top_class','covis_representative')).lower() for x in group['members']) and q!=group['id'].lower():continue
  chosen.append(group)
 pages=max(1,(len(chosen)+5)//6);page=max(0,min(page,pages-1))
 return dict(groups=chosen[page*6:(page+1)*6],summary=r['summary'],total=len(chosen),total_masks=sum(len(g['members']) for g in chosen),page=page,pages=pages)
def detail(scene,mid):
 r=result(scene);row=next(x for x in r['rows'] if x['mask_id']==mid)
 pairs=[p for p in r['pairs'] if mid in (p['a'],p['b'])];pairs.sort(key=lambda p:(not p['visited'],-(p['iou'] or 0)))
 return dict(row=row,pairs=pairs)
