from functools import lru_cache
from common import *
ROOT=Path('/data/chenkejun/CVPR/results/offline_medoids_coverage9085_20261108')
GROUP_ROOT=Path('/data/chenkejun/CVPR/results/offline_medoids_containment_groups040_20261108')
@lru_cache(maxsize=8)
def result(scene):
 assert scene in SCENES
 return read(ROOT/scene/'result.json')
@lru_cache(maxsize=8)
def relation_groups(scene):
 assert scene in SCENES
 g=read(GROUP_ROOT/scene/'result.json');r=result(scene)
 assert {x['mask_id'] for x in g['rows']}=={x['mask_id'] for x in r['rows']}
 assert len({m for group in g['groups'] for m in group['members']})==len(r['rows'])
 return g
def metadata():
 s=read(ROOT/'summary.json');gs={x['scene']:x for x in read(GROUP_ROOT/'summary.json')}
 return dict(scenes={x['scene']:dict(**x,display_groups=gs[x['scene']]['groups']) for x in s},input=sum(x['input'] for x in s),retained=sum(x['retained'] for x in s),excluded=sum(x['excluded'] for x in s),newly_excluded=sum(x['newly_excluded'] for x in s),rule='min>=0.85 and max>=0.90',containment_skipped=True,display_grouping='containment0.4',display_groups=sum(x['groups'] for x in gs.values()))
def rows(scene,page=0,query='',status='all'):
 assert status in ('all','retain','exclude','unmatched','new')
 r=result(scene);relations=relation_groups(scene);byid={x['mask_id']:x for x in r['rows']};q=query.strip().lower();groups=[];matched=0
 for g in relations['groups']:
  members=[byid[m] for m in g['members']]
  def match(x):
   return (status=='all' or x['dual95_decision']==status or status=='unmatched' and x['coverage']['instance_id'] is None or status=='new' and x['newly_excluded']) and (not q or q==g['id'].lower() or q in ' '.join(str(x[k]) for k in ('mask_id','group_id','frame','top_class')).lower() or q==str(x['coverage']['instance_id']))
  hits=[x for x in members if match(x)]
  if not hits:continue
  matched+=len(hits)
  groups.append(dict(id=g['id'],members=members,matched_ids=[x['mask_id'] for x in hits],medoids_clusters=g['medoids_clusters'],edges=g['edges']))
 pages=max(1,(len(groups)+3)//4);page=max(0,min(page,pages-1))
 return dict(groups=groups[page*4:(page+1)*4],summary=r['summary'],display_groups=relations['summary']['groups'],grouping_threshold=.4,total=len(groups),matched=matched,total_masks=sum(len(g['members']) for g in groups),page=page,pages=pages)
def detail(scene,mid):return next(r for r in result(scene)['rows'] if r['mask_id']==mid)
def image_bytes(scene,mid):
 r=detail(scene,mid);p=(ROOT/scene/r['instance_image']).resolve();assert p.is_relative_to((ROOT/scene/'previews').resolve()) and p.suffix=='.jpg'
 return p.read_bytes()
