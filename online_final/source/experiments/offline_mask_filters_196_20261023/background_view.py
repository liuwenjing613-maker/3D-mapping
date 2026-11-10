"""Read-only presentation of medoid representatives after semantic screening."""
from functools import lru_cache
from pathlib import Path
from common import read,SCENES
ROOT=Path('/data/chenkejun/CVPR/results/offline_medoids_background070_20261108')
MEDOIDS=Path('/data/chenkejun/CVPR/results/offline_kmedoids_learned_20261108')

@lru_cache(maxsize=8)
def scene_result(scene):
 assert scene in SCENES
 return read(ROOT/scene/'classification.json')

def rows(scene,page=0,query='',status='all'):
 assert status in ('all','retain','exclude','review')
 result=scene_result(scene);q=query.strip().lower()
 translated={'wall':'墙','floor':'地板','ceiling':'天花板','rug':'地毯'}
 def matches(r):
  if status=='review' and r['decision']!='review':return False
  if status in ('retain','exclude') and r['operational_decision']!=status:return False
  terms=[r['group_id'],r['mask_id'],str(r['frame'])]+[c['name']+' '+translated.get(c['name'],'') for c in r['categories']]
  return not q or q in ' '.join(terms).lower()
 chosen=[r for r in result['records'] if matches(r)];pages=max(1,(len(chosen)+11)//12);page=max(0,min(page,pages-1))
 return dict(version=result['version'],summary=result['summary'],rows=chosen[page*12:(page+1)*12],total=len(chosen),page=page,pages=pages)

def detail(scene,mid):
 return next(r for r in scene_result(scene)['records'] if r['mask_id']==mid)

def image_bytes(scene,mid,kind):
 assert kind in ('context','masked','semantic','confidence')
 row=detail(scene,mid);p=(ROOT/scene/row['images'][kind]).resolve()
 assert p.is_relative_to((ROOT/scene/'previews').resolve()) and p.suffix=='.jpg'
 return p.read_bytes()

def metadata():
 summaries=read(ROOT/'summary.json');medoids={s['scene']:s for s in read(MEDOIDS/'summary.json')}
 return dict(version='medoids-v3-background070-v3-20261108',stage='background_semantic_screening',scenes={s['scene']:dict(**s,medoids_input=medoids[s['scene']]['candidates'],medoids_retained=medoids[s['scene']]['kept']) for s in summaries},total_input=sum(s['input'] for s in summaries),total_retained=sum(s['retained'] for s in summaries),total_excluded=sum(s['excluded'] for s in summaries))
