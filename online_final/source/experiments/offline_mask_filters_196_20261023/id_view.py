from common import *
ROOT=Path('/data/chenkejun/CVPR/results/offline_contain040_pipeline_20261108/id_view')
def metadata():return dict(scenes=read(ROOT/'summary.json'),input=153,inherit=57,new=96,conflict_new=31,no_match_new=65)
def rows(scene,page=0,query='',status='all'):
 assert scene in SCENES and status in ('all','inherit','new','conflict_new','no_match_new')
 rs=read(ROOT/'data.json')[scene];q=query.strip().lower();groups={}
 for r in rs:
  if status!='all' and r['id_action']!=status and r['id_category']!=status:continue
  if q and q not in ' '.join([r['mask_id'],r['group_id'],r['relation_group'],str(r['preview_id'])]).lower():continue
  groups.setdefault(r['relation_group'],[]).append(r)
 gs=[dict(id=k,members=v) for k,v in sorted(groups.items())];pages=max(1,(len(gs)+3)//4);page=max(0,min(page,pages-1))
 return dict(groups=gs[page*4:(page+1)*4],total=len(gs),total_masks=sum(len(g['members']) for g in gs),page=page,pages=pages,summary=metadata()['scenes'][scene])
