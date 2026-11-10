from common import *
ROOT=Path('/data/chenkejun/CVPR/results/offline_contain040_pipeline_20261108/fragment_view')
def metadata():
 s=read(ROOT/'summary.json');return dict(scenes=s,input=277,retained=153,excluded=124)
def rows(scene,page=0,query='',status='all'):
 assert scene in SCENES and status in ('all','retain','exclude')
 d=read(ROOT/'data.json')[scene];groups=[];q=query.lower().strip()
 for g in d['groups']:
  members=[r for r in g['members'] if (status=='all' or r['fragment_decision']==status) and (not q or q in ' '.join([g['id'],r['mask_id'],r['group_id'],str(r['frame'])]).lower())]
  if members:groups.append(dict(id=g['id'],members=members,original_count=len(g['members'])))
 pages=max(1,(len(groups)+3)//4);page=max(0,min(page,pages-1))
 return dict(groups=groups[page*4:(page+1)*4],total=len(groups),total_masks=sum(len(g['members']) for g in groups),page=page,pages=pages,summary=metadata()['scenes'][scene])
