from common import *
ROOT=Path('/data/chenkejun/CVPR/results/offline_fixedH_manual_room0_20261108')
def metadata():
 s=read(ROOT/'summary.json');return dict(scenes=s,**{k:sum(v[k] for v in s.values()) for k in ('input','retained','excluded','windows','incomplete')})
def rows(scene,page=0,query='',status='all'):
 assert scene in SCENES and status in ('all','retain','exclude')
 gs=[];q=query.lower().strip()
 for g in read(ROOT/'data.json')[scene]:
  hits=[r for r in g['members'] if (status=='all' or r['overlap_decision']==status) and (not q or q in (g['id']+' '+r['mask_id']+' '+r['group_id']).lower())]
  if hits:gs.append(g)
 pages=max(1,(len(gs)+3)//4);page=max(0,min(page,pages-1));return dict(groups=gs[page*4:(page+1)*4],summary=metadata()['scenes'][scene],total=len(gs),page=page,pages=pages)
