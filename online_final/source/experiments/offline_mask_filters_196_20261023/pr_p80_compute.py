"""Replay only the last mask filter on immutable formal offline inputs."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/offline_mask_filters_196_20261023')
from appdata import scene_data,all_stage_rows,card
from common import SCENES,BASE,NAMES,sha
OUT=Path('/data/chenkejun/CVPR/results/offline_pr_p80_20261023')
OUT.mkdir(parents=True,exist_ok=True)
result=dict(version='auto-v3-swap / E1 partial',baseline=196,source=str(BASE),old_rule='p > 10% 且 r < 20%',new_rule='10% < p < 80% 且 r < 20%',scenes=[],names=NAMES)
locks={}
for scene in SCENES:
 d=scene_data(scene);sel=d['selection'];src=d['source'];locks.update(d['locks'])
 incoming=set(sel['stages'][3]['retained_group_ids']);old=set(sel['retained_group_ids'])
 exempt=set(sel.get('semantic_pr_exempt_group_ids',[]));manual=set(sel.get('manual_restored_group_ids',[]))
 assert exempt|manual<=incoming
 before=set();after=set();predicates={}
 for gid in incoming:
  c=src['coverage'][gid];i,n,v=(c[k] for k in ['intersection','support_count','instance_visible_count'])
  matched=c['instance_id']>0 and n>0 and v>0
  oldhit=matched and 10*i>n and 5*i<v
  newhit=oldhit and 5*i<4*n
  predicates[gid]=(oldhit,newhit)
  if not oldhit or gid in exempt|manual:before.add(gid)
  if not newhit or gid in exempt|manual:after.add(gid)
 assert before==old,(scene,'baseline replay differs',before^old)
 assert old<=after<=incoming
 added=after-old
 assert all(src['coverage'][g]['intersection']*5>=src['coverage'][g]['support_count']*4 for g in added)
 rows=[]
 for gid,g in d['groups'].items():
  mid=g['primary'];r=d['raw'][mid];rec=d['records'][gid];c=src['coverage'].get(gid);i=c['intersection'] if c else 0;n=c['support_count'] if c else 0;v=c['instance_visible_count'] if c else 0
  history=[]
  for st in ['semantic','covisible_iou_only','full_support_containment','skip_repair_B']:
   item=next((x for x in all_stage_rows(d,st) if x['gid']==gid),None)
   if item:history.append(dict(stage=st,**card(d,item,st)))
  rows.append(dict(gid=gid,id=mid,frame=r['frame'],support=r['support_count'],area=r['area'],class_name=d['semantic'].get(gid,{}).get('top_class',''),old=gid in old,new=gid in after,added=gid in added,entered_pr=gid in incoming,p=i/n if n else None,r=i/v if v else None,intersection=i,mask_points=n,visible_points=v,instance=c['instance_id'] if c else None,exempt=gid in exempt,manual=gid in manual,old_hit=predicates.get(gid,(None,None))[0],new_hit=predicates.get(gid,(None,None))[1],terminal=rec['terminal_stage'],history=history))
 summary=dict(scene=scene,old=len(old),new=len(after),added=len(added),removed=len(old-after),incoming=len(incoming),added_ids=sorted(added),old_ids=sorted(old),new_ids=sorted(after),rows=rows)
 result['scenes'].append(summary)
 (OUT/(scene+'_selection.json')).write_text(json.dumps(dict(source=str(BASE/scene),stage='pr_filter',old_rule=result['old_rule'],new_rule=result['new_rule'],input_group_ids=sorted(incoming),retained_group_ids=sorted(after),excluded_group_ids=sorted(incoming-after),added_group_ids=sorted(added),removed_group_ids=sorted(old-after),semantic_exempt_group_ids=sorted(exempt),historical_restored_group_ids=sorted(manual)),ensure_ascii=False,indent=2))
 print(scene,len(old),'->',len(after),'added',sorted(added),flush=True)
assert sum(s['old'] for s in result['scenes'])==196
assert all(sha(Path(p))==h for p,h in locks.items())
result.update(total=sum(s['new'] for s in result['scenes']),added=sum(s['added'] for s in result['scenes']),removed=0)
(OUT/'data.json').write_text(json.dumps(result,ensure_ascii=False))
(OUT/'audit.json').write_text(json.dumps(dict(status='PASS',baseline_reproduced=196,total=result['total'],added=result['added'],removed=0,source_locks=locks,earlier_stages_reused=True,tracking_run=False,repair_run=False,strict_p_upper_bound=True),ensure_ascii=False,indent=2))
print('TOTAL',result['total'],'ADDED',result['added'])
