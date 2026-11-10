"""Frozen offline auto-v3-swap inputs; no selection or reconstruction is executed."""
from common import *
from functools import lru_cache
from itertools import combinations
from PIL import Image
import io

@lru_cache(maxsize=8)
def scene_data(scene):
 assert scene in SCENES
 root=BASE/scene;v1=root/'v1';v2=root/'v2';fixture=MAIN/'experiments/auto_v3_swap/fixtures'/scene
 paths=[v1/'grouping.json',v1/'semantic_exclusions.json',v1/'seed_manifest.json',v2/'selection.json',v2/'measurements.json',v2/'seed_manifest.json']
 locks={str(p):sha(p) for p in paths}
 for name,p in [('v1_manifest.json',v1/'seed_manifest.json'),('selection.json',v2/'selection.json'),('measurements.json',v2/'measurements.json'),('seed_manifest.json',v2/'seed_manifest.json')]:
  assert read(p)==read(fixture/name),(scene,'frozen fixture mismatch',name)
 cfg=read(MAIN/'experiments/auto_v3_swap/config.json');sel=read(v2/'selection.json');assert len(sel['retained_group_ids'])==cfg['counts'][scene]
 grouping=read(v1/'grouping.json');semex=read(v1/'semantic_exclusions.json');measure=read(v2/'measurements.json')
 raw={};excluded={};seedpaths={};indexes={s['index']:s['id'] for s in grouping['seeds']}
 for key,phase in [('seeds',None),('excluded_crop_masks','boundary'),('excluded_noise_masks','noise')]:
  for s in grouping[key]:
   assert s['id'] not in raw;raw[s['id']]=dict(s);seedpaths[s['id']]=v1/s['mask'];excluded[s['id']]=phase
 for r in semex['excluded']:
  if not r.get('entire_mask_excluded'):continue
  assert r['id'] not in raw
  raw[r['id']]=dict(r,aliases=r['removed_aliases'],representative=r['removed_aliases'][0]);excluded[r['id']]='source'
 # The earliest saved unique candidates include every excluded source mask.
 assert len(raw)==grouping['stats']['source_candidates'],(scene,len(raw),grouping['stats'])
 groups={g['id']:dict(primary=indexes[g['primary']],members=[indexes[i] for i in g['members']]) for g in grouping['groups']}
 membership={mid:gid for gid,g in groups.items() for mid in g['members']}
 ms={s['group_id']:s for s in measure['seeds']};records={r['group_id']:r for r in sel['records']};pairs={tuple(sorted((p['a'],p['b']))):p for p in measure['pairs']}
 # Retain exact original fixed-primary comparison scope for the score table.
 remaining=sorted(sel['stages'][0]['retained_group_ids'],key=lambda a:(-ms[a]['support_count'],ms[a]['frame'],ms[a]['source_model'],ms[a]['mask_digest']))
 visited=set();reconstructed=[]
 while remaining:
  primary=remaining[0];direct=[]
  for other in remaining[1:]:
   k=tuple(sorted((primary,other)));visited.add(k);p=pairs[k]
   if p['union']>0 and 5*p['intersection']>=4*p['union']:direct.append(other)
  reconstructed.append([primary]+direct);removed=set([primary]+direct);remaining=[x for x in remaining if x not in removed]
 assert reconstructed==sel['dedup_groups'],scene
 inter=np.array(grouping['intersection'],dtype=np.int64);counts=np.array([s['support_count'] for s in grouping['seeds']],dtype=np.int64);union=counts[:,None]+counts[None,:]-inter
 with np.load(v1/'pair_metrics.npz') as z:assert np.array_equal(inter,z['intersection'])
 sem=read(v2/'semantic/classification.json');semantic={r['group_id']:r for r in sem['records']}
 rawcfg=read(f'/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/{scene}/configs/raw.json')['source']
 data=dict(scene=scene,key=scene,raw=raw,excluded=excluded,seedpaths=seedpaths,groups=groups,membership=membership,selection=sel,records=records,semantic=semantic,stats=grouping['stats'],rawcfg=rawcfg,locks=locks,root=root)
 data['source']=dict(path=root,groups=groups,seeds={s['id']:s for s in grouping['seeds']},raw=raw,inter=inter,union=union,ms=ms,pairs=pairs,visited=visited,coverage={r['group_id']:r for r in measure['coverage']},selection=sel,records=records,rules=grouping['rule'])
 return data

def mask_array(d,mid):
 r=d['raw'][mid]
 if mid in d['seedpaths']:
  p=d['seedpaths'][mid];assert sha(p)==r['mask_sha256'],(d['scene'],mid)
  a=np.asarray(Image.open(p))>0
 else:
  alias=r['representative'];labels=np.asarray(Image.open(alias['source_path']));local=labels==alias['label']
  if local.shape==(680,1200):a=local
  else:
   x0,y0,x1,y1=alias['source_box'];assert local.shape==(y1-y0,x1-x0);a=np.zeros((680,1200),bool);a[y0:y1,x0:x1]=local
 assert int(a.sum())==r['area'],(d['scene'],mid,'area')
 # Some official records were manually source-corrected; the locked mask SHA is authoritative.
 if mid not in d['seedpaths']:assert hashlib.sha256(r['frame'].to_bytes(4,'little')+np.packbits(a.ravel()).tobytes()).hexdigest()==r['digest']
 return a

def crop_contacts(d,mid):
 r=d['raw'][mid]
 if 'crop_contacts' in r:return r['crop_contacts']
 a=mask_array(d,mid);contacts=[];seen=set()
 for alias in r['aliases']:
  model=alias['model'];box=tuple(alias['source_box'])
  if model==0 or (model,box) in seen:continue
  seen.add((model,box));x0,y0,x1,y1=box;values={}
  if x0>0:values['左']=int(a[y0:y1,x0].sum())
  if y0>0:values['上']=int(a[y0,x0:x1].sum())
  if x1<1200:values['右']=int(a[y0:y1,x1-1].sum())
  if y1<680:values['下']=int(a[y1-1,x0:x1].sum())
  values={k:v for k,v in values.items() if v}
  if values:contacts.append(dict(source_box=box,edge_pixels=values,crop_key=alias['crop_key']))
 r['crop_contacts']=contacts
 assert bool(contacts)==(d['excluded'][mid]=='boundary'),(d['scene'],mid,'crop evidence differs')
 return contacts

@lru_cache(maxsize=48)
def rgb(path):return np.asarray(Image.open(path).convert('RGB'))

@lru_cache(maxsize=180)
def image_bytes(scene,mid,full=False,binary=False):
 d=scene_data(scene);r=d['raw'][mid];a=mask_array(d,mid)
 if binary:im=Image.fromarray(a.astype('uint8')*255)
 else:
  source=d['rawcfg'];color=rgb(str(Path(source['scene_root'])/source['rgb_pattern'].format(frame=r['frame'])))
  view=(color*.65).astype(np.uint8);view[a]=(color[a]*.55+np.array([25,225,170])*.45).astype(np.uint8);im=Image.fromarray(view)
  if not full:
   y,x=np.nonzero(a);pad=25;im=im.crop((max(0,x.min()-pad),max(0,y.min()-pad),min(a.shape[1],x.max()+pad+1),min(a.shape[0],y.max()+pad+1)))
  im.thumbnail((1000,680) if full else (390,260))
 b=io.BytesIO();im.save(b,format='PNG' if binary else 'JPEG',**({} if binary else {'quality':86}));return b.getvalue()

def all_stage_rows(d,phase):
 if phase in ('source','boundary','noise'):
  skip={'source':set(),'boundary':{'source'},'noise':{'source','boundary'}}[phase]
  return [dict(mid=mid,decision='exclude' if reason==phase else 'retain',gid=d['membership'].get(mid)) for mid,reason in d['excluded'].items() if reason not in skip]
 if phase=='grouping':return [dict(mid=g['primary'],gid=gid,decision='retain') for gid,g in d['groups'].items()]
 if phase=='retained':ids=d['selection']['retained_group_ids']
 else:
  index=next(i for i,s in enumerate(d['selection']['stages']) if s['stage']==phase)
  ids=list(d['groups']) if index==0 else d['selection']['stages'][index-1]['retained_group_ids']
 out=[]
 for gid in ids:
  record=d['records'][gid];dec=next((r for r in record['stage_decisions'] if r['stage']==phase),None)
  decision=dec['decision'] if dec else record['decision']
  if phase=='pr_filter' and gid in d['selection'].get('manual_restored_group_ids',[]):decision='retain'
  out.append(dict(mid=record['mask_id'],gid=gid,decision=decision,current_decision=dec))
 return out

def card(d,row,phase):
 from score_logic import row_scores,metric,pct
 mid=row['mid'];r=d['raw'][mid];gid=row.get('gid');src=dict(r,id=mid,mask_id=mid,group_id=gid,current_decision=row.get('current_decision'))
 if phase=='boundary':crop_contacts(d,mid)
 if phase=='source':
  scores=[metric('来源资格','被 SAM 替换' if row['decision']=='exclude' else '合格','窗户／柜子只使用符合原路由的 SAM 候选；无统一数值阈值')]
  ev=r.get('replacement_evidence',{})
  for edge in ev.get('primary_edges',[]) if isinstance(ev,dict) else []:scores.append(metric('与 '+edge['mask_id']+' 的 3D IoU',pct(edge['intersection'],edge['union']),'> 25%',f"{edge['intersection']} / {edge['union']}"))
 else:scores=row_scores(d,phase,src)
 record=d['records'].get(gid,{});cls=d['semantic'].get(gid,{}).get('top_class','') if gid and d['groups'][gid]['primary']==mid else ''
 group_members=[]
 if phase=='grouping' and gid:
  source=d['source'];primary=source['seeds'][mid]['index']
  for member in d['groups'][gid]['members']:
   entry=d['raw'][member];i=source['seeds'][member]['index'];intersection=int(source['inter'][primary,i]);union=int(source['union'][primary,i])
   group_members.append(dict(id=member,frame=entry['frame'],support=entry['support_count'],primary=(member==mid),iou=(intersection/union if union and member!=mid else None),intersection=intersection,union=union))
 return dict(group_members=group_members,id=mid,gid=gid,frame=r['frame'],area=r['area'],support=r['support_count'],model=r.get('representative',{}).get('model'),decision=row['decision'],scores=scores,top_class=cls,
             final=record.get('decision') if gid and d['groups'][gid]['primary']==mid else None,terminal=record.get('terminal_stage') if gid else d['excluded'][mid],member_count=len(d['groups'][gid]['members']) if gid else 0,
             note='原有历史人工恢复：G017 / G025' if record.get('terminal_stage')=='manual_room2_override' else '',aliases=len(r.get('aliases',[])))

def rows_page(scene,phase,page=0,query='',status='all',allowed_groups=None):
 from score_logic import RULES
 d=scene_data(scene);rows=all_stage_rows(d,phase);total=len(rows);kept=sum(r['decision']=='retain' for r in rows)
 def searchable(r):
  group_members=d['groups'][r['gid']]['members'] if phase=='grouping' else [r['mid']]
  return ' '.join([str(r['gid']),str(d['semantic'].get(r['gid'],{}).get('top_class',''))]+[f"{mid} {d['raw'][mid]['frame']}" for mid in group_members]).lower()
 chosen=[r for r in rows if (allowed_groups is None or r['gid'] in allowed_groups) and (status=='all' or (r['decision']=='retain')==(status=='retain')) and (not query or query.lower() in searchable(r))]
 pages=max(1,(len(chosen)+11)//12);page=max(0,min(page,pages-1))
 return dict(rows=[card(d,r,phase) for r in chosen[page*12:(page+1)*12]],input=total,retained=kept,excluded=total-kept,total=len(chosen),page=page,pages=pages,rule=RULES.get(phase,'窗户／柜子 SAM-only 来源资格；重复来源不重复计算 mask。'))

def detail(scene,mid,phase):
 from score_logic import pair_rows
 d=scene_data(scene);r=d['raw'][mid];gid=d['membership'].get(mid);members=d['groups'][gid]['members'] if gid else []
 history=[]
 for st in NAMES:
  found=next((x for x in all_stage_rows(d,st) if x['mid']==mid),None)
  if found:history.append(dict(stage=st,**card(d,found,st)))
 if gid:
  group=d['groups'][gid];scores=pair_rows(scene,'grouping',gid)
 else:group=None;scores=[]
 return dict(id=mid,gid=gid,history=history,aliases=r.get('aliases',[]),group=group,members=[dict(id=m,frame=d['raw'][m]['frame'],support=d['raw'][m]['support_count']) for m in members],pair_preview=scores[:6],pair_total=len(scores),raw_decision=d['records'].get(gid) if gid and group['primary']==mid else None)

def metadata():
 cfg=read(MAIN/'experiments/auto_v3_swap/config.json');current=read(MAIN/'instance_repair/configs/current_auto_pipeline.json')
 assert cfg['retained_total']==196 and current['pipeline']=='partial-e1-swap-20261021'
 return dict(version='auto-v3-swap / E1 partial',result_date=cfg['date'],review_date='2026-10-23',pipeline=current['pipeline'],source=str(BASE),names=NAMES,scenes=[dict(scene=s,counts={p:len(all_stage_rows(scene_data(s),p)) for p in NAMES},retained=cfg['counts'][s],raw_count=len(scene_data(s)['raw']),v1=len(scene_data(s)['groups'])) for s in SCENES],total=196)
