"""Audit mask retention by binary digest, and render original pixels for review."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,tarfile
import numpy as np
from PIL import Image,ImageFilter

R=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
E=R/'window_only_mask_ablation';W=E/'review';W.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def selection(root):
 m=read(root/'all_edge_20261021/v2/seed_manifest.json')
 return {g['primary']['digest']:g for g in m['groups']}

def statuses(root):
 result=defaultdict(list)
 for key,c in read(root/'semantic_route/selection.json')['crops'].items():
  for n in c['nodes']:
   row=c['models'][n['model']]['labels'][str(n['label'])]
   result[n['candidate_id']].append(dict(crop=key,status=row['status'],reason=row['reason'],model=n['model'],semantic_hit=n['semantic_hit'],eligible=n['eligible'],support=n['support']))
 return result

def terminal(root,digest):
 grouping=read(root/'grouping_r2/grouping.json');s=next((s for s in grouping['seeds'] if s['digest']==digest),None)
 if s is None:return '未通过普通来源筛选或来源去重'
 edge=read(root/'all_edge_20261021/boundary_audit.json')
 if any(s['digest']==digest for s in edge['edge_excluded']):return '边界筛选移除'
 if any(s['digest']==digest for s in edge['noise_excluded']):return '形态噪声筛选移除'
 mid=read(root/'all_edge_20261021/v1/seed_manifest.json')
 g=next((g for g in mid['groups'] if g['primary']['digest']==digest),None)
 if g is None:return '分组中作为辅助 Mask，未成为主种子'
 q=next(x for x in read(root/'all_edge_20261021/v2/selection.json')['records'] if x['group_id']==g['id'])
 return q['terminal_stage']

def image_pair(branch,digest,p,old,new,before,after):
 source_root=old if before else new
 maskp=source_root/'all_edge_20261021/v1'/p['mask']
 arr=np.asarray(Image.open(maskp))>0
 assert sha(maskp)==p['mask_sha256'] and int(arr.sum())==p['area']
 frame=p['frame'];rgb=np.asarray(Image.open(R/f'400/{branch}/mapping/inputs/scene/results/frame{frame:06d}.jpg'))
 yy,xx=np.nonzero(arr);pad=max(60,int(max(np.ptp(xx)+1,np.ptp(yy)+1)*.35))
 box=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(rgb.shape[1],int(xx.max())+pad+1),min(rgb.shape[0],int(yy.max())+pad+1))
 # Original RGB and exact binary mask; color overlay only for visualization.
 maskim=Image.fromarray(arr.astype(np.uint8)*255)
 edge=np.asarray(maskim.filter(ImageFilter.MaxFilter(3)))!=np.asarray(maskim.filter(ImageFilter.MinFilter(3)))
 overlay=rgb.copy();overlay[arr]=(rgb[arr]*.55+np.array([255,64,95])*.45).astype(np.uint8);overlay[edge]=[255,55,85]
 folder=W/'images';folder.mkdir(exist_ok=True)
 stem=f'{branch}_{digest[:16]}'
 for name,img in [('mask',overlay),('rgb',rgb)]:
  im=Image.fromarray(img).crop(box);im.thumbnail((680,420));im.save(folder/f'{stem}_{name}.jpg',quality=93)
 # Keep exact full-resolution binary mask as a downloadable artifact.
 (W/'masks').mkdir(exist_ok=True);(W/'masks'/f'{stem}.png').write_bytes(maskp.read_bytes())
 return dict(overlay=f'images/{stem}_mask.jpg',rgb=f'images/{stem}_rgb.jpg',binary=f'masks/{stem}.png',box=box)

assert read(E/'complete.json')['status']=='PASS'
report=dict(scope='第400帧固定修复前检查点；只取消柜子的 SAM 直通及相应 CropFormer 排除',
 unchanged='窗户直通、普通筛选阈值、边界/噪声/包含去重、SegFormer 窗户与柜子的 P/R 豁免均保留。没有追踪、修复、提交或重新评价 AP。',branches={},source_sha256={})
for branch in ['A','B']:
 old=R/'400'/branch/'candidates';new=E/branch/'window_only'
 a,b=selection(old),selection(new);da,db=set(a),set(b);sta,stb=statuses(old),statuses(new)
 assert read(E/branch/'control/complete.json')['control_reproduces_original_final_masks_and_stage_decisions']
 original_tracks={x['mask_id']:x['track_id'] for x in read(R/f'400/{branch}/tracking/seed_id_map.json')['objects']}
 focus_ids=['M0473','M0956','M1023'] if branch=='A' else []
 rows=[]
 for digest in sorted(da|db,key=lambda d:(d not in da^db, next((v['primary']['id'] in focus_ids for v in [a.get(d),b.get(d)] if v),False)==False,(a.get(d) or b[d])['primary']['frame'])):
  g=a.get(digest) or b[digest];p=g['primary'];before=digest in a;after=digest in b
  row=dict(digest=digest,frame=p['frame'],before=before,after=after,change='removed' if before and not after else 'added' if after and not before else 'unchanged',
   before_mask=a[digest]['primary']['id'] if before else None,after_mask=b[digest]['primary']['id'] if after else None,
   original_track=original_tracks.get(a[digest]['primary']['id']) if before else None,
   before_model=a[digest]['primary']['representative']['model'] if before else None,after_model=b[digest]['primary']['representative']['model'] if after else None,
   original_reason=p['representative']['selection_reason'],focus=p['id'] in focus_ids,area=p['area'],support=p['support_count'],
   before_group=a[digest]['id'] if before else None,after_group=b[digest]['id'] if after else None,
   before_aliases=a[digest]['primary']['aliases'] if before else [],after_aliases=b[digest]['primary']['aliases'] if after else [],
   after_terminal='retained' if after else terminal(new,digest))
  if before and after:
   assert a[digest]['primary']['mask_sha256']==b[digest]['primary']['mask_sha256']
   assert a[digest]['primary']['support_sha256']==b[digest]['primary']['support_sha256']
  cids={x['candidate_id'] for x in row['before_aliases']+row['after_aliases']}
  row['route_before']={cid:sta[cid] for cid in cids};row['route_after']={cid:stb[cid] for cid in cids}
  row['images']=image_pair(branch,digest,p,old,new,before,after);rows.append(row)
 oldsem=read(old/'semantic_route/semantic.json');newsem=read(new/'semantic_route/semantic.json')
 oldwindow={k:v for k,v in oldsem['candidates'].items() if v['hit'] and v['target_family']=='window'}
 assert all(newsem['candidates'][k]==v for k,v in oldwindow.items())
 disabled=read(new/'experiment.json')['disabled_cabinet_hits']
 disabled_details=[]
 for cid in disabled:
  original=oldsem['candidates'][cid]
  disabled_details.append(dict(candidate_id=cid,area=original['area'],before=sta[cid],after=stb[cid]))
 stages=[]
 for name in ['semantic_route','grouping_r2','all_edge_20261021']:
  file='audit.json' if name!='all_edge_20261021' else 'complete.json'
  stages.append(dict(stage=name,before=read(old/name/file),after=read(new/name/file)))
 data=dict(before_count=len(a),after_count=len(b),same=len(da&db),removed=len(da-db),added=len(db-da),
  sources_before=dict(Counter(g['primary']['representative']['model'] for g in a.values())),sources_after=dict(Counter(g['primary']['representative']['model'] for g in b.values())),
  disabled_cabinet_direct_candidates=len(disabled),window_direct_candidates=len(oldwindow),window_decisions_exactly_unchanged=True,
  rows=rows,disabled_details=disabled_details,stages=stages,
  final_stages_before=read(old/'all_edge_20261021/v2/selection.json')['stages'],final_stages_after=read(new/'all_edge_20261021/v2/selection.json')['stages'],
  run=read(new/'complete.json'),control=read(E/branch/'control/complete.json'))
 report['branches'][branch]=data
 for p in [old/'all_edge_20261021/v2/seed_manifest.json',new/'all_edge_20261021/v2/seed_manifest.json',new/'semantic_route/semantic.json',new/'complete.json']:
  report['source_sha256'][str(p)]=sha(p)
 print(branch,json.dumps({k:v for k,v in data.items() if k not in ['rows','stages','disabled_details','final_stages_before','final_stages_after','run','control']},ensure_ascii=False))
 print('CHANGED_AND_FOCUS',json.dumps([{k:v for k,v in x.items() if k not in ['images','before_aliases','after_aliases']} for x in rows if x['change']!='unchanged' or x['focus']],ensure_ascii=False))
save(W/'comparison.json',report)
with tarfile.open(E/'review.tar.gz','w:gz') as tar:
 for p in W.rglob('*'):
  if p.is_file():tar.add(p,arcname=p.relative_to(W))
print('REVIEW',str(E/'review.tar.gz'),sha(E/'review.tar.gz'))
