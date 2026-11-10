"""Read-only score presentation from frozen official offline measurements."""
from functools import lru_cache
from itertools import combinations
from common import *

RULES = {
 'boundary':'只排除人工裁剪边缘触边；原图四周边缘触边允许保留。所有原来源记录共同参与检查。',
 'noise':'面积 < 100；或长边 ≥ 80 且 3px 内核占比 ≤ 5%；或长边 ≥ 80、填充率 ≤ 10%、8px 内核占比 ≤ 5%；或长边 ≥ 200、填充率 ≤ 5%、12px 内核占比 ≤ 5%。任一条件成立即排除。',
 'grouping':'按 3D 支持点数从大到小选主种子；与主种子 3D IoU > 25% 的候选进入该组。辅助成员之间的分数用于检查，原分组不要求它们两两达标。',
 'semantic':'排除类语义像素占 mask 面积 ≥ 90% 时排除。',
 'covisible_iou_only':'固定主种子去重：共视 IoU ≥ 80% 时并入主组。仅原实际比较项参与决定。',
 'full_support_containment':'完整 3D 支持交集 / 小 mask 支持点数 > 70% 时排除较小 mask；按正式离线版保存的直接关系执行。',
 'skip_repair_B':'p ≥ 95% 且 r ≥ 95% 时不需修复，排除。p = 交集 / mask 支持点数；r = 交集 / 匹配实例可见点数。',
 'pr_filter':'p > 10% 且 r < 20% 时排除；窗户／柜子语义仅豁免此步。未匹配实例不触发该规则。',
 'retained':'下列分数按原筛选顺序汇总；最终决定和豁免均沿用保存结果。'
}
def pct(n,d):return f'{100*n/d:.2f}%' if d else '无有效分母'
def metric(label,value,rule='',detail=''):
 return dict(label=label,value=str(value),rule=rule,detail=detail)

def source(key):
 from appdata import scene_data
 return scene_data(key)['source']

def pair_rows(key,phase,gid):
 s=source(key);out=[]
 if phase=='grouping':
  group=s['groups'][gid];primary=group['primary']
  for a,b in combinations(group['members'],2):
   i,j=s['seeds'][a]['index'],s['seeds'][b]['index'];c,u=int(s['inter'][i,j]),int(s['union'][i,j])
   out.append(dict(a=a,b=b,value=c/u if u else None,numerator=c,denominator=u,passes=u>0 and 4*c>u,
                   role='主种子与成员 · 分组依据' if primary in (a,b) else '辅助成员之间 · 检查参考',rule='> 25%'))
 else:
  stage=next(i for i,x in enumerate(s['selection']['stages']) if x['stage']==phase)
  incoming=set(s['selection']['stages'][stage-1]['retained_group_ids'])
  for k,q in s['pairs'].items():
   if gid not in k or not set(k)<=incoming:continue
   a,b=k
   if phase=='covisible_iou_only':
    c,u=q['intersection'],q['union'];role='原去重实际比较' if k in s['visited'] else '仅已测量，原去重未比较'
    row=dict(a=a,b=b,value=c/u if u else None,numerator=c,denominator=u,passes=u>0 and 5*c>=4*u,role=role,rule='≥ 80%')
   else:
    na,nb=s['ms'][a]['support_count'],s['ms'][b]['support_count'];small,large=(a,b) if na<nb else (b,a)
    c,u=q['original_intersection'],min(na,nb)
    row=dict(a=small,b=large,value=c/u if u else None,numerator=c,denominator=u,passes=10*c>7*u and na!=nb,role='小 → 大 · 原直接比较' if na!=nb else '等支持点数',rule='> 70%')
   out.append(row)
  out.sort(key=lambda r:(not r['passes'],-(r['value'] or 0),r['a'],r['b']))
 return out

def pair_page(key,phase,gid,page=0):
 assert phase in ('grouping','covisible_iou_only','full_support_containment')
 rows=pair_rows(key,phase,gid);pages=max(1,(len(rows)+49)//50);page=max(0,min(page,pages-1))
 return dict(rows=rows[page*50:(page+1)*50],page=page,pages=pages,total=len(rows),rule=RULES[phase])

def row_scores(d,phase,row):
 if phase not in RULES:return []
 s=source(d['key']);gid=row.get('group_id');mid=row.get('mask_id',row.get('id'));ev=(row.get('current_decision') or {}).get('evidence');out=[]
 if phase=='boundary':
  r=s['raw'][mid];contacts=r.get('crop_contacts',[])
  out.append(metric('人工裁剪边缘',f'{len(contacts)} 处触边来源','任一人工边接触 > 0 即排除'))
  for c in contacts:out.append(metric('crop '+str(c['source_box']),', '.join(f'{k}: {v}px' for k,v in c['edge_pixels'].items()),'',c['crop_key']))
  out.append(metric('原图边缘','允许触边','此离线版本不因整帧四周触边而删除'))
 elif phase=='noise':
  r=s['raw'][mid];f=r['noise_features'];area=f['area'];core=f['core_pixels'];flags=r['noise_decision']['matched_rules']
  out=[metric('面积',f'{area:,} px','< 100'),metric('长边',f"{f['extent']} px",'细带/稀疏 ≥ 80；大稀疏 ≥ 200'),metric('填充率',pct(area,f['bbox_area']),'稀疏 ≤ 10%；大稀疏 ≤ 5%',f"{area} / {f['bbox_area']}")]
  out += [metric(f'{radius}px 内核占比',pct(core[str(radius)],area),'≤ 5%，须同时满足长边/填充条件',f'{core[str(radius)]} / {area}') for radius in [3,8,12]]
  out.append(metric('命中规则','、'.join(flags) or '无'))
 elif phase=='grouping':
  out=[metric('3D 支持点数',f"{s['seeds'][mid]['support_count']:,}",'按支持点数选主种子')]
 elif phase=='semantic':
  r=s['ms'][gid];out=[metric('排除类语义占比',pct(r['high_excluded_pixels'],r['area']),'≥ 90% 排除',f"{r['high_excluded_pixels']} / {r['area']} 像素")]
 elif phase in ('covisible_iou_only','full_support_containment'):
  rows=pair_rows(d['key'],phase,gid)
  if phase=='covisible_iou_only':rows=[r for r in rows if r['role']=='原去重实际比较']
  if isinstance(ev,dict) and ev.get('union'):
   out.append(metric('触发共视 IoU',pct(ev['intersection'],ev['union']),'≥ 80% 排除',f"与 {ev['primary_group_id']}；{ev['intersection']} / {ev['union']}"))
  elif isinstance(ev,list) and ev:
   out += [metric('直接包含比例',pct(v['intersection'],v['small_count']),'> 70% 排除小 mask',f"被 {v['larger_group_id']} 包含；{v['intersection']} / {v['small_count']}") for v in ev]
  else:
   # For a retained small mask, show only larger comparison partners; it may also contain smaller masks.
   relevant=rows if phase=='covisible_iou_only' else [r for r in rows if r['a']==gid and r['role']!='等支持点数']
   valid=[r for r in relevant if r['value'] is not None]
   if valid:
    best=max(valid,key=lambda r:r['value']);other=best['b'] if best['a']==gid else best['a']
    out.append(metric('最高实际共视 IoU' if phase=='covisible_iou_only' else '被较大 mask 包含的最高比例',pct(best['numerator'],best['denominator']),'≥ 80% 合并，保留支持较大的主种子' if phase=='covisible_iou_only' else '> 70%',f"与 {other}；{best['numerator']} / {best['denominator']}"))
   else:out.append(metric('可比较项','无有效比较项','不以 0 分替代缺失分母/比较对象'))
 elif phase in ('skip_repair_B','pr_filter'):
  c=s['coverage'][gid];common=c['intersection'];total=c['support_count'];visible=c['instance_visible_count']
  out=[metric('p · mask 被实例覆盖',pct(common,total),'≥ 95%' if phase=='skip_repair_B' else '> 10%',f'{common} / {total}'),metric('r · 实例被 mask 覆盖',pct(common,visible),'≥ 95%' if phase=='skip_repair_B' else '< 20%',f'{common} / {visible}'),metric('匹配地图实例',c['instance_id'] or '未匹配','两项条件同时满足才触发')]
  if phase=='pr_filter':out.append(metric('窗户／柜子语义豁免','是' if gid in s['selection'].get('semantic_pr_exempt_group_ids',[]) else '否','只豁免 p/r 筛选'))
 elif phase=='retained':
  for dec in s['records'][gid]['stage_decisions']:
   st=dec['stage']
   if st in RULES:out += [dict(x,label=NAMES.get(st,st)+' · '+x['label']) for x in row_scores(d,st,dict(row,current_decision=dec))]
 return out

def decorate(d,phase,row):
 if row.get('role')=='auxiliary':
  if phase=='grouping':
   scores=row_scores(d,phase,row);s=source(d['key']);group=s['groups'][row['instance_group']];primary=group['primary'];mid=row['id'];i,j=s['seeds'][primary]['index'],s['seeds'][mid]['index'];c,u=int(s['inter'][i,j]),int(s['union'][i,j])
   scores.append(metric('与主种子 '+primary+' 的 3D IoU',pct(c,u),'> 25%',f'{c} / {u}'));row['scores']=scores
  return row
 row['scores']=row_scores(d,phase,row)
 if phase in ('covisible_iou_only','full_support_containment'):row['score_pairs']=dict(phase=phase,id=row['group_id'])
 return row
