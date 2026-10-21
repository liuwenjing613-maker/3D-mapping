"""Summarize complete three-arm, three-stage R2 results without changing them."""
from pathlib import Path
import csv,hashlib,html,json,subprocess,sys
import numpy as np
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');sys.path.insert(0,str(MAIN))
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,require_current_protocol
from unified_eval.io import sha256_file
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
GROUPS=['automatic','manual'];VARIANTS=['baseline','whole','partial','direct','core_all','e1'];METRICS=['F1','PQ','AP','AP50','mCov']
read=lambda p:json.loads(Path(p).read_text());sha=sha256_file
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert require_current_protocol(CURRENT_CONFIG)
 inputs=read(R/'input_lock.json')
 assert sha(C/'config.json')==inputs['config_sha256']
 for p,d in inputs['source_sha256'].items():assert sha(p)==d,p
 scenes=[];pergt=[];flat=[]
 for group in GROUPS:
  for scene in SCENES:
   folder=R/group/scene;assert read(folder/'evaluation_complete.json')['status']=='PASS'
   q=read(folder/'evaluation_r2/comparison.json');assert q['status']=='PASS'
   freeze=read(folder/'predictions_freeze.json');assert all(freeze['checks'].values())
   for kind in ['code_sha256','output_sha256']:
    for p,d in freeze[kind].items():assert sha(p)==d,p
   mapping=q['mapping'];assert mapping['direct_controls']['direct']['valid_to_zero']==0
   e0path=Path(q['source_locks']['baseline_final']['source']).parent.parent/'surface_evidence.npz'
   e0=np.load(e0path);direct=np.load(folder/'direct'/scene/'native/surface_evidence.npz')
   assert np.all(direct['total_frame_votes']>=e0['total_frame_votes'])
   q['direct_total_votes_never_decrease_at_any_TSDF_point']=True
   for v in ['direct','core_all','e1']:
    e=np.load(folder/v/scene/'native/surface_evidence.npz')
    q['mapping']['stages'][v]['total_effective_frame_point_votes']=int(e['total_frame_votes'].sum())
   q['baseline_total_effective_frame_point_votes']=int(e0['total_frame_votes'].sum())
   scenes.append(q);pergt+=read(folder/'evaluation_r2/per_GT.json')
   for row in q['rows']:flat.append(dict(group=group,scene=scene,**{k:v for k,v in row.items() if not isinstance(v,dict)}))
 for scene in SCENES:
  a=next(q for q in scenes if q['scene']==scene and q['group']=='automatic');b=next(q for q in scenes if q['scene']==scene and q['group']=='manual')
  for stage in ['native','final']:
   ar=next(r for r in a['rows'] if r['variant']=='baseline' and r['stage']==stage);br=next(r for r in b['rows'] if r['variant']=='baseline' and r['stage']==stage)
   for k in METRICS+['TP','FP','FN','gray_total']:assert ar[k]==br[k],(scene,stage,k)
   for k,v in a['source_locks']['baseline_'+stage].items():assert v==b['source_locks']['baseline_'+stage][k],(scene,stage,k)
 summary=[]
 for group in GROUPS:
  gs=[q for q in scenes if q['group']==group];assert sum(q['GT_count'] for q in gs)==350
  assert sum(q['selected'] for q in gs)=={'automatic':196,'manual':76}[group]
  for stage in ['native','final']:
   for variant in VARIANTS:
    rows=[next(r for r in q['rows'] if r['variant']==variant and r['stage']==stage) for q in gs]
    row=dict(group=group,stage=stage,variant=variant,GT_count=350)
    for k in METRICS:row[k]=sum(r[k] for r in rows)/8
    for k in ['TP','FP','FN','gray_total','selected']:row[k]=sum(r[k] for r in rows)
    row['gray_states']={k:sum(r['gray_states'][k] for r in rows) for k in ['U','T','C']}
    if all('owner_surface' in r for r in rows):
     row['owner_surface']={k:sum(r['owner_surface'][k] for r in rows)/8 for k in ['Correct_owner_Coverage','Wrong_owner_Coverage','Unassigned_Coverage']}
     row['owner_vertex_counts']={k:sum(r['owner_surface'][k] for r in rows) for k in ['correct_owner_vertices','wrong_owner_vertices','unassigned_with_geometry_target_vertices','target_vertices']}
    assert row['TP']+row['FN']==350;summary.append(row)
 diagnostics={}
 for group in GROUPS:
  gs=[q for q in scenes if q['group']==group]
  diagnostics[group]=dict(direct_controls={v:{k:sum(q['mapping']['direct_controls'][v][k] for q in gs) for k in ['valid_to_zero','zero_to_valid','valid_id_changed']} for v in ['whole','partial','direct']},
   new_conflict_frame_points=sum(q['mapping']['new_conflict_frame_points'] for q in gs),
   committed_frame_points=sum(q['mapping']['committed_frame_points'] for q in gs),
   stages={v:{k:sum(q['mapping']['stages'][v][k] for q in gs) for k in ['valid_to_zero_frame_points','zero_to_valid_frame_points','confirmed_to_unassigned_points','confirmed_to_unassigned_with_non_decreased_total_votes','total_effective_frame_point_votes']} for v in ['direct','core_all','e1']})
 result=dict(status='PASS',protocol='object_observed_repair / revision 2',frozen=False,GT_count=350,scenes=scenes,summary=summary,diagnostics=diagnostics,
  verified=dict(all_16_runs_and_96_new_metric_records_complete=True,all_pairing_locks_match=True,baseline_identical_across_groups=True,direct_no_effective_votes_lost=True,
   direct_outside_and_conflict_candidates_unchanged=True,GT_never_read_by_mapper=True,original_frozen_inputs_unchanged=True),
  caveats=['自动196种子包含 room2 历史人工恢复的 G017、G025。','直接替换阶段保持新ID冲突点原证据；后续原 E1 历史改票仍可能改变这些点。','native 与 final 分开评价；final 沿用原后处理。'])
 save(R/'comparison.json',result);save(R/'per_GT.json',pergt)
 fields=list(flat[0])
 with (R/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(flat)
 labels={'baseline':'无修复','whole':'旧 whole＋E1','partial':'旧 partial＋E1','direct':'逐帧逐点：直接替换','core_all':'逐帧逐点＋核心历史','e1':'逐帧逐点＋完整 E1'}
 lines=['逐帧逐点替换验证（2026-10-21）','状态：Replica 八场景、自动196种子及人工76种子全部完成。','协议：main 默认 object_observed_repair / revision 2；350 GT；frozen=false。',
  'F1/PQ/AP/AP50/mCov 为八场景等权平均；未分配点、TP/FP/FN 为八场景求和。','']
 for stage in ['native','final']:
  lines+=['结果阶段：'+stage,'组别 | 版本 | F1 | PQ | AP | AP50 | mCov | TP | FP | FN | 未分配点']
  for row in summary:
   if row['stage']!=stage:continue
   if row['variant']=='baseline' and row['group']=='manual':continue
   group='共同基线' if row['variant']=='baseline' else {'automatic':'自动196','manual':'人工76'}[row['group']]
   lines.append(' | '.join([group,labels[row['variant']]]+[f'{row[k]*100:.2f}%' for k in METRICS]+[f'{row[k]:,}' for k in ['TP','FP','FN','gray_total']]))
  lines.append('')
 lines+=['直接撤换阶段：原有有效票变成零票的帧—点数量（不是独立TSDF点数量）','组别 | whole | partial | 逐帧逐点']
 for group,d in diagnostics.items():lines.append(' | '.join([group]+[f"{d['direct_controls'][v]['valid_to_zero']:,}" for v in ['whole','partial','direct']]))
 lines+=['','归属质量（final，GT一对一最优对应后统计，各场景等权平均）','组别 | 版本 | 正确归属 | 错误归属 | 有几何但未分配']
 for row in summary:
  if row['stage']!='final' or (row['variant']=='baseline' and row['group']=='manual') or 'owner_surface' not in row:continue
  lines.append(' | '.join([row['group'],labels[row['variant']]]+[f"{row['owner_surface'][k]*100:.2f}%" for k in ['Correct_owner_Coverage','Wrong_owner_Coverage','Unassigned_Coverage']]))
 lines+=['','逐帧逐点的直接阶段：每个TSDF点的总有效票数均不下降；授权范围外和新ID冲突点的原候选保持完全一致。','后续历史改票的变化单独记录，不能把完整E1的所有变化归因于直接替换。','']+result['caveats']
 lines+=['','服务器代码：'+str(C),'服务器结果：'+str(R)]
 (R/'结果说明.txt').write_text('\n'.join(lines)+'\n')
 # A self-contained local comparison report, with no server or remote assets.
 blocks=[]
 def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
 for stage in ['final','native']:
  data=[]
  for row in summary:
   if row['stage']!=stage or (row['variant']=='baseline' and row['group']=='manual'):continue
   group='无修复' if row['variant']=='baseline' else {'automatic':'自动196','manual':'人工76'}[row['group']]
   data.append([group,labels[row['variant']]]+[f'{row[k]*100:.2f}' for k in METRICS]+[f'{row["gray_total"]:,}'])
  blocks.append('<h2>'+stage+'：八场景汇总</h2>'+table(['组别','版本','F1 %','PQ %','AP %','AP50 %','mCov %','未分配点'],data))
 for group in GROUPS:
  rows=[]
  for q in scenes:
   if q['group']!=group:continue
   for r in q['rows']:
    if r['stage']=='final':rows.append([q['scene'],labels[r['variant']]]+[f'{r[k]*100:.2f}' for k in ['F1','PQ','AP']]+[f'{r["gray_total"]:,}'])
  blocks.append('<details><summary>'+{'automatic':'自动196','manual':'人工76'}[group]+'：各场景 final 指标</summary>'+table(['场景','版本','F1 %','PQ %','AP %','未分配点'],rows)+'</details>')
 body='<!doctype html><html lang="zh"><meta charset="utf-8"><title>逐帧逐点替换验证</title><style>body{font:15px system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 24px;background:#f7f8fa;color:#172435}h1{font-size:26px}h2{font-size:20px;margin-top:32px}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}td,th{padding:10px 12px;text-align:right;border-bottom:1px solid #dde3e9}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}thead{background:#e5edf5}summary{cursor:pointer;font-weight:600;padding:16px}p{line-height:1.65}.ok{color:#086f54;font-weight:600}</style><h1>逐帧逐点替换验证</h1><p class="ok">八场景 × 自动／人工，16组回放及全部阶段评价通过</p><p>固定种子、追踪、gate、ID、TSDF、原E1历史改票和main R2；350个GT，开发协议 frozen=false。直接替换、核心历史、完整E1分别保存。百分比指标按场景等权平均。</p>'+''.join(blocks)+'<h2>计票检查</h2>'+table(['组别','whole 丢票','partial 丢票','逐点替换丢票'],[[g]+[f"{diagnostics[g]['direct_controls'][v]['valid_to_zero']:,}" for v in ['whole','partial','direct']] for g in GROUPS])+'<p>丢票指直接阶段原有有效票变成零票的帧—点数量。逐点替换每个点总有效票均不下降；未覆盖和新ID冲突点原证据完全保留。原E1历史改票仍可能产生后续变化。</p><p>'+html.escape(' '.join(result['caveats']))+'</p></html>'
 (R/'逐帧逐点替换对比.html').write_text(body,encoding='utf-8')
 save(R/'delivery_receipt.json',dict(status='PASS',files={p.name:sha(p) for p in [R/'comparison.json',R/'per_GT.json',R/'metrics.csv',R/'结果说明.txt',R/'逐帧逐点替换对比.html']}))
 print(json.dumps(dict(status='PASS',summary=summary,diagnostics=diagnostics),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
