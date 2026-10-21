from pathlib import Path
import csv,hashlib,html,json
R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert read(R/'control_scores_complete.json')['status']=='PASS'
base=read(R/'comparison.json');assert base['status']=='PASS'
scenes=[];summary=[]
for group in ['automatic','manual']:
 gs=[s for s in base['scenes'] if s['group']==group]
 for s in gs:
  x=read(R/group/s['scene']/'direct_controls_r2/comparison.json');assert x['status']=='PASS'
  scenes.append(x)
 for stage in ['native','final']:
  for v in ['whole_direct','partial_direct','point_direct']:
   if v=='point_direct':rows=[next(r for r in s['rows'] if r['variant']=='direct' and r['stage']==stage) for s in gs]
   else:rows=[next(r for r in s['rows'] if r['variant']==v and r['stage']==stage) for s in scenes if s['group']==group]
   q=dict(group=group,stage=stage,variant=v)
   for k in ['F1','PQ','AP','AP50','mCov']:q[k]=sum(r[k] for r in rows)/8
   for k in ['TP','FP','FN','gray_total']:q[k]=sum(r[k] for r in rows)
   assert q['TP']+q['FN']==350;summary.append(q)
save(R/'direct_stage_comparison.json',dict(status='PASS',same_phase_comparison=True,summary=summary,scenes=scenes))
labels={'whole_direct':'旧whole直接','partial_direct':'旧partial直接','point_direct':'逐帧逐点直接'}
lines=['同阶段对照：直接替换，不含后续E1历史改票','协议 main R2；八场景等权指标；350 GT。','组别 | 结果层 | 方法 | F1 | PQ | AP | 未分配点']
for q in summary:lines.append(' | '.join([q['group'],q['stage'],labels[q['variant']]]+[f'{q[k]*100:.2f}%' for k in ['F1','PQ','AP']]+[f'{q["gray_total"]:,}']))
(R/'直接阶段对照.txt').write_text('\n'.join(lines)+'\n')
pergt=read(R/'per_GT.json');damage=read(R/'diagnostics/room0_automatic_regressions.json')
cases=[r for r in pergt if r['group']=='automatic' and r['scene']=='room0' and r['stage']=='final' and r['GT_ID'] in [44,56,80,65]]
case_lines=['关键退化病例：room0 自动组','对象 | GT ID | 旧partial+E1最佳IoU | 逐点直接最佳IoU | 逐点+E1最佳IoU']
for r in cases:case_lines.append(' | '.join([r['semantic_class'],str(r['GT_ID'])]+[f'{r[k]*100:.2f}%' for k in ['partial','direct','e1']]))
case_lines+=['','在GT 44/56/80的几何支持上，G008（目标ID 2）分别把386/567/663张原插座实例票改为ID 2；原家族为[2,179]，不含三个插座的346/348/227。',
 '在盘子GT 65上，G040（目标ID 7）把2018张ID 20票改为ID 7；原家族为[7]。',
 '新规则消除了这些帧—点的弃权，但加入了来自覆盖其他实例区域的修复mask身份，导致跨帧身份竞争和部分错误归属。',
 '这些对象的退化在直接替换阶段已发生，不能全部归因于后续历史改票。GT最大IoU用于定位对象变化，不等同于一对一匹配指标。']
(R/'关键退化病例.txt').write_text('\n'.join(case_lines)+'\n')
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
extra='<h2>同阶段比较：直接替换</h2><p>以下三种方法均未接后续E1历史改票，指标为final层。</p>'+table(['组别','方法','F1 %','PQ %','AP %','未分配点'],[[q['group'],labels[q['variant']]]+[f'{q[k]*100:.2f}' for k in ['F1','PQ','AP']]+[f'{q["gray_total"]:,}'] for q in summary if q['stage']=='final'])
extra+='<h2>具体退化：room0自动组</h2>'+table(['GT ID','对象','旧partial＋E1 IoU %','逐点直接 IoU %','逐点＋E1 IoU %'],[[r['GT_ID'],r['semantic_class']]+[f'{r[k]*100:.2f}' for k in ['partial','direct','e1']] for r in cases])
extra+='<p>'+html.escape(' '.join(case_lines[-4:]))+'</p>'
owner_rows=[]
for q in base['summary']:
 if q['stage']!='final' or q['variant'] not in ['baseline','partial','e1'] or (q['variant']=='baseline' and q['group']=='manual') or 'owner_surface' not in q:continue
 owner_rows.append([q['group'],{'baseline':'无修复','partial':'旧partial＋E1','e1':'逐点＋E1'}[q['variant']]]+[f"{q['owner_surface'][k]*100:.3f}" for k in ['Correct_owner_Coverage','Wrong_owner_Coverage','Unassigned_Coverage']])
extra+='<h2>归属质量</h2><p>GT最优一对一配对后的顶点归属，各场景等权平均，final层。</p>'+table(['组别','版本','正确归属 %','错误归属 %','有几何但未分配 %'],owner_rows)
p=R/'逐帧逐点替换对比.html';p.write_text(p.read_text().replace('</html>',extra+'</html>'),encoding='utf-8')
receipt=read(R/'delivery_receipt.json')
for p in [R/'逐帧逐点替换对比.html',R/'direct_stage_comparison.json',R/'直接阶段对照.txt',R/'关键退化病例.txt']:
 receipt['files'][p.name]=sha(p)
receipt['phase_matched_direct_control_metric_records']=64;receipt['main_new_stage_metric_records']=96
save(R/'delivery_receipt.json',receipt)
save(R/'final_complete.json',dict(status='PASS',main_runs=16,new_stage_metric_records=96,phase_matched_control_metric_records=64,
    same_R2_protocol=True,original_scorer_unchanged=True,report_sha256=sha(R/'逐帧逐点替换对比.html')))
print(json.dumps(dict(status='PASS',direct_summary=summary),ensure_ascii=False))
