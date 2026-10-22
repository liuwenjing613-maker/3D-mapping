from pathlib import Path
import json, html
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
R=T/'review';D=R/'pr95-90'
source=json.loads((R/'threshold95_90_independent_audit.json').read_text())
assert source['status']=='PASS'
for arm in ['A','B']:
 assert (D/arm/'index.html').is_file(),f'{arm} gallery not ready'
 assert (T/'pr95_90'/arm/'seed_manifest.json').is_file(),f'{arm} seeds not ready'
 assert (T/'pr95_90'/arm/'audit.json').is_file(),f'{arm} QA not ready'
 qa=json.loads((T/'pr95_90'/arm/'audit.json').read_text())
 actual=json.loads((T/'pr95_90'/arm/'comparison.json').read_text())
 assert qa['status']=='PASS' and actual['status']=='PASS'
 assert qa['only_stage4_skip_condition_changed'] and qa['stage5_pr_rule_and_semantic_exemption_unchanged']
 assert not qa['tracking_performed'] and not qa['repair100_applied'] and not qa['map_write']
 assert actual['new_retained_group_ids']==source['arms'][arm]['new_retained']
 assert actual['newly_excluded_group_ids']==source['arms'][arm]['newly_removed']

def pct(x): return '—' if x is None else f'{100*x:.3f}%'
def table(arm):
 a=source['arms'][arm]
 out='<table><thead><tr><th>候选</th><th>来源处理帧</th><th>p</th><th>r</th><th>背景模型预测</th></tr></thead><tbody>'
 for x in a['rows']:
  if not x['newly_removed']:continue
  out+=f"<tr><td>{x['group_id']} · {x['mask_id']}</td><td>{x['frame']//5+1}（raw{x['frame']}）</td><td>{pct(x['p'])}</td><td>{pct(x['r'])}</td><td>{html.escape(x['background_top_class'] or '—')}</td></tr>"
 return out+'</tbody></table>'
A=source['arms']['A'];B=source['arms']['B']
summary={'status':'PASS','checkpoint_processed_frame':100,'raw_cutoff':495,'rule_version':'pr95_90','new_exclusion':'(p > 0.95 and r > 0.90) or (r > 0.95 and p > 0.90)','stage_modified':'stage4_skip_repair_B','old_exclusion':'p >= 0.95 and r >= 0.95','stage5_unchanged':True,'stage5_semantic_exemption_unchanged':True,'new_tracking_run':False,'repair100_applied':False,'same_geometry_and_measurements':True,'arms':{}}
for arm,a in [('A',A),('B',B)]:
 summary['arms'][arm]={'old_retained':len(a['old_retained']),'retained':len(a['new_retained']),'newly_removed':a['newly_removed'],'newly_restored':a['newly_restored'],'retained_group_ids':a['new_retained'],'page_url':f'http://127.0.0.1:8798/online-repair/pr95-90/{arm}/'}
D.mkdir(parents=True,exist_ok=True)
(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
style='''*{box-sizing:border-box}body{font:16px/1.7 system-ui,sans-serif;max-width:1160px;margin:28px auto;padding:20px;color:#1c3440;background:#f2f6f7}h1{font-size:30px}h2{font-size:22px;margin-top:0}section,.card{background:white;border:1px solid #dae5e7;padding:24px;border-radius:14px;margin:18px 0}a{color:#08747b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{margin:0}.number{font-size:44px;font-weight:750;color:#08747b}.muted{font-size:14px;color:#607681}.button{display:inline-block;border-radius:8px;background:#08747b;color:white;padding:10px 16px;text-decoration:none}.notice{background:#fff5df;border-left:4px solid #dba034;padding:14px 18px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dfe7e9;text-align:left;vertical-align:top}.formula{font-size:20px;font-weight:650;background:#eaf6f3;border-radius:8px;padding:15px}.tag{font-size:13px;color:#607681}@media(max-width:700px){body{padding:10px}.grid{grid-template-columns:1fr}td,th{padding:6px;font-size:13px}}'''
out=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第100帧 · 新95%/90%规则留存结果</title><style>{style}</style><div class="tag">ROOM0 · 第100帧 · 最新留存结果</div><h1>95% / 90% 规则重筛</h1><p>两组都使用第50帧修复后续建到100帧的同一地图。仅调整“已经足够一致、跳过修复”的条件，重新执行完整筛选流程。</p><p class="formula">排除条件：p ＞ 95% 且 r ＞ 90%，或 r ＞ 95% 且 p ＞ 90%</p><p class="notice"><strong>第100帧实际修复尚未执行。</strong>本页展示新规则的留存与新增排除。此前13/12个种子的追踪属于旧规则；本轮尚未为新集合重新追踪或计算门控。</p><div class="grid"><section class="card"><div class="tag">A · 从处理帧1～100选候选</div><h2>全历史来源</h2><div class="number">{len(A['new_retained'])}<span style="font-size:18px"> 个保留</span></div><p>原来{len(A['old_retained'])}个 → 现在{len(A['new_retained'])}个，本次新增排除{len(A['newly_removed'])}个。</p><a class="button" href="A/">检查A组留存mask</a><p class="muted">来源raw0～495；保留组：{', '.join(A['new_retained'])}</p></section><section class="card"><div class="tag">B · 只从处理帧51～100选候选</div><h2>新增50帧来源</h2><div class="number">{len(B['new_retained'])}<span style="font-size:18px"> 个保留</span></div><p>原来{len(B['old_retained'])}个 → 现在{len(B['new_retained'])}个，本次新增排除{len(B['newly_removed'])}个。</p><a class="button" href="B/">检查B组留存mask</a><p class="muted">来源raw250～495；保留组：{', '.join(B['new_retained'])}</p></section></div><section><h2>改动作用在哪一步</h2><table><tr><th>步骤</th><th>本轮规则</th></tr><tr><td>第4步：跳过已一致的候选</td><td>由原代码的p≥95%且r≥95%，改为上面的严格95%/90%对称条件。所有进入本步骤的候选都适用。</td></tr><tr><td>第5步：原p/r过滤</td><td>保留原p＞10%且r＜20%的排除条件及窗/柜语义豁免。豁免只作用于第4步幸存的候选，不恢复第4步已排除项。</td></tr></table><p>全局和局部触边仍最先排除，包含、去重、来源模型选择和评分数据保持原样。独立审查先逐阶段重现旧13/12结果，再核对新版完整流程与整数公式，结果一致。</p><p class="muted">比例判断使用未四舍五入的计数，表中百分比只用于显示。两组都没有恰好90%或95%的边界样本。</p></section><section><h2>A组本次新增排除</h2>{table('A')}</section><section><h2>B组本次新增排除</h2>{table('B')}</section><section><h2>仍然留下的方窗</h2><p>两组的同一个方窗种子仍保留：p=91.888%，r=68.696%，不满足新的排除条件。它的内部漏分和旧规则下的门控分组问题仍需结合图像检查；本轮没有补填mask或改变门控算法。</p><p><a href="../tracking100/paired/">旧规则追踪的同帧证据</a> · <a href="../compare100/">旧13/12候选及追踪、计时对照</a></p></section><section><h2>核验与原始记录</h2><p><a href="../threshold95_90_independent_audit.json">独立完整筛选核验</a> · <a href="summary.json">本轮结果摘要</a> · <a href="../report50.html#main-mapping-time">主建图及第50帧修复计时</a> · <a href="../50/">第50帧实例PLY</a></p><p class="muted">结果保留在服务器，通过当前通道网页查看。</p></section></html>'''
(D/'index.html').write_text(out)
banner='<p class="notice" id="latest-pr95-90"><strong>此页为旧双95规则结果（A13 / B12）。</strong> 最新95%/90%重筛结果为A6 / B8，<a href="../pr95-90/">点击检查最新留存mask</a>。下面的追踪与门控属于旧候选集合。</p>'
p=R/'compare100/index.html';s=p.read_text()
if 'id="latest-pr95-90"' not in s:s=s.replace('<h1>候选来源范围对照</h1>','<h1>候选来源范围对照</h1>'+banner)
p.write_text(s)
p=R/'compare100/summary.json';d=json.loads(p.read_text());d.update(rule_version='original_double95',latest_candidate_page='../pr95-90/',new_rule_tracking_run=False,second_repair_applied=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
p=R/'index.html';s=p.read_text()
if 'id="latest-pr95-90"' not in s:
 s=s.replace('<section><h2>第100帧两组候选与追踪</h2>','<section id="latest-pr95-90"><h2>第100帧 · 新95%/90%规则</h2><p><a href="pr95-90/"><strong>打开最新留存：A6个 / B8个</strong></a></p><p>本轮只重筛候选，尚未用新集合重跑追踪，100帧实际修复未执行。</p></section><section><h2>第100帧旧规则候选与追踪</h2>')
p.write_text(s)
p=R/'status.json';d=json.loads(p.read_text());d.update(latest_candidate_page='pr95-90/',latest_candidate_rule='pr95_90',latest_candidate_counts={'A':6,'B':8},latest_candidate_tracking_ready=False,repair100_applied=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps({'status':'PASS','new_counts':{'A':len(A['new_retained']),'B':len(B['new_retained'])},'page':'http://127.0.0.1:8798/online-repair/pr95-90/'},ensure_ascii=False))
