"""Publish the actual four-map experiment, its full PLYs and a fixed-cohort comparison."""
from pathlib import Path
import hashlib,html,json,re,shutil
WORK=Path(__file__).resolve().parent
BASE=WORK.parents[1]/'results/固定案例_三模型对比_20261006'
WEB=BASE/'continuous_tracking_ablation_20261008'
DATA=WORK/'server_results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(x):return f'{x*100:.2f}'
def delta(x,old,higher=True):
 d=x-old;c='better' if (d>0 if higher else d<0) else 'worse' if d else ''
 return f'<span class="{c}">{pct(x)}</span>'

def main():
 complete=read(DATA/'complete.json');assert complete['status']=='PASS'
 models=WEB/'models';models.mkdir(exist_ok=True)
 scenes=complete['scenes'];reports={};blocks=[];identity_blocks=[];facts={}
 for scene,dirname,original_work in [('room0','room0_repair_batch_20261007','room0_human_repair_20261007'),('room2','room2_top30_repair_20261008','room2_top30_repair_20261008')]:
  summary=scenes[scene];audit=read(DATA/scene/'comparison_audit.json');info=read(DATA/scene/'comparison_pack.json')
  assert summary['status']==audit['status']==info['status']=='PASS'
  assert info['roundtrip_geometry_and_labels_exact'] and not info['GT_used_for_rendering'] and not info['downsampled']
  source=read(BASE/dirname/'comparison_data.json');playback=read(WEB/(scene+'_preview_data.json'))
  info['cases']=source['model']['cases'];ids=sorted({t['persistent_id'] for t in playback['tracks']});info['selected_persistent_ids']=ids
  info['identity_sources']={str(pid):[t['ROI']+' mask'+str(t['mask']) for t in playback['tracks'] if t['persistent_id']==pid] for pid in ids}
  info['path']=scene+'_comparison_all_points.bin'
  for n in ['comparison_all_points.bin','gap0_full_instance.ply','gap1_full_instance.ply']:
   original=(DATA/scene/n) if n.startswith('comparison_') else (DATA/scene/n[:4]/(scene+'_'+n))
   dest=models/(scene+'_'+n)
   if n.startswith('comparison_'):assert sha(original)==info['sha256']
   else:assert sha(original)==summary['conditions'][n[:4]]['PLY']['sha256']
   if not dest.exists() or sha(dest)!=sha(original):shutil.copyfile(original,dest)
  dump(models/(scene+'_comparison_pack.json'),info)
  source_eval=read(WORK.parents[0]/original_work/'evaluation_summary.json')
  original_report=read(WORK.parents[0]/original_work/('full_track_complete.json' if scene=='room0' else 'validated/full_track/complete.json'))
  rows=[];reference=summary['unrestricted_metrics'];target_original=audit['unrestricted']['target_mean_best_IoU']
  states=[('baseline','原始P1-A1',summary['baseline_metrics'],original_report['whole_scene_unknown_before'],audit['baseline']),
          ('unrestricted','原完整追踪修复',reference,original_report['whole_scene_unknown_after'],audit['unrestricted'])]
  states.extend((name,'严格连续' if name=='gap0' else '允许空1帧',row['metrics'],row['repair']['whole_scene_unknown_after'],audit['conditions'][name]) for name,row in summary['conditions'].items())
  for name,title,m,unknown,a in states:
   changed=name in ['gap0','gap1'];c='condition' if changed else 'reference' if name=='unrestricted' else ''
   unknown_class='better' if unknown<original_report['whole_scene_unknown_after'] else 'worse' if unknown>original_report['whole_scene_unknown_after'] else ''
   vals=[delta(m['CA_AP50_uniform'],reference['CA_AP50_uniform']),delta(m['CA_PRF1_0_5']['F1'],reference['CA_PRF1_0_5']['F1']),delta(m['CA_PQ']['PQ'],reference['CA_PQ']['PQ'])] if changed else [pct(m['CA_AP50_uniform']),pct(m['CA_PRF1_0_5']['F1']),pct(m['CA_PQ']['PQ'])]
   rows.append(f'<tr class="{c}"><td>{title}</td>'+''.join('<td>'+v+'</td>' for v in vals)+f'<td class="{unknown_class if changed else ""}">{unknown:,}</td><td>{delta(a["target_mean_best_IoU"],target_original) if changed else pct(a["target_mean_best_IoU"])}</td><td>{m["structure"]["split_gt_count"]}</td><td>{m["structure"]["merge_prediction_count"]}</td></tr>')
  item=summary['conditions']['gap1'];v=item['vs_unrestricted'];a=audit['conditions']['gap1'];same=info['gap0_gap1_array_equality']['instance_id']
  facts[scene]={'gap0_gap1_final_labels_identical':same,'unknown_delta_vs_unrestricted':item['repair']['whole_scene_unknown_after']-original_report['whole_scene_unknown_after'],
   'AP50_delta_vs_unrestricted':item['metrics']['CA_AP50_uniform']-reference['CA_AP50_uniform'],
   'F1_delta_vs_unrestricted':item['metrics']['CA_PRF1_0_5']['F1']-reference['CA_PRF1_0_5']['F1'],
   'PQ_delta_vs_unrestricted':item['metrics']['CA_PQ']['PQ']-reference['CA_PQ']['PQ'],
   'target_best_IoU_delta_vs_unrestricted':v['target_mean_best_IoU_delta'],
   'target_mean_completeness_delta_vs_unrestricted':v['target_completeness_mean_delta'],
   'non_target_lost':v['non_target_previously_correct_lost'],'non_target_degraded_gt':[row['GT_id'] for row in a['non_target_IoU_degraded_over_0_01']],
   'new_unknown_points':v['new_unknown_points'],'restored_assigned_points':v['restored_assigned_points'],
   'surface_points':info['points']}
  caution=f'非目标原正确匹配丢失 {len(v["non_target_previously_correct_lost"])} 个；非目标IoU下降超过1个百分点 {len(a["non_target_IoU_degraded_over_0_01"])} 个。'
  example_gt,example_roi,example_pid=(5006,'ROI-C0001',799) if scene=='room0' else (5005,'ROI-C0008',359)
  example=next(row for row in a['per_GT_comparison'] if row['GT_id']==example_gt)
  example_note=f'<p><a href="model.html?scene={scene}&roi={example_roi}">直接查看明显退化区域：{example_roi} · ID{example_pid}</a>。该GT的v3最佳实例IoU从 {pct(example["before"]["best_IoU"])}% 降至 {pct(example["after"]["best_IoU"])}%，失去原有的正确匹配。</p>'
  blocks.append(f'''<section class="results"><div class="scene-head"><h2>{scene} · 实际换票与扩散结果</h2><span><a class="button" href="model.html?scene={scene}">完整3D并排对照</a> <a class="button" href="playback.html?scene={scene}&policy=gap1">逐帧mask对照</a></span></div>
  <div class="table-wrap"><table class="metrics"><thead><tr><th>方案</th><th>AP50 ↑</th><th>F1 ↑</th><th>PQ ↑</th><th>未分配点 ↓</th><th>目标平均IoU ↑</th><th>拆分GT数</th><th>合并实例数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
  <p class="muted">AP50、F1、PQ、IoU均以百分数显示。目标集合固定为原评估的 {len(summary['fixed_original_target_GT_ids'])} 个GT；统计整幅地图。绿/红表示相对“原完整追踪修复”变好/变差；灰点减少本身不证明实例更正确。</p>
  <p>{'两档最终实例标签逐点相同。' if same else '两档最终实例标签存在差异。'}允许空1帧：原有实例转为灰点 {v['new_unknown_points']:,} 个，原灰点获得实例 {v['restored_assigned_points']:,} 个。{caution}</p>{example_note}<p class="muted">最终图相同的原因：{'允许空1帧多保留的74个mask×建图帧，均未通过原有可靠性门槛。' if scene=='room0' else '允许空1帧多保留的1个非空mask位于非建图帧。'}原生证据、投票账及最终NPZ文件的SHA完全相同。</p>
  <p class="muted"><a href="models/{scene}_gap0_full_instance.ply" download>严格连续完整彩色PLY</a> · <a href="models/{scene}_gap1_full_instance.ply" download>允许空1帧完整彩色PLY</a> · <a href="reports/{scene}_evaluation_summary.json">统一v3报告</a> · <a href="reports/{scene}_comparison_audit.json">逐GT与等价性检查</a></p></section>''')
  diag_rows=[];old_diag={row['persistent_id']:row for row in source_eval['fixed_surface_diagnostics'] if row.get('GT_id') is not None}
  new_diag={row['expected_persistent_id']:row for row in item['fixed_identity_diagnostics']['targets']}
  assert set(old_diag)==set(new_diag)
  for pid,old in old_diag.items():
   after=new_diag[pid];before=old['conditions']['full_track'];track=next(t for t in playback['tracks'] if t['persistent_id']==pid)
   diag_rows.append(f'<tr><td><a href="model.html?scene={scene}&roi={track["ROI"]}">{track["ROI"]} · ID{pid}</a></td><td>{old["GT_id"]}</td><td>{pct(before["expected_identity_fixed_IoU"])}</td><td>{delta(after["expected_identity_fixed_IoU"],before["expected_identity_fixed_IoU"])}</td><td>{pct(before["expected_identity_full_GT_coverage"])} → {pct(after["expected_identity_full_GT_coverage"])}</td><td>{pct(before["expected_identity_reference_purity"])} → {pct(after["expected_identity_reference_purity"])}</td></tr>')
  identity_blocks.append(f'<details><summary>{scene} · 人工选定身份的覆盖、纯度与IoU</summary><p class="muted">按固定身份逐个统计；多个身份对应同一GT仍分别列出。此处GT仅用于完成修复后的评估，未用于截断或建图。右列为允许空1帧结果。</p><div class="table-wrap"><table><thead><tr><th>对应目标</th><th>GT</th><th>原身份IoU</th><th>新身份IoU</th><th>原→新覆盖率</th><th>原→新纯度</th></tr></thead><tbody>{"".join(diag_rows)}</tbody></table></div></details>')
  for name in ['evaluation_summary.json','comparison_audit.json','source_freeze.json','predictions_frozen.json','evaluation_freeze.json']:
   dest=WEB/'reports'/(scene+'_'+name);dest.parent.mkdir(exist_ok=True);shutil.copyfile(DATA/scene/name,dest)
  reports[scene]=summary
 # Retain the verified trajectory timelines, collapsed below the actual map results.
 donor=WEB/'preview_timeline_snapshot.html'
 if not donor.exists():shutil.copyfile(WEB/'index.html',donor)
 old=donor.read_text(encoding='utf-8');timelines=re.findall(r'<section>.*?</section>',old,re.S);assert len(timelines)==2
 good=all(f['F1_delta_vs_unrestricted']>=0 and f['PQ_delta_vs_unrestricted']>=0 and not f['non_target_lost'] for f in facts.values())
 conclusion=('本轮整体匹配指标未下降，但仍应结合目标覆盖和局部错误审查后决定是否替换默认规则。' if good else '本轮没有证明硬截断能稳定改善整体地图，建议保留为诊断对照，暂不替换现有默认修复结果。')
 extra=''.join(f'<li>{scene}：F1变化 {f["F1_delta_vs_unrestricted"]*100:+.2f} 个百分点，PQ变化 {f["PQ_delta_vs_unrestricted"]*100:+.2f} 个百分点；未分配点净变化 {f["unknown_delta_vs_unrestricted"]:+,}。</li>' for scene,f in facts.items())
 css='''*{box-sizing:border-box}body{margin:0;background:#edf2f6;color:#193044;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:18px}h1{font-size:26px;margin:0}h2{font-size:20px;margin:0}p{margin:8px 0}a{color:#146d86}.notice{background:#e5f3ee;border:1px solid #95c9b8;padding:12px;border-radius:9px}.muted{font-size:13px;color:#607688}.results,details{background:white;border:1px solid #d0dce6;padding:15px;border-radius:10px;margin-top:14px}.scene-head{display:flex;gap:12px;justify-content:space-between;align-items:center}.button{display:inline-block;background:#176c83;color:white;padding:6px 10px;border-radius:6px;text-decoration:none;margin-left:5px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dfe7ed;padding:8px;text-align:left;vertical-align:middle}th{background:#f2f5f8}table.metrics td:not(:first-child){font-variant-numeric:tabular-nums}tr.reference{background:#f3f6f8;font-weight:600}.better{color:#147b58}.worse{color:#b5483e}summary{cursor:pointer;font-weight:600}.rules{display:grid;grid-template-columns:1fr 1fr;gap:12px}.rules>div{background:white;border:1px solid #d0dce6;border-radius:8px;padding:12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0}.cards>div{background:#edf4f7;padding:12px;border-radius:8px;font-size:13px}.cards strong{display:block;font-size:23px}svg{width:100%;height:24px;display:block}td:nth-child(3){min-width:70px}@media(max-width:900px){.scene-head{display:block}.scene-head>span{display:block;margin-top:10px}.rules{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}}'''
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>room0 / room2 · 连续性修复实验结果</title><style>{css}</style><main><h1>room0 / room2 · 连续性修复实验结果</h1><p class="notice"><strong>四组完整地图、扩散与统一v3对照均已完成。</strong>原地图保留。这里展示实际结果，完整3D和逐帧mask均可直接打开。</p>{''.join(blocks)}
 <section class="results"><h2>这次实验说明什么</h2><p>{conclusion}</p><ul>{extra}</ul><p>该规则能阻断断档后的无依据续接，也会丢掉正常换视角后的重现；无法发现“始终非空却已追偏”。停止某条SAM轨迹，只停止它之后的修复贡献，不会自动删除这些帧中原有的CropFormer观测。</p><p>下一步建议：检查连续区间内的漂移，验证重新出现的mask是否仍属同一物体；确认后再建立独立追踪段。这个建议尚未加入本轮算法。</p></section>
 <details><summary>本轮具体改了什么，以及正确性检查</summary><div class="rules"><div><strong>严格连续</strong><p>原始mask一帧为空，就永久停止该轨迹的这个方向。</p></div><div><strong>允许空1帧</strong><p>连续两帧为空，就永久停止该轨迹的这个方向。</p></div></div><p>从各自人工种子，前向和后向独立计数；按全部2000原始帧，空mask为原始1200×680保存结果的像素数=0。各轨迹独立，包括共用身份的多种子轨迹。</p><p>本轮未重新跑SAM：对已保存结果做精确截断，再真实撤回/新增投票、确认、扩散和有界提交。保留mask像素完全不变。10项连续性测试、4000原始帧无损mask检查、完整帧重放与增量票账一致、撤回精确回滚、修复范围外标签变化为0均通过。</p><p>人工标注、原身份与别名、可靠性及替换门槛、投票权重、扩散参数、v3代码/协议/开关全部沿用原方案。补充报告改为逐个身份统计同GT的多个选定mask；3D导出用原有固定配色函数补齐遗漏的恢复ID。上述两项只修正报告与显示。</p><p class="muted">沿用当前v3调试协议，协议仍待冻结；这些分数用于同协议离线对照，不能作为正式benchmark结果。GT只用于修复完成后的评估。</p></details>
 {''.join(identity_blocks)}<details><summary>查看全部55条轨迹的连续区间和断档位置</summary>{''.join(timelines)}</details><p class="muted"><a href="experiment_summary.json">完整实验JSON</a> · <a href="reports/native_mask_validation.json">原始mask无损验证</a> · <a href="../room0_tracking_playback_20261008/index.html">room0原轨迹</a> · <a href="../room2_tracking_playback_20261008/index.html">room2原轨迹</a></p></main></html>'''
 (WEB/'index.html').write_text(page,encoding='utf-8')
 preview=read(WEB/'experiment_summary.json')
 dump(WEB/'experiment_summary.json',{'status':'PASS','repair_and_v3_executed':True,'SAM_rerun':False,
      'formal_benchmark_result':False,'old_maps_preserved':True,'raw_mask_truncation_statistics':preview.get('raw_mask_truncation_statistics',preview),
      'facts':facts,'server_complete':complete})
 dump(WEB/'experiment_status.json',{'status':'PASS','stage':'complete','all_four_maps_complete':True,'unified_v3_complete':True,'SAM_rerun':False})
 shutil.copyfile(WORK/'native_mask_validation.json',WEB/'reports/native_mask_validation.json')
 shutil.copyfile(WORK/'why_policies_equal.json',WEB/'reports/why_policies_equal.json')
 dump(WORK/'final_publish_receipt.json',{'status':'PASS','URL':'http://127.0.0.1:8785/continuous_tracking_ablation_20261008/index.html',
    'facts':facts,'website_files_sha256':{str(p.relative_to(WEB)):sha(p) for p in WEB.rglob('*') if p.is_file()},'original_maps_modified':False})
 print(json.dumps(facts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
