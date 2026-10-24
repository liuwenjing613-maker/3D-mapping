from pathlib import Path
import json,collections,datetime,shutil
L=Path(__file__).resolve().parent
s=json.loads((L/'review/summary.json').read_text(encoding='utf-8'))
d=json.loads((L/'diagnostics.json').read_text(encoding='utf-8'))
lines=['七场景完整在线 A/B · 最终交付报告','',
'范围：room1、room2、office0、office1、office2、office3、office4；每景A/B各从第1帧空状态开始，400个建图帧（原始帧0,5,...,1995）。50/100/200/300/400各完成一次修复与提交，共14条分支、70次修复、140份完整PLY。SAM特殊来源仅针对窗户；柜子按普通来源筛选，原P/R语义豁免仍保留。SAM2、Gate、partial+E1及数值规则未变。A候选范围为全部已到达帧，B为当前区间；历史投票都完整。','',
'结论：最后一次400帧修复的AP50，6条分支提高、7条下降、1条不变。新规则没有消除错误修复；office2的瓶子混入同一预测实例、office0的植物越过IoU阈值及office4 B的椅子碎片化，是明确退化。每次完整PLY均可在页面切换查看。','',
'评价范围：main当前object_observed_repair / revision 2，frozen=false。全协议350GT，本批278GT。最终400帧前后固定相同几何与对应索引；不将最终GT用于在线前缀。下表的修复前已经包含50～300帧的历史修复，不是完全无修复基线。A/B是候选策略，不是原版与优化版对照。','',
'场景/分支 | AP50 前→后 | Δpp | AP 前→后 | F1 前→后 | PQ 前→后 | mCov 前→后 | Correct-owner 前→后 | TP/FP/FN 前→后']
pct=lambda x:f'{100*x:.2f}%'
for r in s['rows']:
 a,b=r['before'],r['after'];p,q=a['CA_PRF1_0_5'],b['CA_PRF1_0_5']
 fields=[f"{r['scene']}/{r['branch']}",f"{pct(a['CA_AP50_uniform'])}→{pct(b['CA_AP50_uniform'])}",f"{100*r['AP50_delta']:+.2f}"]
 for fn in [lambda m:m['CA_AP_uniform'],lambda m:m['CA_PRF1_0_5']['F1'],lambda m:m['CA_PQ']['PQ'],lambda m:m['CA_mCov'],lambda m:m['owner_surface']['Correct_owner_Coverage']]:fields.append(pct(fn(a))+'→'+pct(fn(b)))
 fields.append('/'.join(str(p[k]) for k in ['TP','FP','FN'])+'→'+'/'.join(str(q[k]) for k in ['TP','FP','FN']))
 lines.append(' | '.join(fields))
lines+=['','可核验的退化实例（计数在固定R2对应空间，不等于原生PLY点数）：',
'office2 A/B：GT89瓶子IoU均59.90%→16.83%；GT76瓶子A 52.52%→13.25%、B 51.08%→14.32%，两条分支都丢失这两个GT的50匹配。修复后A的ID507/B的ID456同时覆盖两只瓶子和GT87纸巾：A相交66/64/275，B相交66/69/275。对应本次种子均来自原始1550帧model1，A track19/M0161，B track7/M0048。原来各自的ID20、24、21覆盖被削弱，表明输出发生跨对象合并；GT87纸巾IoU也从80.27%/81.60%降到54.35%/53.92%，仍勉强保持50匹配。A TP32→30、FP17→20；B TP33→31、FP13→17。这是本批最明显损伤。',
'office0 A：GT22室内植物（ID57）50.31%→49.61%，丢失50匹配。其正确交集7729→7888，但预测大小12963→13659，额外错误覆盖增长更快。GT9沙发（ID106）94.65%→71.91%，仍保留50匹配；其中5764个GT对应点另归ID871（本次track7/M0095、raw705、model2），显示沙发被切出碎片。整体TP19→18、FP20→25，故不仅有阈值翻转，还有新增误检。',
'office0 B：GT22植物50.61%→46.53%，ID57交集7728→7352，且ID765形成263个植物对应点的额外碎片（track9/M0147、raw1790、model0）；TP19→18、FP17→19。',
'office4 B：GT19椅子54.04%→49.64%。ID141正确交集1659→1524，同时另一个既有ID126的交集489→636（本次track10/M0091、raw1840、model1）。椅子继续分裂，跨过50阈值，TP20→19、FP14→18。',
'office3 A/B：没有GT丢失或新增50匹配，TP均保持37；FP分别26→28、25→26，因此AP50下降主要反映误检增加，而非必然有一个GT大幅损坏。',
'上面的来源是提交身份与本次种子的对应关系，不能单凭最终重叠统计判定具体哪一步Gate/E1造成错误；未进行数值规则消融或手工更换候选。详见diagnostics.json及页面逐GT表。','',
'主要改善：room1 A/B各新增2个50匹配；room2 A新增3个、B新增1个；office1 B新增2个且FP14→13；office4 A新增1个但FP也15→17。具体新增GT与全部微小改善/退化保留在diagnostics.json。','',
'时间与缓存：']
totals=collections.Counter();origins=collections.Counter();fails=[]
for r in s['rows']:
 for a in r['stage_attempts']:
  totals[a['stage']]+=a['seconds']
  if a['returncode']:fails.append(dict(scene=r['scene'],branch=r['branch'],**a))
 for t in r['tracking_origins']:
  for k in ['cases','actual_inferred_cases','exact_cache_reused_cases']:origins[k]+=t[k]
lines+=['从首次启动2026-10-23 13:49 UTC到后处理完成2026-10-24约02:50 UTC，约13小时1分（北京时间10月23日21:49至24日10:50）。包含分支排队、GPU等待、失败恢复、审计、评价及导出；下载和本地页面整理另计。最多4条分支并行。',
f"共{origins['cases']}组最终追踪任务：{origins['actual_inferred_cases']}组真实推理，{origins['exact_cache_reused_cases']}组严格指纹缓存复用；计数从最终case来源重建，包含队列失败前已完成的推理。不能把缓存执行时间称为从头推理耗时。",
'以下是14分支各阶段调用墙钟的累计，不是全任务墙钟；有并行重叠，并保留失败调用。selection_tracking包含候选生产、追踪及等待，不能视为纯SAM2时间。']
for k,v in totals.items():lines.append(f'  {k}: {v:.2f}秒（{v/3600:.3f}小时）')
lines+=['失败与恢复：4条首批分支的50帧候选阶段遇到场景catalog路径适配问题，修正路径后保留已完成建图续跑。office3 A50遇到两个线程写同一gpu_wait.json.tmp的文件重命名竞争，临时文件名改为含PID与线程ID；8线程800次状态写入压测通过。其余JSON序列化及数值算法未变，已完成追踪保留。共5次失败调用记录，合计'+f"{sum(x['seconds'] for x in fails):.2f}秒。",'',
'完整性核验：70次提交与历史像素、持久ID、严格投票核验通过；14条分支首次50帧从空状态开始，实际51/101/201/301帧均验证使用已提交修复状态。840个冻结数值代码文件核验通过。review中的1147份小型证据SHA-256通过；140份PLY哈希和每组前后逐元素坐标通过。PLY保留全部float64坐标、int32真实ID；页面颜色来自导出ID，不为消除差异而手工统一候选。','',
'限制：room0此前只在400帧重跑新规则，早期历史不同，不能合为同条件八景。此前100→200原版/优化版真实闭环存在Gate阈值翻转及2点最终标签差异，严格等价没有通过；固定同表面控制精确一致不能替代真实闭环一致。本批结果不解决这一限制，也没有与七场景旧柜子特殊规则全流程做成对比较，不能据此宣称新规则在七景整体优于旧规则。','',
'页面：http://127.0.0.1:8801/replica7/index.html','文件：deliverables/<scene>/<scene>_<A/B>_<checkpoint>_<before/after>.ply','详细证据：review/summary.json、review/evidence_manifest.json、diagnostics.json、delivery_verification.json。']
(L/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
W=L.parent/'online_full_scene_room0_AB_20261023/comparison_page/replica7';W.mkdir(exist_ok=True,parents=True);shutil.copyfile(L/'FINAL_REPORT.txt',W/'FINAL_REPORT.txt')
(L/'timing_delivery.json').write_text(json.dumps(dict(stage_attempt_seconds=dict(totals),tracking_origin=dict(origins),failed_attempts=fails),indent=2)+'\n')
print('REPORT_READY')
