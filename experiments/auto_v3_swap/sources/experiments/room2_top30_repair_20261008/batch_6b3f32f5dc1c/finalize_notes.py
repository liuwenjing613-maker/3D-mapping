"""Write the full outcome, including regressions, from frozen receipts only."""
from pathlib import Path
import json,tarfile
HERE=Path(__file__).resolve().parent
def read(n):return json.loads((HERE/n).read_text(encoding='utf-8'))
def pct(n):return f'{100*n:.4f}%'
def main():
 ev=read('evaluation_summary.json');full=read('validated/full_track/complete.json');seed=read('validated/seed_only/complete.json')
 with tarfile.open(HERE/'review_bundle.tar.gz') as bundle:
  data=json.load(bundle.extractfile('room2_top30_repair_20261008/comparison_data.json'))
 frames=data['timeline']['frames'];delta_frames=sum(d.get('removed_votes',0)>0 or d.get('added_votes',0)>0 for d in frames)
 reasons=sorted({reason for d in frames for c in d.get('objects',{}).values() for reason in c.get('reasons',[])})
 cross=read('cross_seed_review/comparison.json');decisions={d['frame']:d for d in frames}
 lines=['# room2 本轮修复：指标改善，完整度与结构仍有问题','',
 '原始导出18处标注：14组可用选择、25个mask，4处不可靠；12处未标注。所有选择逐像素核对，原标注和基线保留。14组各完成2000帧独立前后传播，28,000张传播图已核对；只在原来的400个建图帧换票。','',
 'C0002 mask7/C0029 mask4，以及C0002 mask8/C0029 mask10分别是同一把椅子。RGB和种子几何核对后，冻结24→4、25→5，共23个身份；这个决定早于换票和任何GT评估。其他mask不按GT合并。','',
 '数字质量门槛、投票权重与去重、原生至少2票/占比≥0.67、后处理、v3代码/参数保持。旧家族仍按原始观察的主导ID确定，尚未扩展完整历史谱系。局部种子不推测补齐，原始CropFormer选择恢复为同帧完整原mask。','',
 '## 统一v3的真实指标','',
 '| 版本 | AP50 | F1 | PQ | TP/FP/FN | 合并预测/分裂GT |','|---|---:|---:|---:|---|---|']
 metrics={'原始地图':ev['baseline_v3_metrics'],'仅种子':ev['comparisons']['seed_only']['scene_metrics'],'完整双向追踪':ev['comparisons']['full_track']['scene_metrics']}
 for title,m in metrics.items():
  f=m['CA_PRF1_0_5'];s=m['structure'];lines.append(f'| {title} | {pct(m["CA_AP50_uniform"])} | {pct(f["F1"])} | {pct(m["CA_PQ"]["PQ"])} | {f["TP"]}/{f["FP"]}/{f["FN"]} | {s["merge_prediction_count"]}/{s["split_gt_count"]} |')
 cmp=ev['comparisons']['full_track'];gray=full['whole_scene_unknown_after']-full['whole_scene_unknown_before']
 lines+=['',f'固定目标平均最佳IoU变化：{100*cmp["target_mean_IoU_delta"]:+.4f}个百分点。严格成功：{cmp["strict_success"]}，结果为mixed。目标结构计数变化：{cmp["target_structure_count_delta"]:+d}。未选目标原先正确匹配丢失：{cmp["non_target_previously_correct_lost"]}；最佳IoU下降超过1个百分点：{cmp["non_target_IoU_degraded_over_0_01"]}。',
 '', '协议仍为统一v3的DEBUG_ONLY / NON_OFFICIAL，保持原参数，GT只在预测冻结后用于评估，不作为正式主表结果。目标定义仍按人工种子表面、1cm近邻、至少10点/至少5%匹配确定；非目标无损伤仅限这些既定指标与目标定义。','',
 f'385/400帧通过检查并参与重建；其中{delta_frames}/400帧确实产生不同的票据。最终标签改变{full["strict_commit"]["final_inside_changes"]:,}点；范围外0点。全图灰点{full["whole_scene_unknown_before"]:,}→{full["whole_scene_unknown_after"]:,}（{gray:+,}）。已有实例变灰{full["assigned_to_unassigned_inside"]:,}点，原灰点获得实例{full["unassigned_to_assigned_inside"]:,}点。',
 f'撤回涉及{full["retired_support_points"]:,}个表面点，最终仍有{full["retired_support_final_unknown"]:,}个灰点；按原生状态分解：{full["retired_support_final_unknown_by_native_state"]}。撤回{full["removed_frame_surface_identity_votes"]:,}票，补入{full["added_frame_surface_identity_votes"]:,}票。票按帧/表面点/实例去重，不能将累计票数当作物体完整度。','',
 f'仅种子改变{seed["strict_commit"]["final_inside_changes"]:,}点；完整追踪明显增加了修复范围，也增加了灰点和分裂。视觉上分开不等于完整修复成功。','',
 '## 真实追踪失败与实际换票','']
 for r in cross['comparisons']:
  check=decisions[r['frame']]['objects'][str(r['propagated_track_id'])]
  used=r['propagated_track_id'] in decisions[r['frame']].get('accepted_track_ids',[])
  lines.append(f'- f{r["frame"]}，人工轨迹{r["human_track_id"]} / 传播轨迹{r["propagated_track_id"]}：IoU={pct(r["IoU"])}，换票={used}，原因={check["reasons"]}。实际图片见cross_seed_review，绿色人工、粉色传播、黄色重叠。')
 roi_by_uid={r['case_uid']:r['source_choice']['ROI'] for r in data['human_selections']}
 zero=[{'ROI':roi_by_uid[a['case_uid']],'mask_id':a['native_mask_id'],'track_id':a['track_id']} for a in data['seed_audits'] if a['accepted_mapping_frames']==0]
 lines+=['',f'全程没有进入换票的选择：{zero}。C0015 mask23 / 轨迹20的有效种子表面只有14点，未放宽至少20个可见点的原门槛。所有raw mask保留。','',
 '事后GT诊断发现多个独立选择落入同一实例：'+json.dumps(ev['multiple_selected_parts_of_same_GT_instance'],ensure_ascii=False)+'。这个诊断只用于解释过分割，本轮没有据此改变预测。','',
 '## 正确性与实现变更','',
 '13项已有联合操作核验通过；全量400帧票据重算＝增量更新；准确回滚通过；未撤观察、范围外所有逐点证据/票据/最终标签保留；几何/RGB保留；28,000张传播图核对；PLY xyz/RGB/三个版本label逐字段回读相等；评估后预测哈希不变。',
 '原版证据中的map_version和association_decisions_sha256保留在来源引用中；本轮独立NPZ不复制这两项旧批次元数据。实际逐点字段和范围外票据已逐项核对，详细区别见complete.json。','',
 '本轮改动：新room2批次和精确来源校验；正确区分18条已标注与12条未标注；两对重复椅子共用身份；使用此前已修正的完整颜色表导出器；增加椅子在对方人工种子帧的真实漂移对照图与中文说明。追踪模型、原始mask、数值门槛、投票算法和评估规则未改。','',
 '额外漂移对照图仅用于展示，未改变mask、投票或地图。算法运行没有失败记录；本地输入验证时修正了来源manifest字段名称和人工裁剪边缘判断，均在启动追踪前完成。','',
 '本地网页验证器首次在写出最后核验报告时失败：漂移图片循环的record变量覆盖了bundle记录。已改为cross_record并重新通过19份PLY、408张结果图、全部网页链接和原标注哈希检查；未修改任何修复地图或指标。','',
 '服务器代码：`/home/chenkejun/CVPR/experiments/room2_top30_repair_20261008/batch_6b3f32f5dc1c`。',
 '服务器结果：`/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c`。',
 '最终实例地图：`validated/full_track/final/instance_surface.npz`。三版本完整PLY：`review/ply_models/room2_full.ply`。',
 '下一步应先按RGB和3D核对同物体的部件分组、处理同类椅子的传播漂移，并研究历史ID撤票范围；本轮没有为了提高分数自动实施这些改变。']
 with (HERE/'CHANGES_CN.md').open('w',encoding='utf-8',newline='\n') as s:s.write('\n'.join(lines)+'\n')
 summary={'status':'PASS','participating_mapping_frames':385,'mapping_frames_with_actual_vote_delta':delta_frames,'gray_delta':gray,'zero_accepted_tracks':zero,'rejection_reason_names':reasons,'strict_success':cmp['strict_success'],'maps_or_metrics_changed_by_reporting':False}
 (HERE/'reporting_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
