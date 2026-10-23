"""Write the final verdict from completed evidence, preserving actual branch differences."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(name):return json.loads((OUT/(name+'.json')).read_text())

def main():
    fixed=read('fixed_repair_comparison');tracking=read('fixed_tracking_comparison')
    audit=read('final_audit');impact=read('downstream_difference_diagnosis');timing=read('timing_summary')
    assert read('validation_tail_status')['status']=='COMPLETE'
    assert fixed['status']==tracking['status']=='PASS'
    assert read('repair_comparison')['status']==read('tracking_comparison')['status']=='DIFFERENT'
    assert audit['each_mode_inherits_its_own_new_commit'] and audit['frame201_new_association_decisions_exact']
    assert not any(x['blocked_nonprobe_opens'] for x in audit['causal_access_audit'])
    assert read('current_source_audit')['status']=='PASS'
    assert impact['final_map']['changed_points']==2 and impact['history_pixels']['unique_frame_pixel_positions']==20409
    a,b=timing['rows'];ma,mb=a['mapping_timing'],b['mapping_timing']
    summary=dict(execution_status='COMPLETE',strict_actual_closed_loop_equivalence='NOT_PASS',
        ROI_Top20='PASS',seeds='PASS',SAM2_dense_labels='PASS',actual_Gate='DIFFERENT',
        fixed_surface_Gate='PASS',fixed_surface_repair_and_commit='PASS',fixed_surface_checks=len(fixed['checks']),
        actual_final_map_different_points=2,actual_history_different_frames=8,
        actual_history_unique_frame_pixel_positions=20409,actual_commit_equivalence='DIFFERENT',
        each_mode_frame201_inheritance='PASS',frame201_new_observation_decisions='PASS',frame201_new_observations=36,
        no_GT_quality_conclusion=True,no_candidate_unification=True,no_production_sorting_or_Gate_change=True,
        mapping_main_speedup=ma['main_mapping_seconds']/mb['main_mapping_seconds'],mapping_total_speedup=ma['total_seconds']/mb['total_seconds'])
    (OUT/'acceptance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=f'''# room0 / A / 100→200→201 闭环验证报告

实验已完成，**真实闭环的严格等价验收未通过**。ROI、种子和SAM2输出一致，但原有Gate按临时点编号抽样，对表面点排列顺序敏感，产生一次接受/拒绝变化，并传到历史身份提交。固定相同表面的控制实验全部通过，不能用它替代两组真实输出中的差异。

两组读取此前同条件完整重跑的 `reference_200_A/mapping`、`checkpoint_200_A/mapping`，起点为同一第100帧已修复提交；本次独立执行第200帧候选、追踪、Gate、partial＋E1、提交，并真正处理第201个建图帧（原始帧1000）。算法代码基于 `bdf6958aafa0d3459a7f2e8fe4db5679f45eae83`，生产建图、Gate和修复数值逻辑均未新增修改。

| 验收阶段 | 实测结果 | 判断 |
|---|---|---|
| 第200帧ROI | 三状态各Top20，共60条排名；35个独立区域的物理成员、历史视角及裁剪框一致 | 通过 |
| 候选及种子 | 28个有效种子的像素、来源身份、顺序一致；两组均无完全重复种子 | 通过 |
| SAM2与Gate | 21个联合追踪任务、20,916张稠密标签一致；5,600个对象×历史帧最终判定中1个接受/拒绝不同 | SAM2通过，真实Gate不通过 |
| partial＋E1 | 固定相同表面时精确一致；两组真实最终标签有2个非重复位置不同 | 固定表面对照通过，真实结果不完全一致 |
| 修复提交 | 各自提交核验通过，但8个历史帧的身份层、关联支持与身份票数不同 | 提交有效，跨组一致性不通过 |
| 第201帧 | 都正确恢复自己的新提交，36个新观察的关联决定一致，历史状态原样保留 | 继承通过；两组历史状态仍有差异 |

**差异原因。** `continuation300/track_batch.py` 先对临时表面点编号排序，再用 `np.linspace` 抽最多2048点。两组28个种子的完整空间支持集合相同，但其中13个抽样子集不同。原始帧445（第90个建图帧）、track 13均有1101个可见锚点，原版覆盖880个，即79.9273%；优化版覆盖882个，即80.1090%。原有阈值为80%，所以原版拒绝、优化版接受，接受Mask相差20,214像素。原始帧640的track 15仅拒绝原因不同，两组都拒绝。

这不是SAM2推理差异，也不能仅归结为重复坐标顶点。相同空间点的存储顺序会改变Gate实际检查的位置。额外的原版对原版复跑也出现13个锚点子集变化，但该次没有跨过接受阈值；其提交关联检查点与历史归档相同。这表明顺序敏感性已存在于原流程，不能据此抹掉本次原版与优化版的真实差异。

**传播范围。** 最终地图共有1,529,866点，2点标签由原版的未分配（−1）变为优化版的ID 7，比例约0.0001307%；两组均有80个正ID实例。E1 owner在252个非重复位置不同，历史身份层涉及8帧、10个图层文件、40,268个“图层×像素”条目；按同帧位置去重后为20,409像素。按实际体素坐标与持久ID对齐，提交身份计数在314个体素的ID 7上各相差1票。提交关联观察条目为9,744与9,745，next_instance_id同为423。最终地图变化很小，但历史状态差异确实存在；本实验没有使用GT评分，不能断言哪组修复更正确或证明质量完全无损。

**固定表面对照。** 两组仍使用各自真实流程选出的种子，分别读取此前各自原实现生成且已核验相同的固定表面证据，重新执行投影、Gate、partial＋E1和提交。Gate的决定、理由、指标与接受Mask全部精确一致；{len(fixed['checks']):,}项数组、键集合及提交检查全部通过，浮点容差为0，覆盖原始NPZ数组、三种partial变体、E1区域/owner、逐帧严格票与事务、历史像素、关联检查点、身份票数和来源像素。

首次比较仅剩3,636条 `sampling_source_mask` 的绝对路径不同：它们分别指向两组建图输入目录。核验这些源Mask的文件哈希一致后，比较器显式归一化该输入目录前缀；原始记录和第一次比较报告均保留。此处理只用于结果比较，没有修改候选、表面、Gate或修复结果，也没有增加生产点排序。

**第201帧继承。** 两组分别验证新提交检查点SHA，恢复map_version 203后才对原始帧1000关联与融合，完成后均为204。历史源像素和持久ID恢复核验通过，已有提交条目保持原样。两组36个新观察的关联决定一致，但它们继承的历史检查点不同，所以后续总状态不能标为跨组精确一致。六组运行输入访问审计均没有非探针越界阻断，原第100帧提交及两组建图输入保持不变。

| 时间范围（秒） | 原版 | 优化版 |
|---|---:|---:|
| 101～200主建图，前次独立实测 | {ma['main_mapping_seconds']:.3f} | {mb['main_mapping_seconds']:.3f} |
| 同次完整建图进程，含恢复与导出 | {ma['total_seconds']:.3f} | {mb['total_seconds']:.3f} |
| 本次候选生产阶段累计 | {a['candidate_production_seconds_including_precomputed_stages']:.3f} | {b['candidate_production_seconds_including_precomputed_stages']:.3f} |
| 本次追踪编排＋Gate | {a['track_orchestration_seconds']:.3f} | {b['track_orchestration_seconds']:.3f} |
| 本次partial＋E1 | {a['repair_seconds']:.3f} | {b['repair_seconds']:.3f} |
| 本次提交 | {a['commit_seconds']:.3f} | {b['commit_seconds']:.3f} |

前次主建图加速{summary['mapping_main_speedup']:.2f}倍、完整建图进程加速{summary['mapping_total_speedup']:.2f}倍。后半段均使用同等严格缓存条件：每组599个CropFormer与599个SAM前端任务命中缓存，21/21个SAM2追踪任务完整指纹匹配，未重跑SAM2视频推理；所有Gate重新计算。候选累计时间包含提前独立完成的阶段，排除QA、可视化和GPU等待。后半段时间受缓存及并行诊断负载影响，仅作执行记录；不把它拼成冷启动端到端加速结论。

本次支持“减少全量表面重建次数显著提速，固定相同输入时修复内核结果一致”。它**不支持**“现有真实在线闭环已严格等价”。后续若要通过该项验收，需要另行确定如何消除Gate按临时点编号抽样的顺序依赖，再复验；本次未实施这个额外算法处理。

完整运行结果：`/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023`。复现脚本位于同名实验代码目录，小型证据及哈希目录见 [evidence/manifest.json](evidence/manifest.json)。核心证据为 [验收结论](evidence/acceptance_summary.json)、[真实结果比较](evidence/repair_comparison.json)、[固定表面对照](evidence/fixed_repair_comparison.json)、[第201帧审计](evidence/final_audit.json) 和 [差异量化](evidence/downstream_difference_diagnosis.json)。未使用旧GT评价器，也未改动main的R2协议。
'''
    (HERE/'REPORT.md').write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
