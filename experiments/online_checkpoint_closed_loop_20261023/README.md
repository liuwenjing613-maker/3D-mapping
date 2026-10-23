# room0 / A / 100→200 修复闭环对照

实验完成，真实闭环严格等价验收未通过：一次Gate接受/拒绝差异传到历史提交；固定相同表面的2,470项检查全部通过，两组第201帧均正确继承各自提交。详细结论、差异量化与计时见 [REPORT.md](REPORT.md)。

本实验延续检查点重建频率优化的验证，算法源于服务器分支 `online-v1-checkpoint-surface` 的 `bdf6958aafa0d3459a7f2e8fe4db5679f45eae83`。实验没有修改生产建图、ROI规则、Gate规则、SAM2或冻结partial＋E1内核。

代码目录为 `/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_checkpoint_closed_loop_20261023`；数据、日志和结果目录为 `/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023`。

## 两组真实闭环

`reference` 和 `checkpoint` 分别读取上一次同条件完整重跑的 `reference_200_A/mapping`、`checkpoint_200_A/mapping`，二者均来自同一个已修复第100帧提交。映射输出以只读输入方式引用，后续结果分别写入新目录。

每组独立执行原有候选目录、三状态Top20、前端输入准备、原始模型缓存核验、原有语义与Mask筛选、原始种子分组、SAM2、Gate、partial＋E1和修复提交。没有复制另一组的ROI、候选或Gate决定。`prepare_experiment.py` 只适配当前输出与执行目录，保留历史缓存来源；修复、提交和Gate的数值函数与原实现相同。

SAM2追踪复用通过原有完整 `fingerprint_data`：种子像素及顺序、联合分组、种子帧、完整RGB序列、模型权重、代码、配置和运行参数全部匹配。`track_queue_with_cache.py` 将可查询来源扩展到已有200/A、200/B归档，使用原封不动的复用函数核验每张标签及目标种子。所有Gate决定重新计算。

`run_frame201.py` 调用各自原有建图主函数，只将范围设为201帧、前序设为200帧，并指定本次新提交指针。它实际恢复历史像素、身份状态和TSDF，然后对原始帧1000（第201个建图帧）执行关联与融合，使用原有恢复及首帧审计。

## 固定表面控制

`fixed_reference` 和 `fixed_checkpoint` 是额外控制实验，不替代真实闭环结果。两边使用此前各自原实现生成、已精确核验相同的固定表面投票结果，仍保留各自真实流程选出的种子。重新执行种子投影、Gate、partial＋E1和提交，比较原始数组，不进行点排序、合并或候选统一。

## 证据和比较

- `compare_roi.py` 比较三状态Top20排序、空间成员多重集合、历史视角及裁剪框。临时点编号生成的UID映射到精确坐标多重集合；不使用浮点容差。
- `compare_tracking.py` 比较种子来源与像素、完整追踪指纹、每帧SAM2标签、Gate决定和接受Mask。输出目录与ROI临时编号仅作明确的元数据归一化，原始差异仍保留。
- `compare_repair.py` 使用一个固定的全局点置换检查真实结果的投票、状态、最终标签、修改范围、事务和历史像素；固定表面对照额外逐数组核验所有E1、partial和提交NPZ。
- `diagnose_anchor_order.py` 只诊断原Gate按点编号等距抽样最多2048个点的顺序依赖，不修改抽样。
- `final_audit.py` 检查第201帧确实消费新提交、历史身份保留、两组关联结果以及因果输入访问记录。

脚本保存本次实际执行的固定路径。原始运行目录和控制目录必须保持独立；重新实验需明确设置一组新路径，不能覆盖现有输出。全部大文件留在数据盘，代码分支仅保存脚本、报告和小型JSON证据。本实验不做GT评分，不改变main的R2评价协议。
