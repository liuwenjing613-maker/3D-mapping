# 固定表面诊断与严格局部提交

本次仅实现引用对话要求当前先完成的两项：固定几何诊断、最终标签提交范围约束。
实验仍为固定 TSDF 表面的离线回溯修复，使用已完成的 0–1999 帧追踪。

## 行为与边界

- 有门槛、无门槛原结果均作为只读候选保留。
- 允许集合直接使用每组原试验的 `intervention_surface_points.npy`，即被撤回旧观察支持与新 mask 支持的并集。没有新增空间外扩、GT 选区或自动 ROI。
- 候选后处理结果在允许集合内逐点保留；集合外恢复基线最终标签，输出单独的 `gated_strict`、`ungated_strict`。这是提交范围约束；候选计算仍使用既有全图上下文。
- 全部范围外原始证据字段和完整投票账本重新核对。原账本已通过的 400 帧独立回放、精确回滚验证继续适用，因为本步骤不改任何票或原始状态。
- 基线证据文件另有整图 `map_version`、`association_decisions_sha256` 元数据，既有试验文件未复制这两个字段。它们单独记录，全部逐点证据字段仍必须完整存在并逐项校验；缺少真实证据字段会拒绝提交。
- 限制前的越界标签逐点记录到 `blocked_outside_changes.npz`。

## 固定表面诊断

`fixed_surface_mapping.npz` 对全 TSDF 顶点和完整 GT 参考几何建立一次对应关系。
未分配顶点仍是最近邻候选。所有版本复用同一缓存；几何、点顺序或距离设置改变会拒绝复用。
距离沿用 1 cm，判断为严格小于；该诊断不修改统一 v3。

同时报告完整 GT 参考点数、几何可对应点数、期望身份覆盖、其他目标身份残留、其他实例身份、未分配量、纯度、固定对应 IoU、标签转移。
期望身份为 ID52→GT4001、ID355→GT4002，来自现有事后目标记录，只用于诊断。基线同时报告其最佳旧预测的诊断 IoU，避免把原共享 ID 的覆盖误写成完全没有预测。
GT 在两个严格局部预测冻结后读取；GT 不参与允许集合、投票或标签提交。

主评分继续使用原统一 v3 的代码、参数与协议。现有协议 `pending`、`frozen=false`，本次仍是调试结果。
正式协议校准/冻结需要独立开发场景，未对当前案例调整阈值。

## 未改动的内容

TSDF 几何、RGB、人工种子、SAM2 权重与追踪、帧筛选策略、实例 ID、投票账本、三状态重算、至少2票/67%确认条件、原邻域传播参数、原 v3 结果均保持不变。
自动种子、自动 ROI 排序、观察级“未知保留”的替换策略和在线关联状态更新属于后续阶段，本次没有引入这些机制。

## 运行

服务器运行目录为 `/home/chenkejun/CVPR/experiments/pilot10_repair_20261007/strict_local_repair_20261007`，结果在 `/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/strict_local_repair_20261007/room2_mask14_mask24`。

```sh
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python -m unittest -v test_fixed_surface_repair
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python -u run_strict_local_trial.py
```

`source_freeze.json` 逐项声明变更与不变项，`prediction_freeze.json` 记录评估前预测哈希。
`commit_validation.json` 和最终 `complete.json` 保存范围校验、主评分与非目标退化检查。
再次运行已完成任务只校验来源、已提交预测和映射缓存，并保留已有结果。

依据：引用对话 `6ac603af-0364-83ec-a695-c0f78f82d34b` 最新解释明确建议现在处理固定表面评价与修复范围越界，在线化留待离线机制验证后。
