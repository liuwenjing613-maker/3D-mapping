# 检查点表面重建优化

基于 `online-v1` 提交 `20cb6a164d9aba465e020bc026e086357b88ceea`，开发分支 `online-v1-checkpoint-surface`。服务器工作区：`/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023`。所有新结果保存在 `/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023`，原实验及冻结算法归档保持不变。

## 改动范围

修改 `experiments/online_room0_repair50_to100_20261022/continuation300/map_continue.py` 和 `map_support.py`，适用于现有 100→200、200→300 帧续建入口。50、100 帧的历史入口没有改动。

- 默认 `CVPR_SURFACE_POLICY=checkpoint`：每帧仍加载输入、关联持久 ID、更新 lookup、融合 TSDF、保存历史观察，只在目标检查点提取表面并执行完整历史投票。
- `CVPR_SURFACE_POLICY=every_frame` 保留逐帧重建执行方式。
- `strict_surface()`、投票阈值、歧义弃权、表面后处理、历史 journal 恢复及关联数值实现不变。
- 未重建帧的表面统计明确写为 `null`，并记录 `surface_rebuilt=false`；对应计时为零。检查点输出保持完整。
- 必须用 `CVPR_MAPPING_RUN_ROOT` 指定独立输出目录，限制在 `/data/chenkejun/CVPR` 的子目录下，禁止指向原实验归档。已完成目录禁止重新初始化，错误重跑不会写入失败标记覆盖完成状态。

## 单次续建

必须使用新的输出目录；以下两个命令依次执行。`CVPR_CHECKPOINT` 取 200 或 300，`CVPR_TRACK_GROUP` 取 A 或 B，恢复对应已有修复提交。目标帧计数为建图帧数，原始帧仍使用 stride=5。

```bash
cd /home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_room0_repair50_to100_20261022/continuation300
export CVPR_CHECKPOINT=200 CVPR_TRACK_GROUP=A CVPR_SURFACE_POLICY=checkpoint
export CVPR_MAPPING_RUN_ROOT=/data/chenkejun/CVPR/results/YOUR_NEW_RUN
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python
"$PY" -B map_support.py
CVPR_ONLINE_GUARD="$CVPR_MAPPING_RUN_ROOT/mapping/input_guard.json" PYTHONPATH="$PWD/map_guard" "$PY" -B -u map_continue.py
```

原归档的总调度器仍包含历史绝对路径；验证使用以上续建入口，不通过历史调度器写入旧实验目录。运行停止在本检查点修复之前；此次验证不包含重新执行候选筛选、SAM2、E1 和修复提交。

## 验证方法

`run_validation.py` 串行执行优化版 200/A、未修改原版 200/A，以及优化版 300/A、200/B、300/B。原版文件直接从基准 Git 提交提取，算法源码逐字节保留；仅共享允许独立输出的路径适配器。原版与优化版使用相同前序修复提交、输入、线程数和因果输入保护。A/B、200/300 的恢复均来自各自既有提交，未将 A 的身份状态用于 B。

`compare_outputs.py` 对关联 JSON、关联检查点所有数组、身份票数、TSDF 所有体素属性、表面坐标/法向量/颜色、完整证据数组、原生与最终实例标签、逐帧投影支持以及表面图进行精确比较，不设置浮点容差。体素按 block key 对齐，表面使用同一个点置换对齐所有关联数组和图边。原始点顺序差异单独记录。

`validate_run_contracts.py` 验证真实输出的关联版本链、已提交状态保留、中间帧日志、计时求和、检查点状态统计及文件访问审计。

`replay_common_surface.py` 是额外的验证步骤：把原版与优化版各自的完整历史输入和身份决定，分别交给各自未经修改的 `strict_surface()`，使用同一份保存的表面坐标、精确 RGB、法向量和点顺序，逐数组比较所有票数、状态、标签、逐帧支持和图。它不改变在线算法，不排序或删除点，也不作为在线耗时样本。

独立 Open3D 提取存在重合点顺序问题。200/A 的独立运行对照中有少量重合点接收投票的位置发生互换，不能把这种独立运行称为所有中间数组完全一致。额外的确定性排序未获批准、未实施。完整结论和原始检查结果见验证报告及 JSON 证据。

验证脚本使用本次独立输出目录，不能直接重复覆盖已有结果。复现单次续建请使用上面的入口并指定新的输出目录。最终增加的输出目录保护通过6项实际测试；数值函数和逐帧主函数与计时版本的 AST 完全相同，实际执行过的源码另存于 `tested_source/`。

本次没有使用 GT 评分，不改变 main 的 R2 评价协议，也不将最终 400 帧 GT 范围用于在线前缀。
