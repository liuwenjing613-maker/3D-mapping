# SAM-track：人工种子追踪与 TSDF 实例换票

本目录归档 2026-10-07 实际运行的代码。原仓库的基础版本为
`c27ceb9b47b532bf60afec5bbbd04efd6055ed93`（`analyse_error_v1`）。

## 代码位置

- `pilot10_repair_20261007/`：人工选择、SAM2.1 短程及完整追踪、观察替换、账本重算、v3 评估、PLY 导出和网页构建。
- `frozen_p1a1_20261002/`：本次试验实际导入的建图与统一 v3 代码快照，避免仓库其他版本影响复现。
- `viewer_vendor/`：已验证的 Three.js r180 离线模块、许可证和来源记录。
- `source_manifest.json`：归档来源和逐文件 SHA256。
- `docs/operation_and_evaluation_audit.md`：操作范围、已观察到的改善、剩余问题和评估限制。

代码保留本次实际运行版本及其服务器路径，不将其包装成通用追踪库。
跟踪权重、RGB-D、人工 seed PNG、mask 序列、NPZ、PLY、日志和评分结果位于
`/data/chenkejun/CVPR`；此处提供代码、配置和说明。

## 当前服务器布局

代码：`/home/chenkejun/CVPR/experiments/pilot10_repair_20261007`

冻结依赖：`/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot`

基线：`/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002`

输入：`/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main`

联合试验结果：
`/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/joint_mask_replacement_20261007/room2_mask14_mask24`

核心依赖为 SAM2.1、PyTorch、NumPy、SciPy、Pillow、Open3D；冻结建图环境和
SAM2 环境的解释器路径保留在 launch 脚本中。SAM2 权重 SHA256 为
`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`。

## 运行链路

以下命令使用已有服务器布局及其人工种子、基线和权重。launcher 会启动后台进程，
下一阶段必须等待上一阶段的状态记录为 `PASS`。若要开展新的参数试验，先设置独立
输出目录并更新源代码和策略冻结记录；已有结果是带源代码校验的历史试验缓存。

```bash
bash launch_joint_chairs_tracking.sh
bash progress_joint_trial.sh
# 等待 tracking/status.json 为 PASS 后：
bash launch_joint_mask_trial.sh
bash progress_joint_trial.sh
# 等待顶层 status.json 为 PASS 后：
bash export_joint_trial_review.sh
bash diagnose_joint_geometry.sh
bash audit_split_evaluation.sh
```

主程序：

1. `track_joint_chairs.py`：mask14、24 在 2000 个原始帧中独立前向、反向传播并联合分配。
2. `run_joint_mask_trial.py`：固定可靠性规则后，比较局部像素撤票和整张旧观察撤票；完整重算 400 个建图帧，验证增量账本及精确回滚。
3. `joint_vote_ops.py`：按 `(帧, TSDF 表面点, 持久实例 ID)` 去重，保留未撤回观察的共享表面支持。
4. `evaluate_pilot_v3.py`：使用冻结的同一 v3 调试配置进行事后评分，GT 不参与追踪、筛帧或修复。
5. `export_joint_trial_review.py`：导出三个标签版本共用的原密度 PLY 和固定视角投影。
6. `build_joint_trial_review.py`：本地 Windows 审阅页构建器，依赖已下载的 bundle、历史 viewer 资源和微软雅黑字体；bundle SHA256 固定为本次导出的实际文件。

对既有短程/8目标完整追踪的代码也保留，以便核对本次联合试验的对照历史。
发布辅助脚本 `finalize_joint_trial_review.py` 记录已经完成的人工浏览器检查并准备
交付文件；它自身不驱动浏览器，也不是自动 UI 测试。

## 结论边界

已观察到两个实例的局部归属分离和纯度提升，但 mask24 仍有旧 ID52 残留与覆盖不足。
“局部分离有效”与“完整实例验收通过”是不同结论。当前评分为
`DEBUG_ONLY / NON_OFFICIAL`，不是正式 benchmark 结果；详见审计说明。

## 2026-10-07 增量实现

固定表面诊断和严格局部提交已经完成，原149份源文件快照保持不变。

- [变更、结果与局限](docs/strict_local_changes_20261007.md)
- [新入口与接口说明](strict_local_repair_20261007/README.md)
- 无门槛对照驱动与配置已补入pilot10_repair_20261007目录。

## 2026-10-07：逐对象离线修复入口

原mask只改关联、逐对象接收可靠证据、未知部分保留及配置化入口已实现。SAM2仍为独立双向完整历史追踪。本轮结果尚未实现完整修复，不能视为正式benchmark。旧入口仅保留复现对照。

[全部改动和实际效果](docs/objectwise_changes_20261007.md) · [新入口与配置](objectwise_repair_20261007/README.md)


## 2026-10-09 当前room0/room2方法与实际效果

新分支 `repair-room0/2-track-1009` 基于SAM-track：
- [总报告、指标、3D截图、连续性截断及局限](continuous_tracking_ablation_20261008/README_CN.md)
- [最新room2前30人工标注批次](room2_top30_repair_20261008/ARCHIVE_1009_CN.md)
- [原始帧顺序mask检查页面源代码](chronological_playback_20261008/README_CN.md)
- [本次来源与逐文件SHA清单](publication_20261009/source_manifest.json)

room0既有代码保持；results_1009补齐效果收据。gap0/gap1指标下降，未改默认追踪或原门槛。原始地图和实验输出保留，当前v3仍为pending/debug离线对比。大体积数据/PLY留在/data，分支记录可核验路径与SHA。
