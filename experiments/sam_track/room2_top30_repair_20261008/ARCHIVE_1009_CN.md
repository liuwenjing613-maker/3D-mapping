# room2 最新前30批次归档说明

保留本目录实际运行代码字节、原README/CHANGES、14组配置、身份核验及原始人工JSON。`results_1009/` 包含实际运行的种子、输入/评估freeze、完整指标、修复与导出核验。`proofs/` 是真实3D截图。本轮完整追踪修复：AP50=30.0330%、F1=54.7368%、PQ=37.5218%，未分配184351；原始P1-A1分别28.4378%、53.0120%、34.8715%，未分配150416。数字来自同一pending/debug统一v3，未声称正式benchmark。

此批次没有叠加旧5例结果。全部数据、传播PNG、NPZ与完整PLY保留 `/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c`。根因审计 `root_cause_audit.py`、`root_cause_vote_provenance.py`、残留票解释和跨种子身份审计均归档。连续性试验和含3D截图的总体报告见 `../continuous_tracking_ablation_20261008/README_CN.md`。不调整原门槛、不改v3、不推广指标下降的截断策略。
