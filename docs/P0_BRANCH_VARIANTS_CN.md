# main：baseline + 部分未分配 TSDF 点扩散处理

本分支以无扩散 `baseline-P0` 为消融对照，加入已验证的离线三状态补全 v2（原扩散提交 `8149df420bc7d4a3caab16de24eff72759b41f55`）。

扩散只处理仍未分配的 U、T、CONFLICT 原生 TSDF 表面点。先补保守封闭孔洞，再通过有总路径预算和实例竞争约束的多源 Dijkstra 扩散；原已发布身份全部保留为竞争源，可靠身份才可发起赋值。T/CONFLICT 还需符合候选证据。原 CONFIRMED 标签、原始观测证据、几何、颜色和点序不变；推断标签不升级为新的 CONFIRMED 证据。

扩散实现：`revisable_instance_map/src/revisable_instance_map/offline_surface_assignment.py`；入口：`revisable_instance_map/tools/repair_p0_surface.py`。该步骤为显式离线后处理，不会自动改写原始 P0 建图输出。使用入口的 `--source-dir`、`--geometry`、`--output-dir` 参数；最终方法输出是 `holes_geodesic/instance_surface.npz`，其他变体仅作消融对照。

已评估结果保存于服务器 `/data/chenkejun/CVPR/revisable_instance_map/offline_three_state_p0_20260930_v2`，新增归属 158,249 个点。16 项行为测试和八场景独立路径核验已通过；与已评估版本的三份代码文件内容一致。此方法不使用 GT 产生赋值。

## 实验与评估口径

- 八场景：room0、room1、room2、office0、office1、office2、office3、office4。
- 输入 400 帧，帧序列 0:2000:5；共享 TSDF 几何、点序和已发布实例标签保持不变。
- 使用统一 Replica-CA-v3：主映射 0.01 m，诊断映射 0.02 m，GT 最小 100 顶点，显著交集 10 顶点 / GT 占比 0.05。
- 配置为 `unified_eval/configs/replica_ca_v3.pending.json`，这些结果标记为 `DEBUG_ONLY / NON_OFFICIAL`。
- GT 辅助填充是独立诊断实验，不属于 baseline 或 main 的方法，不纳入这两个分支。

| 八场景 pooled | AP | AP50 | PQ | F1 |
|---|---:|---:|---:|---:|
| baseline P0，无本次扩散 | 3.3788% | 9.0519% | 21.9185% | 29.6651% |
| baseline + 部分未分配 TSDF 点扩散处理 | 3.6680% | 10.4730% | 23.3997% | 32.1656% |
