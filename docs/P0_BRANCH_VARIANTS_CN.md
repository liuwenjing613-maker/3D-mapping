# baseline P0：无扩散消融

本分支保存三状态离线扩散引入前的代码，基于提交 `c27ceb9b47b532bf60afec5bbbd04efd6055ed93`。对应已评估的原始 P0 发布结果，而非旧版 local fill 对照。

消融输入为服务器 `/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/{scene}/surface_p0/instance_surface.npz`；它与上一轮评估的 `baseline` 变体相同。不要运行任何离线补洞或扩散工具来产生这一消融结果。本分支不包含新加入的 `offline_surface_assignment.py`、`repair_p0_surface.py` 及其测试。

对照分支：`main` = **baseline + 部分未分配 TSDF 点扩散处理**。

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
