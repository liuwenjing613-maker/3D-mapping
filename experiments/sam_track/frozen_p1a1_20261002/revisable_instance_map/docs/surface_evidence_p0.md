# P0：TSDF 表面多视角实例证据（Replica room0）

## 固定输入与方法

此次只替换实例地图的 materialization。逐帧 CropFormer/深度处理、11,729 条观测的关联决定和 400 帧共享 TSDF 保持不变；输入 SHA-256 已与旧版 `map_ovimap_parent_regrouped_multi_vf_dedup/materialization_report.json` 对齐。GT 仅由评估器读取，建图程序不读取 GT。

对整帧 mask 和有效深度按固定 2 像素网格采样，反投影至世界坐标，用 KDTree 查找 1,633,034 个 TSDF 表面点中 1.5 cm 内的最近点。每帧的 `(表面点, persistent instance ID)` 至多贡献一票。先保存与 persistent ID 无关的 `(表面点, 局部 mask ID)` 逐帧记录，以便日后更改关联后重算。跨帧汇总每个点的 top1/top2、各自票数、总票数、胜者比例及差距。

内部状态：`CONFIRMED` 要求胜者至少 2 票且比例至少 0.67；有第二 ID 且胜者比例不足 0.67 为 `CONFLICT`；其余有证据点为 `TENTATIVE`；无证据为 `UNOBSERVED`。主地图 `instance_surface.npz` 只发布 `CONFIRMED`；`TENTATIVE` 保留在证据和状态 PLY 中，另提供包含它的诊断投影。`CONFLICT` 始终不硬赋值。此阶段没有进行表面邻域补全，也没有修改关联器。

## 运行与验证

程序：`tools/materialize_surface_evidence.py`，实现函数：`src/revisable_instance_map/surface_evidence.py`，单元测试：`tests/test_surface_evidence.py`。使用 `configs/replica_room0_stride5_ovimap_parent_regrouped.json`、`observations_ovimap_parent_regrouped/observations.jsonl`、`association_ovimap_parent_regrouped_multi/associations.jsonl` 和 `geometry_full400_1cm_capacity100k/surface.ply`。完整结果位于 `/data/chenkejun/CVPR/revisable_instance_map/surface_p0_room0_stride2_15mm_final/`，固定统一评估结果位于同级的 `evaluation_surface_p0_room0_stride2_15mm_final/metrics/`。

完整 400 帧输入有 78,772,444 个有效采样像素，78,740,234 个通过 1.5 cm 最近点门槛，去重后为 65,231,346 次逐帧实例表面投票。全场景表面状态：高置信 1,436,742，待确认 16,925，冲突 72,216，无观测 107,151。主地图覆盖 TSDF 表面点的 87.98%。构图约 85 秒，进程峰值 RSS 约 1.17 GiB。

`python -m unittest discover -s tests -v`：7/7 通过。独立产物检查 `qa_report.json`：PASS；逐点投票总和、状态、主地图标签、逐帧文件数和 PLY 点数均一致；PLY 与 TSDF 坐标最大误差 0；重跑得到完全相同的 `surface_evidence.npz` SHA-256。统一评估协议为 `/home/chenkejun/CVPR/unified_eval/configs/replica_ca_v1.json`，同一 GT、同一适配器与 room0 场景。

| 版本 | CA-AP50 | CA-PQ | TP / FP / FN |
|---|---:|---:|---:|
| 旧版 1 cm 支持票 + 最近邻传播 | 0.2999 | 0.3981 | 36 / 27 / 32 |
| P0，2 像素步长，只发布高置信 | **0.4300** | **0.4768** | **38 / 11 / 30** |
| P0，包含待确认点的诊断投影 | 0.4214 | 0.4748 | 38 / 12 / 30 |

主输出 PLY：`instance_surface_colored.ply`（实例着色，未发布的点灰色）；诊断 PLY：`surface_evidence_states.ply`（高置信为实例色、待确认为白色、冲突为红色、无观测为灰色）。两者均有 1,633,034 点。完整票数在 `surface_evidence.npz` 与 `surface_instance_frame_votes.npz`；逐帧原始区域支持在 `frame_region_support/`，索引及哈希在 `frame_region_support_manifest.jsonl`。

## 解释与边界

AP50 增加 0.1301、PQ 增加 0.0787，且 FP 从 27 降至 11。在固定前端、关联和 TSDF 的条件下，这支持旧版 materialization 的稀疏体素采样及传播是重要损失来源；不能据此宣称前端或关联已经没有问题。仍有 72,216 个冲突点，需要后续分析其边界混合与 persistent ID 碎片化来源。最近点查找尚无显式可见性或遮挡检查，1.5 cm 是预设的起始门槛；当前结论只在 room0 验证。P1 的表面感知补全和 P2 的关联修改均未执行。
