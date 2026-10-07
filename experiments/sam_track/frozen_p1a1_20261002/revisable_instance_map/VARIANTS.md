# room0 三组无修复建图版本

三组结果共用 `src/revisable_instance_map/` 和 `tools/`，因此保存在同一个分支。固定输入为 Replica room0 的第 0、5、…、1995 帧（400 帧），共用同一份 1 cm TSDF 表面及冻结的 Replica-CA-v1 评测协议。GT 仅供建图完成后的离线评估。

| 版本 | 输入配置 | 关联参数 | CA-AP50 | CA-PQ | TP / FP / FN |
| --- | --- | --- | ---: | ---: | --- |
| 原始 CropFormer | `configs/replica_room0_stride5.json` | 默认：每个已有实例同帧最多接收一个观测 | 0.2068 | 0.3234 | 34 / 49 / 34 |
| OVI 几何细化 | `configs/replica_room0_stride5_ovimap_refined.json` | 同上 | 0.0629 | 0.1676 | 27 / 143 / 41 |
| OVI 几何细化，同帧多观测 | 同上 | `--allow-multiple-observations-per-instance-per-frame` | 0.1266 | 0.2620 | 28 / 64 / 40 |

三者的关联入口均为 `tools/run_baseline_association.py`。前端切换由 `--config` 指定的 mask 来源决定；第三组只增加表中命令行开关。后续的 `tools/materialize_instance_map.py` 和 `tools/adapt_instance_map_replica_ca.py` 共用，分别生成实例表面和类别无关评测输入。

细化掩码由 `tools/build_ovimap_depth_masks.py`、`tools/build_ovimap_refined_masks.py` 生成。细化前端与官方 OVI-MAP 对齐；三组使用的跨帧关联均为本项目无修复基线，不是 OVI-MAP 的完整后端。完整输入、算法、审计和结果见 `docs/no_repair_baseline.md` 与 `docs/ovimap_depth_refinement.md`。

权重、数据集、掩码、TSDF、日志和评测产物未提交；它们保留在服务器数据目录。上述分数是 room0 单场景的自定义类别无关指标，不是 ScanNet 官方 AP。
