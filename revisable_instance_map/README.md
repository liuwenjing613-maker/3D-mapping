# Revisable Instance Map

CVPR 项目的统一实现目录。当前目标是从空图跑通**无修复**的在线 RGB-D 实例建图；即时与历史修订留待基础建图验收后接入。

## 目录约定

- 当前开发只在服务器上进行；不向本地同步新增代码或结果。
- 服务器 `/home/chenkejun/CVPR/revisable_instance_map/`：代码、配置和文档。
- 服务器 `/data/chenkejun/CVPR/revisable_instance_map/`：本项目产生的数据缓存、权重引用清单、日志和结果。

已有数据集、模型和评估器通过明确路径引用；不复制实体文件。所有实验保存输入帧清单、配置与代码版本。基础建图推理不使用 GT，也不使用尚未到达的帧。

## 第一阶段验收目标

在固定输入帧清单上，从空地图依次读取 RGB-D、位姿和原始 mask，输出可评估的实例地图与逐帧快照。此阶段不执行 mask 或实例的历史修复。

首个开发输入为 Replica room0 的 400 帧协议，详见 `docs/input_protocol.md`、`configs/replica_room0_stride5.json` 和 `reports/input_audit_room0_stride5.json`。

读帧与 RGB-D 投影已单独验证，记录见 `docs/projection_step.md`。

共享 TSDF 的 10 帧空图烟测已通过，见 `docs/geometry_step.md`；此时仍无实例关联。

400 帧共享 TSDF 容量验证见 `docs/geometry_full400_step.md`。逐帧原始实例观测已建表，见 `docs/observation_step.md`。无修复跨帧实例关联、共享 TSDF 上的实例表面及 room0 离线评估已经跑通，见 `docs/no_repair_baseline.md`。本阶段仍无 mask 或历史身份修订。

OVI-MAP 的逐帧 CropFormer + 深度几何细化已按官方实现接入并完成 400 帧对照，见 `docs/ovimap_depth_refinement.md`；前端结果与后续提示分割修复分开保存。

最新无修复来源归组架构、消融与测试：`docs/parent_proposal_regrouped_v1.md`。
