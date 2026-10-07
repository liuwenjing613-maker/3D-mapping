# room0 人工选择的离线联合修复

本目录保存实际运行的代码、12份追踪配置、原始人工JSON、GT无关身份决定及来源哈希。数据、权重、日志和PLY在服务器 `/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d`，没有提交到Git。

运行依赖沿用已提交的pilot10修复工具、SAM2.1仓库与现有服务器环境。代码根为 `/home/chenkejun/CVPR/experiments/room0_repair_20261007/batch_386cdcff710d`。

实际顺序：prepare_local.py核对本地选择 → prepare_server.py核对并冻结原始mask、深度和表面支持 → tracking_queue.py在GPU0–2排队完成12组各2000帧独立前后传播 → cross_seed_diagnostics.py和alias_review.py冻结11→8 → run_batch.py执行全局关联和仅种子/完整追踪联合换票 → evaluate_batch.py保持统一v3进行预测冻结后的评估 → export_review_v2.py导出真实PLY与视角图 → 本地build_review.py构建中文结果页。

首轮pipeline.py在导出颜色表时失败，其源文件与export_review.py保留。export_review_v2.py补齐未在最终地图获胜的轨迹ID颜色；export_recovery.py确认修复地图、指标和冻结代码哈希保持。最终展示使用v2导出器，原始失败记录公开。

run_batch.py对已经完成的结果会核对哈希，避免重复换票。本轮额外重跑验证表明全部NPZ不变。pipeline/recovery调度器使用独立启动记录与独占日志，防止重复启动；复现实验应使用新的批次目录并同步修改路径，不能覆盖原批次。

结果为统一v3的pending/debug协议。目标IoU、F1和PQ提高，但目标结构错误计数增加，因此严格成功仍为false；完整数字、灰点、用帧数和局限见CHANGES_CN.md。原人工mask、数字门槛、投票单位、几何/RGB和v3参数保留，GT不用于修复。
