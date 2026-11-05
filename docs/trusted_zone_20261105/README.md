# trusted-zone：可信区域在线策略实验

本分支从最新 online-v2 的 9cb5aa23 创建，归档 2026-11-04 的完整在线可信区域实验、2026-11-05 的同版 DBSCAN 对照和效果分析。它是可追溯的服务器实验快照，不是已经全面优于 v2-B 的正式默认算法，不替换冻结 E1 或当前评分协议。

- [详细效果分析](EFFECT_ANALYSIS.md)：四组指标、八场景、350 GT、TP/FP 分解、严重退化与机制证据。
- [源码清单](source_manifest.json)：原始服务器路径、逐文件 SHA256、字节数及分支基底。
- [四组完整结果](evidence/comparison/comparison.json)、[350 GT 表](evidence/comparison/per_gt.csv)。
- [指标提升原因逐实例分析](evidence/diagnostics/metric_paradox/analysis.json)。
- [退化实例各阶段证据](evidence/diagnostics/severe_case_postprocess_audit.json)。

## 代码导航

以下路径相对仓库根目录：

| 位置 | 内容 |
|---|---|
| experiments/trusted_full_online_20261104/online/room0/trusted_partition.py | 四种分票/弃权规则；其他场景保留实际部署副本 |
| experiments/trusted_full_online_20261104/online/room0/trusted_online.py | 可信区域投影、像素分区、关联器证据反馈 |
| 同目录 map_continue.py、map_support.py、commit_continue.py、scene_continue.py | 在线续建、可信历史与提交接入 |
| experiments/trusted_full_online_20261104/prepare_policy.py | 各场景接入准备 |
| experiments/trusted_full_online_20261104/run_parallel_v2.py | 本次八场景 online_B 调度入口 |
| 同目录 test_policy_integration.py、test_family_skip.py、test_exact_tracking_cache.py | 策略接入、分支兼容和精确缓存测试 |
| experiments/trusted_dbscan_20261105/run_parallel.py | 8 场景控制精确复现 → 8 场景候选后处理 → 四组评价 |
| 同目录 run_scene.py | 调用原 DBSCAN 数值函数，仅重绑定输入/输出根目录 |
| 同目录 compare_results.py、explain_metrics.py | 四组正式比较及指标原因分解 |
| experiments/unified_full_depth_20261101/ablations/dbscan_identity_completion_guarded_20261102 | 原始同版 DBSCAN 和历史核心补全 |
| experiments/trusted_dbscan_20261105/build_ply_view.py、viewer/ply.html、viewer/ply.js | 实例 PLY 双视图，同步视角、RGB、GT 参照及顶部裁剪 |
| experiments/trusted_dbscan_20261105/viewer/index.html | 四组指标与全部 GT 筛选页面 |

保持实际执行源码原始字节，未为发布而修改数值逻辑。历史 trusted_partition.py 的模块说明仍写着 “not yet wired”，这是归档注释；实际 trusted_online.py 和各场景 map_continue.py 已接入，并有完整运行与接入收据。部分保留的 coordinate.py/run_parallel.py 属于上游通用/历史调度，不应替代本次 run_parallel_v2.py。

## 策略实际行为

可信区仅使用当前检查点及之前已提交的修复区域。以 1.5 cm 查询可信 owner；有效支持门槛为至少 10 个不同点、且至少占 mask 投影点的 5%。不读取未来修复或最终 GT 进行在线决策。

1. 多个显著可信 owner：按 owner 分区，只保留显著 owner 支持的区域，其余弃权。
2. 只有一个显著 owner，且与原关联 ID 不同：整张 mask 弃权。
3. 没有上述情况，但存在局部竞争 owner 或歧义 owner：只对冲突部分弃权。
4. 没有冲突：沿用原关联。

改变的不仅是表面显示票：它移除关联器中不可分的原观察，并提交 /partition-id 子观察，重建关联索引，再处理下一帧。因此它会改变后续关联、告警区域、候选排名及修复历史，不能理解为“修好的实例简单加一个保护罩”。同帧重复归并、冲突弃权保持原规则；原生确认仍要求至少 2 票、置信比例至少 0.67。

50 帧以前没有前序可信修复，使用严格核验的相同前缀；从帧 51 开始干预。后续修复使用策略自己的历史。跟踪缓存只在 RGB、种子、对象顺序、模型、推理代码、配置指纹全部相同的情况下复用，不能复用改变后的关联或修复产物。

## 运行环境与复现边界

这是服务器绑定的科研快照，**不能宣称 clone 后即可在任意电脑直接训练/重跑**。源码中的绝对路径和哈希有意保持实际运行时原样：

- 代码：/home/chenkejun/CVPR/experiments/trusted_full_online_20261104、/home/chenkejun/CVPR/experiments/trusted_dbscan_20261105。
- 结果：/data/chenkejun/CVPR/results/trusted_full_online_20261104、/data/chenkejun/CVPR/results/trusted_dbscan_20261105。
- 原始对照：/data/chenkejun/CVPR/results/unified_full_depth_20261101。
- 原 DBSCAN：上述对照目录的 ablations/dbscan_history_covisible_20261102。
- main 评价工作区：/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009。
- 在线运行 Python：/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python。
- 汇总/诊断 Python：/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python。

依赖包括原始 RGB-D、模型权重、原生映射/跟踪、检查点收据、固定 TSDF、对应缓存、冻结 E1、原实验源码锁及 main 当前评价入口。数据、权重、点云、掩码、运行日志没有上传。原来的 policies 符号链接按源码锁中实际依赖文件展开保存，并不假装打包了完整外部部署环境。

以下仅记录已用入口，不应在已有结果目录上盲目重跑：

    # 完整在线，使用上述 online Python 和服务器路径
    python -B -u /home/chenkejun/CVPR/experiments/trusted_full_online_20261104/run_parallel_v2.py
    # 精确原版控制、同版候选 DBSCAN 及四组正式比较
    python -B -u /home/chenkejun/CVPR/experiments/trusted_dbscan_20261105/run_parallel.py
    # 对已完成结果做只读归因，使用汇总/诊断 Python
    python -B /home/chenkejun/CVPR/experiments/trusted_dbscan_20261105/explain_metrics.py

新实验必须先指定独立输出和正确部署路径、重建相应源码锁，不能把旧锁当成新实验收据。不要自动把本分支文件覆盖到已冻结的服务器实验目录。归档中 source_lock.json 描述原部署路径；本目录 source_manifest.json 描述此提交中的对应文件。

## 评价与展示

统一使用 main 默认 object_observed_repair / revision 2，350 固定 GT，frozen=false。同一最终 400 帧几何、RGB、GT 范围、对应索引；没有将最终 400 帧 GT 用于在线前缀选择。评分基准 cd260569e89de1e9525f82c45ee10842b80616b0、审核归档 d035f68770cefab27c9751e97062c72faa07367c 的含义不变。保留的运行 config 中出现历史工作区名称，不表示从该工作区执行了旧评估器。

两边都经过相同 final B 和相同终末 DBSCAN，后处理不回写在线状态。先对旧方案重跑 8 场景，逐点验证 xyz/RGB/instance_id 完全一致，再处理新方案。没有按 GT 调参。

PLY 只用于查看：固定相同点索引抽样，局部最多 18 万点、完整场景最多 24 万点；GT 用于事后裁剪、参照和配色，不用于算法决策。配色尽量保持同实例一致，真实分裂保留不同颜色；顶部裁剪仅改变显示。指标始终来自完整输出，不来自显示抽样。
