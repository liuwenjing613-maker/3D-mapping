# two-audit：原始观测缓存 v1

本环节已获用户确认：离线修复与 online-v2 共用原始观测缓存，先在离线上验证。这里只建立来源可追溯、绑定当前几何的 Mask 支持和可见性；稳定簇、不一致检测、候选保护、去重及修复决策等待后续逐项确认。

## 入口与基线

代码在服务器 two-audit 分支，Git 起点为全深度实验登记的 online-v2 提交 a15f7612c71ce38466c07024a45653ac10b85a92。缓存通过明确的 mapping-root 读取已有快照；离线读取现有 main 的全场景无修复建图快照，online-v2 读取对应已到达检查点。没有用 online-v2 的筛选参数替代离线 main，也没有修改现有建图、种子、E1、最终 B、DBSCAN 或评价代码。

共享入口：

    python -B -m instance_repair.two_audit_cache build \
      --mapping-root /data/chenkejun/CVPR/results/scannetpp_v2_unified_dbscan_20261103/main/5ee7c22ba0/1523/A/mapping \
      --output /data/chenkejun/CVPR/results/two-audit/observations_v1_20261104/offline/5ee7c22ba0 \
      --mode offline --checkpoint 1523

    python -B -m instance_repair.two_audit_cache verify \
      --cache /data/chenkejun/CVPR/results/two-audit/observations_v1_20261104/offline/5ee7c22ba0 \
      --verify-sources

在 /home/chenkejun/CVPR/worktrees/two-audit 中运行。已验证的 Python 是 /data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python，设置 PYTHONDONTWRITEBYTECODE=1、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。重复构建应使用新的独立输出目录；已有目录不会被覆盖。

online-v2 使用相同入口，传入对应的 mapping-root、--mode online-v2 和检查点帧数。例如 100 个建图帧对应原始帧 0,5,...,495；checkpoint 不是原始帧编号。离线必须覆盖 original_config 指定的整个实验序列；在线必须是该序列的精确前缀。缓存没有截断为候选生成的新增时段。

## 原始证据与来源约束

读取原始 CropFormer 标签 PNG、原始深度、相机位姿、静态或逐帧内参，以及原始 observations.jsonl 和推理 frames.jsonl。逐帧 Mask 路径必须解析到 original_config 的原始来源，Mask 文件哈希、ID 集、像素数及有效深度数必须与原始记录一致。地图实例 ID、association ID、修复像素 journal 和带历史修复身份的 frame_support 均不作为观测输入。

observation_key 由数据集、场景、帧号、帧内 Mask ID、原始标签图 SHA256 和来源模型计算。source_mask_sha256 指整个原始标签 PNG 的文件哈希，配合 mask_local_id 唯一确定该帧区域；不是把整张标签图当成一个 Mask。重复读取同一原始观测不会产生新的独立证据身份。

深度、Mask 按允许的建图帧列表逐帧读取。位姿、逐帧内参及原始 Mask 元数据仅消费截至当前 raw_cutoff 的前缀，并记录所消费前缀的字节哈希。构建末尾重新校验所有实际使用的输入，源文件或符号链接变化会失败。

## 几何与投影

geometry_id 绑定 TSDF 有序 xyz 的形状、dtype 与精确字节哈希。点集合相同但顺序变化也属于不同索引空间。每个帧包带 geometry_id；原始观测身份可以跨检查点保持，表面索引必须重新投影。

投影采用仓库已有 surface_projective_evidence.project_surface_regions 的 surface-to-pixel 约定：相机坐标 z>0 且 z<10 m，numpy.rint 取最近像素，像素在图内、测量深度有效且小于 10 m，并满足绝对深度差不超过 0.015 m。这些是继承的几何有效性参数，不是稳定性或异常检测阈值。参考源码 SHA256：affb77e3458572984bc621d79709ff5a4ffaa9804a9ebb1902e321fd0895de62。

V_t 保留所有通过深度检查的表面点，包括标签为 0 的背景像素。每个原始 Mask 的 S_t 是 V_t 中投影像素标签等于该 Mask ID 的子集。无有效深度、无表面支持及触边 Mask 均保留记录和质量标记；这些标记不是修复候选的 ACCEPT/REVIEW/REJECT。该审查投影不改变原建图的严格一票投影和投票。

## 输出

| 文件 | 内容 |
|---|---|
| request.json | 模式、检查点、几何身份、投影设置、输入及实现来源锁 |
| mask_observations.jsonl | 原始观测身份、来源、像素/深度质量、支持数量与帧包引用 |
| frames.jsonl | 位姿、当帧内参、原始文件哈希、可见性统计和帧包哈希 |
| frames/fNNNNNN.npz | V_t、有序投影像素及帧内标签，按 Mask 分组的 S_t 索引和 offsets |
| status.json | 当前进度及失败原因 |
| manifest.json | 所有构建和输入不变性检查完成后才发布的完成清单 |
| validation.json | 实验驱动器执行独立读回与来源校验后的结果 |

缓存不复制数据集、RGB、深度或 TSDF 坐标实体；大数组仅是当前表面的支持索引。全部结果位于 /data/chenkejun/CVPR/results/two-audit 下。manifest 的 COMPLETE 只表示观测缓存完成，不表示异常筛查、修复或效果评价完成。

读取点索引时必须显式绑定几何：

    from instance_repair.two_audit_cache import ObservationCache
    cache = ObservationCache(cache_path)
    geometry_id = cache.require_geometry(current_xyz)
    support = cache.support(observation_key, geometry_id=geometry_id)
    visible = cache.visible(frame_id, geometry_id=geometry_id)

批量读取同帧多个 Mask 时可用 read_frame(frame_id, geometry_id=geometry_id) 一次取出分组 offsets 与索引，避免逐个重复解压。

## 验证与下一环节

单元测试覆盖深度遮挡和背景可见性、无效/零支持 Mask、逐帧内参、离线完整序列、在线禁止未来输入、原始来源替换、几何重排、缓存损坏和输入修改。真实帧正 Mask 支持与原 P0.1 投影函数进行逐点对照。全序列缓存完成后独立读回全部帧包，并重新核验实际输入。

本环节不读取 GT、不生成 F1/AP50。后续实际评价仍从 /home/chenkejun/CVPR/worktrees/v3-r2-main-20261009 的当前 R2 入口运行，保持相同几何、GT 范围与对应索引。下一环节讨论共同可见区域比较、跨帧关联、独立视角和稳定性条件。
