# 最终种子 ABC：仅删除 A，窗户不变

当前策略是 final-abc-remove-a-only-keep-windows-20261101，由四个活动算法分支的 instance_repair/configs/current_auto_pipeline.json 注册。公共实现为 instance_repair.seed_final_abc。它在原筛选、第五步豁免、历史例外及投影预检完成，**原最终种子清单已经确定之后、跟踪之前**执行。不得插入语义、去重、包含、skip-repair 或 P/R 的中间；否则删除 A 可能使原本被过滤的候选重新进入。

## 精确行为

输入为原最终清单，输出为它的有序子集。只移除有合格证据的**确切 A ID**，不连带删除同类代表的其他 ID。不加入 B/C，不提升辅助 mask，不恢复旧候选，不重新执行后续筛选，不改动任何保留组的 ID、像素、辅助列表及其他元数据。删除 primary A 时移除其跟踪组，不从该组提升新的 primary。B/C 只是证据，即使原流程已经过滤它们也不会新增。若某个 B/C 自己也是原最终清单中的合格 A，则按它自己的证据独立判定。

原逐 mask 语义 top_class 为 windowpane 或 blind 的最终种子完全保留。不能用 ROI 或 alias 的 semantic_family=window 替代逐 mask 类别；例如窗户裁剪区域中的沙发仍按沙发处理。柜子不属于这次窗户保护。缺失逐 mask 语义时拒绝交接，不猜测类别。

沿用用户已确认的容差 ABC 判据：投影后 A 不小于 B、C；三者最小来源可见率 ≥0.80；|A∩(B∪C)|/|A| >0.70；|A∩(B∪C)|/|B∪C| >0.70；|B∩C|/min(|B|,|C|) <0.10。这不是像素严格相等。多个合格 BC 对不阻止删除 A，只要存在一对即可。

证据池沿用已通过来源、边界、噪声规则的完整 V1 池，并补齐全部 mask 的原语义判断；仅语义合格者参加 ABC。按原 primary 优先、支持点数降序及原 tie-break 取固定代表；双向共同可见覆盖严格 >0.90 归入固定代表，不做传递合并。仅直接相关代表参与 BC：支持点数 ≥20、与 A 交集 ≥20 且交集/较小支持 ≥0.20；枚举全部 BC 对。该证据去重不改写最终清单。

每组三元组仅在当前已到达帧中按既有 minimax 位姿距离顺序选视角：首个三者可见率均 ≥0.95 的帧，否则选最小可见率最高的帧。最后仍用 ≥0.80 门槛。深度容差 0.02m，往返误差 ≤1.5px，最近邻回投，不膨胀、不切割 mask。数据绑定来源 mask 哈希、帧、支持点、语义及几何；每次修复新建证据目录，不复用其他前缀的投影缓存。

## 当前入口与跟踪交接

先完成该分支已有的全触边前置筛选、重分组、语义/v2、豁免和预检，写出最终 seed_manifest.json，状态为 complete 或 automatic_v2_complete。本次不新增任何末尾触边筛选。此前独立采用的全触边前置规则继续由 seed_boundary 校验。

准备当前修复的 context.json，所有文件都必须来自同一个场景、同一个已到达前缀，路径可相对此文件：

    {
      "grouping": "candidates/v1/grouping.json",
      "supports": "candidates/v1/surface_supports.npz",
      "pairs": "candidates/v1/pair_metrics.npz",
      "mask_root": "candidates/v1",
      "visible_root": "candidates/overmerge",
      "geometry": "candidates/geometry/geometry_prefix400.json",
      "mapping_config": "mapping/config.json",
      "semantics": "abc_semantics/semantics.json"
    }

其中 geometry 的 frame_ids 必须是本次实际已到达帧，不能把最终400帧当作在线早期前缀。mapping_config 沿用 camera、source.scene_root、source.depth_pattern 和 source.trajectory（默认 traj.txt）。源深度应指向 mapping 可用数据，不能指向缺深度的 tracking inputs。

若现有语义只覆盖原 primary，先用原模型和缓存补齐证据池的逐 mask 语义（仅供证据，不改原筛选清单）。新帧推理使用已安装的本地 SegFormer 权重，GPU float32、关闭 TF32，不下载模型：

    python -B -m instance_repair.seed_abc_semantics --context context.json --classification candidates/v2/semantic/classification.json --out abc_semantics

该命令保存键为 mask ID 的 records（至少 top_class、exclude）和 source_hashes。semantics 路径必须指向其结果；已有全池语义须有相同 grouping 的来源哈希。随后执行注册的末尾入口：

    python -B -m instance_repair.seed_final_abc finalize --context context.json --manifest final_preflight/seed_manifest.json --out final_abc
    python -B -m instance_repair.seed_final_abc validate-selected --manifest final_abc/seed_manifest.json --receipt final_abc/receipt.json

**跟踪适配器必须把源清单切换为 final_abc/seed_manifest.json**，并运行注册的 tracking_handoff_guard。original_final_manifest.json 是审计副本，不能继续交给跟踪。校验拒绝使用旧清单、已变更的 mask/证据/代码和被篡改的最终清单。最终组的原 ID 保留，不重新编号。

当前仓库的历史运行器是带来源锁的实验快照；它们不会因新增模块自动改写。新运行应使用注册入口接入适配器，不直接重跑旧 launcher。此次不修改冻结 E1、历史实验代码/输出，也不触及 R2 或整场景末尾 B；这里的 B/C 证据与 scene-final-B 是不同概念。

## 验证范围

    python -B -m unittest discover -s instance_repair/tests -p 'test_seed*.py' -v

测试覆盖严格/包含等号边界、多组有效 BC、窗户/百叶窗保护、窗户 ROI 中的非窗户对象、只删 exact A、辅助不提升、缺失语义/早期清单/陈旧证据拒绝、原顺序和元数据保留，以及证据生成到交接的完整流程。

对用户确认的历史 main 八场景最终清单，回归为 163→156，删除7、增加0、恢复0，19个 windowpane/blind 全部不变。具体删除 room1 M0163/M0186/M0381、office0 M0016、office1 M0182、office3 M0194、office4 M0094；office4 M0197 继续保留。回归重算全部634条已缓存投影的面积判据，不用硬编码删除列表驱动算法。结果仅代表该历史输入的种子变化；最新 main 已另外更新 skip90 和前置全触边，本次不声称最新组合仍是163→156，也没有重跑跟踪、修复或地图评价。审计摘要见本目录 regression.json。

投影和全池搜索分别移植自服务器已验证的 near_complement_projection_20261026/projection.py、main_abc_evidence_select_20261101/search.py；最终处理对应 main_abc_final_a_only_keep_windows_20261101。可运行模块不依赖这些绝对实验路径。

额外用 office4 原始深度、位姿从零重建证据，25组三元组的目标视角、投影面积与通过/拒绝判定全部与已确认试验一致，未复用投影缓存。
