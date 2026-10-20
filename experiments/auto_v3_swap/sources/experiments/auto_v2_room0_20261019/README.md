# room0 auto-v1 → 完整 v2 → 原版 E1 → R2

本次运行由用户确认：room0 若没有自动 v1 结果，就按 room2 的 auto-v1 流程生成种子，再运行完整 v2、原版 E1 和当前 R2 评价。旧的 room0 人工种子清单不作为本次输入。

服务器代码：`/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019`。数据、日志和结果：`/data/chenkejun/CVPR/results/auto_v2_room0_20261019`。入口与输出均使用本次独立目录；不要重跑已完成阶段并覆盖结果。

## 输入与唯一的人工来源纠正

auto-v1 使用与 room2 相同的 C/U/T 各 top20、三视角候选、过合并检查、ROI 交集版本选择、语义 SAM 路由、裁剪边缘和噪声过滤、完整 3D IoU > 0.25 固定主种子分组。原始分割 mask 不重画、不腐蚀、不扩大。相同输入的模型缓存逐文件校验后复用。

用户明确纠正：“M0466不采用SAM结果，因为不是柜子”。`grouping_r2/source_overrides.json` 将纠正限定在已核验的 M0466 对象组；该对象回到普通候选分支，不使用语义 SAM 优先替换。其最终主种子是 G030 / M0466，来源为 CropFormer，原始 mask SHA-256 为 `aa312039fa7b94998f7807f5e39f3557aa152366eba35d121bae853174c2471c`。它仍照常经过全部 v2 规则，最终保留进入 E1，没有人工恢复 v2 排除项。

另一跨视角混合来源组按既有 SAM-only 约束去除 M1112 的 CropFormer 别名后重新分组。该约束处理记录于 `source_policy_update_audit.json`；在 room2 原始 836 个候选、202 个分组上的回放没有新增排除。`grouping_r3` 继承已验证的来源资格和人工纠正，同时重新核验像素与支撑哈希。

## v2 的实际结果

| 步骤 | 输入 | 保留 | 本步移除 |
| --- | ---: | ---: | ---: |
| 背景语义：置信度 ≥ 0.7 的墙/地板/天花板/地毯像素占 mask ≥ 90% | 103 | 80 | 23 |
| 共同可见 3D IoU ≥ 0.8，固定主种子去重 | 80 | 65 | 15 |
| 覆盖充分：p ≥ 0.95 且 r ≥ 0.95，跳过修复 | 65 | 50 | 15 |
| 完整支撑交集 / 较小支撑严格 > 70%，去掉较小支撑 | 50 | 43 | 7 |
| p > 0.1 且 r < 0.2 | 43 | 27 | 16 |

共 27 张进入 E1、61 张被筛选排除、15 张跳过修复。p = C/N，r = C/I；N 是种子全部支撑（含未分配点），I 是 matched strict 实例在来源帧可见的全部支撑，C 是交集。没有额外裁剪 I，没有传递去重或 room2 的人工恢复。详细逐步证据以 `v2/selection.json`、`v2/measurements.json` 和 `v2/semantic/classification.json` 为准。

## 原版 E1 与评价一致性

27 张主种子保留原始像素和原始 track ID；同帧完全不相交的 mask 可合并为追踪任务，实际共 22 个任务。E1 使用 main 冻结算法的数值实现，运行前通过 54 项既有检查并冻结输入、代码。流程包含 identity、追踪、raw repair、strict repair、history core、expand、E1。排除和跳过的种子不进入修复。

评价从 `/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009` 的默认入口运行，协议为 `object_observed_repair / revision 2`，开发协议仍为 `frozen=false`。修复前对照是 ROI 选择所用的同一 strict 地图；修复后是本次 27 张种子的原版 E1 输出。本实验并未另跑一份“103 张 v1 种子全部修复”的对照，页面不将 strict 基线标作 v1 修复结果。

GT 只在预测冻结后进入评价。修复前后分别评价 native/final，并执行配对修复评价；程序核验几何、RGB、400 帧来源范围、GT 范围、observed support、固定对应索引和协议配置一致。指标完成状态必须以 `e1_repair/run_complete.json` 和 `e1_repair/evaluation_complete.json` 为准，不以进程退出、页面存在或阶段完成代替。

## 查看和核验

`build_review.py` 生成结果目录下的 `review/index.html`，展示 103 张原始种子、27 张保留种子、56 个来源帧的前后对照、逐步排除原因、对应 mask、语义证据、来源模型完整分割及真实评价表。E1/R2 尚未完成时，页面明确显示指标待完成；只有整体通过才读取评价结果。页面下载区包含原始二值 mask、筛选与一致性报告。

第 2、3 步支持“按组对照”。第 2 步直接使用 `selection.dedup_groups`：80 张种子组成 65 组，默认展示含重复项的 10 组 / 25 张，列出固定主种子、15 张排除成员及逐对 IoU、交并计数。第 3 步仅收录实际进入该步的 65 张种子，按 matched strict 实例 ID 分为 32 组；未匹配正实例的种子各自单列。默认展示含跳过项的 15 组 / 22 张，包含同组继续筛选的成员，逐张列出 p、r、N、I、C 和覆盖叠图。组内 I 按各来源帧分别计算。两种视图均区分本步决定与最终去向，搜索命中成员时保留完整分组；不改变任何 mask、选择决定、修复预测或评价值。直达片段为 `#groups-iou` 和 `#groups-coverage`。

关键结果文件：`v1/seed_manifest.json`、`v1/audit.json`、`v2/complete.json`、`v2/selection.json`、`v2/source_lock.json`、`e1_repair/selected_input_audit.json`、`e1_repair/pipeline_code_freeze.json`、`e1_repair/e1/E1_expanded_history/predictions_freeze.json`、`e1_repair/evaluation_r2/comparison.json`、`e1_repair/evaluation_r2/protocol_freeze.json`、`e1_repair/evaluation_r2/per_GT.json`。

`review/pr_exemption.html` 是仅改变第 5 步豁免范围的新增 mask 可视化预览。`build_pr_exemption_preview.py` 从既有选择清单和主种子来源标记推导：上游 `semantic_hit=true` 且 family 为 window/cabinet 时，新增 G026/M0301、G027/M0350、G063/M1082、G097/M1280，27 → 31；仅按后续 SegFormer 主类别的宽口径还新增 G058/M1012、G069/M1150、G102/M1297、G103/M1306，27 → 35。页面区分两种口径，提供原图、二值 mask、叠加图与局部放大；`pr_exemption.json` 保存清单、p/r 计数、来源哈希及未重跑标记。前 4 步保持原决定，预览不改写任何选择或修复输出，也没有重跑 E1/R2。

历史失败尝试保存在各 `attempt*` 目录中，用于追溯来源守卫和路径适配；它们不属于最终成功结果。

E1 严格投票首次启动时，历史双来源哈希检查引用的 room0 批次路径不存在。经确认原依赖、本次依赖和 main 冻结依赖的 SHA-256 均为 `c3c956071eea48ae0b13db33363eccb2d5bc05cfdf87d05f416301d975f0b19b`，缺失路径仅补建指向 main 冻结文件的符号链接。`e1_repair/legacy_path_compatibility.json` 记录完整路径和核验；原 E1 源码哈希完全未变，保留已通过的追踪与 raw repair，从 strict repair 继续。失败日志保存在 `e1_repair/attempt1_strict_path_failure`。

## 无修复、自动与人工的统一对比

按用户追加要求，`compare_three_way.py` 从 main 默认 R2 入口重新评分三组冻结地图的 native/final：strict 无修复、当前 27 张自动 v2 种子 + E1、历史 41 张人工标注种子 + E1。人工组使用 main `instance_repair/frozen_e1_20261012/room0/reference.json` 锁定的 `/data/chenkejun/CVPR/results/room0_observation_repair_20261012/E1_expanded_history`；其旧报告中的 C0010 对照已经经过修复，不当作无修复基线。

新评分保存在 `comparison_three_way`，不覆盖既有指标或地图。六份预测都核验固定几何、RGB、400 帧来源、72 个 GT、observed support、固定对应索引和 R2 配置；同时执行每个输出层的无修复→自动、无修复→人工、人工→自动配对评价。自动与无修复指标须精确复现上一轮结果，人工 final 指标须精确复现 main 冻结参考。页面展示三方主指标、参考面归属质量和逐 GT 最佳 IoU。种子数量、所选对象及身份关联不同，因此这是实际结果对比，不是等数量种子消融；人工组也不等于直接以 GT 作为预测。
