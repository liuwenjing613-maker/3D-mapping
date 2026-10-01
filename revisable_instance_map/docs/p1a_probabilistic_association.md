# P1-A：可撤销帧级计票与概率支持关联

基于 `main` 的 `52b10d9652dfe45829bac41b3ff6dc60814398e5`。分支为 `codex/p1a-probabilistic-association`。

本版仍是在线硬关联：一条观测只有一个操作性 ID。体素支持比例随新观测变化；历史观测不随 argmax 自动迁移，实例 ID 不自动改名。P1-B 的暂定隔离、自动修复、因果重放和新去噪均未启用。

## 三个对照入口

`run_baseline_association.py --association-mode MODE` 支持：

| MODE | 用途 |
|---|---|
| `legacy`（默认） | A0，原二值支持关联 |
| `binary-ledger` | A1，新增计票账本，评分与选择仍用原方法 |
| `probabilistic` | A2，以概率几何一致性替换评分的几何项 |

固定来源归组 mask、采样方式、3 cm 格子、本格加 18 邻域、原始几何覆盖率门控、可见性检查、原阈值与同帧多 mask 关联策略。所有新实验从空图开始。A0/A1 要逐字段复现原关联，并逐数组复现原 P0。

## 计票与评分

`C(v,k)` 是支持格子 v、实例 k 的不同处理帧数。`P_v(k)=C(v,k)/sum_j C(v,j)`。同帧、同格、同实例最多一票；同帧不同实例仍保留竞争。总帧实例票数和不同观测帧数分别记录，二者不能混用。

每个当前观测格子的匹配强度为邻域中 `P_u(k)` 的最大值；再对当前观测全部格子求均值，未匹配格子贡献零。这个邻域匹配强度不是跨实例归一化的概率分布。

有可见性验证时 `S=0.5*G_prob+0.5*visible_overlap`；缺少可见支持时 `S=G_prob`。`geometric_coverage` 保留旧含义，新字段为 `probabilistic_geometric_consistency`。

所有 mask 先对上一处理帧结束的同一状态评分，再提交整帧。邻域只读；只给实际投影到的格子计票。

候选相对权重按 softmax 计算，含 null 候选（默认基准为原最低总分 0.35，温度 0.1）。分母覆盖空间召回并评分的全部至多 8 个候选，即使日志的原 `candidates` 只保留前三。相对权重仅用于诊断，不作为已校准后验，也不用于计票或门控。

## 来源、撤票、版本

不可变观测 ID、帧号、mask 哈希和采样格子是重建依据。当前操作性归属附有 `assignment_version`；整个状态有 `map_version`。每帧提交与显式归属修订提升状态版本。

`R(f,v,k)` 保存同帧重叠观测的引用数。只有引用从 1 变 0，才撤掉该帧的票；A→B→A 应恢复原统计。无效或重复事务在变更前拒绝；不变的重分配不加票、不记修订事件。

`reassign_observations(updates, reason)` 只改指定观测，增量撤票与加票，重建二值空间索引，并记录 `REASSIGN` 事件。它不重新执行后续关联。持久化 `next_instance_id` 高水位，即使最高 ID 已退休也不复用。

`observation_support_3cm.npz` 是可恢复检查点，包含来源、当前归属、版本、高水位、配置和修订事件。加载后由账本重建概率缓存；不从最终缓存反推原始观测。

`identity_voxel_counts_3cm.npz` 是派生计数缓存。每帧 `candidate_support_traces/fXXXXXX.npz` 保存所有已评分候选的邻域最大值来源格子、对应 ID 的支持帧数、总帧实例票数、不同支持帧数，以及查询时的状态版本。未匹配的格子计零，不写入该稀疏追踪文件。

## P0 与显式修订

3 cm 身份缓存不直接发布表面标签。P0 继续使用逐帧、逐表面点、逐实例去重，并沿用至少 2 票、比例 0.67 的发布规则。

`rebuild_p0_from_regions.py` 校验原始观测目录、配置、缓存 manifest 和逐帧支持文件哈希，调用原 P0 的计票与发布函数重新融合。最终 TSDF 的缓存仅用于完成后的离线发布，绝不进入在线关联。

`revise_identity_associations.py` 接收 JSON 格式的 `observation_id -> instance_id` 显式更新，生成新的关联表、检查点、概率缓存、修订日志，并重建 P0。输出 `revision_manifest.json` 为 PASS 才表示完整发布，输入目录不被修改。历史候选仍表示原决策时查询的证据，不冒充修订后的新评分。

```bash
python revisable_instance_map/tools/revise_identity_associations.py \
  --association-dir /data/chenkejun/CVPR/revisable_instance_map/RUN/room0/A2/association \
  --updates-json /home/chenkejun/CVPR/updates.json \
  --source-surface-dir /data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/surface_p0 \
  --observations /data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/observations/observations.jsonl \
  --config /data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/configs/p0_parent_regrouped.json \
  --output-dir /data/chenkejun/CVPR/revisable_instance_map/REVISED_RUN \
  --reason 'explicit reviewed identity correction'
```

main 已有的离线扩散仍可作为单独后处理：对各版本重新运行同一 `repair_p0_surface.py` 配置，不回灌补全标签为观测票。

`run_p1a_postprocessing.py --native-root NATIVE --output-root NEW_OUTPUT` 会分别重新运行 A0/A1/A2 的原有扩散，并对原生与扩散结果执行统一 v3 pooled 评估。默认覆盖原生验证中的全部场景。它核验原生运行的源码快照与配置来源，不要求后续只改检查点读取效率的代码仍与旧快照逐字一致。

## 验证与评估

```bash
python -m unittest discover -s revisable_instance_map/tests -v
python revisable_instance_map/tools/run_p1a_validation.py \
  --scenes room0 --jobs 3 --audit-prefix \
  --output-root /data/chenkejun/CVPR/revisable_instance_map/NEW_RUN
```

验证入口运行新的 A0/A1/A2 400 帧、重建 P0、统一 v3 评估、原基线逐字段/逐数组等价核验，以及独立前 50 帧的因果核验。v3 参数沿用当前统一口径：主半径 0.01 m、诊断半径 0.02 m、GT 最小 100 顶点、显著交集 10 顶点/比例 0.05；配置当前仍标记 `DEBUG_ONLY`。

源码哈希、输入配置/目录哈希、阈值、逐帧决定、计数、追踪和指标都保存在新实验目录。首轮保持同参数，结果出来后再讨论开发场景调参，不能预设概率版必然优于基线。

边界：一帧纯错误支持也可能在邻格产生比例 1；持续错误且缺少独立正确实例假设时不会自动分裂。3 cm 量化误差不会因概率表示消失。模糊关联在 P1-A 中仍提交，避免其污染属于后续 P1-B。

2026-10-01 的首次八场景验收与指标见 [P1-A 验收报告](p1a_validation_20261001.md)。
