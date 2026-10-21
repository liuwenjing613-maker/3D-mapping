# onebyone：Replica 八场景逐帧逐点替换验证

本次验证已经完成。逐帧逐点替换消除了直接撤换导致的有效票丢失，但实例指标低于原 partial；目前作为独立实验保留，不改变已冻结 E1 和自动流程的默认注册。分支基于 `auto-v3-swap` 的 `1103d8196a980874ef743f572d3b4c0748a4fdab`。

## 验证范围与规则

沿用相同无修复基线、自动196种子和人工76种子，在 Replica room0–2、office0–4 的400帧上验证。自动组包含 room2 历史人工恢复的 G017/G025。种子、追踪、门控、ID关联、TSDF、可信范围、阈值与后处理均沿用原配置。用户确认只改直接撤换，随后接原 E1；本次没有实现新的可信历史重关联规则。

对每个“帧—TSDF点”，同一新 persistent ID 去重后只提交一次；若恰好一个新 ID，则撤换该帧该点的全部旧候选，不受 old family 限制；多个新 ID 冲突时保留原候选；没有新覆盖时保留原候选。实现见 [point_replace.py](sources/point_replace.py)、完整回放见 [replay.py](sources/replay.py)。

独立保存 `direct`（直接替换）、`core_all`（追加原核心历史改票）、`e1`（追加原自适应扩展历史改票）。每阶段均保存 native 和 final；final 使用原后处理。新 ID 冲突点在 direct 阶段保留原证据，后续原 E1 仍可能修改它。

评价从服务器 main 工作区 `/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009` 运行默认 R2 CLI：`object_observed_repair / revision 2`，固定350个可评价 GT，`frozen=false`。评分源码基准为 `cd260569e89de1e9525f82c45ee10842b80616b0`，审核归档为 `d035f68770cefab27c9751e97062c72faa07367c`。成对结果的几何、GT范围、对应索引和评分源码一致；预测在读 GT 前封存。

## 主要结果

以下均为 final，F1/PQ/AP 是八场景等权平均（%）；不是把所有场景 TP/FP/FN 合并后计算的 F1。

|组别|方法|F1|PQ|AP|
|---|---|---:|---:|---:|
|共同基线|无修复|62.79|49.08|18.77|
|自动196|原 whole＋E1|62.27|49.61|19.92|
|自动196|原 partial＋E1|62.48|49.69|19.91|
|自动196|逐帧逐点＋原 E1|61.79|49.14|19.48|
|人工76|原 whole＋E1|63.91|50.42|20.17|
|人工76|原 partial＋E1|64.61|50.88|20.49|
|人工76|逐帧逐点＋原 E1|64.16|50.48|20.06|

相对 partial＋E1，自动组 F1/PQ/AP 分别下降 **0.68/0.55/0.43 个百分点**，人工组下降 **0.46/0.40/0.43 个百分点**。人工组仍高于无修复及 whole，不能概括为所有版本都变差。

同阶段的直接替换对照也出现退化，说明问题已经发生在 E1 历史改票之前：

|组别|whole 直接 F1|partial 直接 F1|逐帧逐点直接 F1|追加核心历史 F1|追加完整 E1 F1|
|---|---:|---:|---:|---:|---:|
|自动196|63.63|63.85|62.44|61.33|61.79|
|人工76|63.08|63.78|62.97|63.71|64.16|

## 票数验证与退化解释

直接阶段的“原来有有效票、修改后变成零票”的帧—点数量：自动组 whole=3,737,138、partial=2,374,295、逐帧逐点=0；人工组 whole=1,663,922、partial=546,771、逐帧逐点=0。这是跨帧计数，不是独立 TSDF 点数。逐帧逐点后每个 TSDF 点的总有效票数也均不下降，授权范围外和新 ID 冲突点的原候选完全一致。

仍有自动组135,864、人工组38,137个 TSDF 点从基线的 confirmed 变成 direct 阶段未分配，且这些点总有效票数都没有下降；原因属于身份竞争，不能再解释为撤票后没有票。详细状态转移见 [comparison.json](evidence/comparison.json)。

具体案例是自动 room0 的 GT44/56/80 三个插座：partial＋E1 的最佳 IoU 为0.643/0.614/0.724，逐帧逐点＋E1 降为0.345/0.038/约0.00007。G008/M0027 的 persistent ID2 mask 覆盖这些插座，分别把原 ID346/348/227 的386/567/663条帧—点有效票改成 ID2。原 family `[2,179]` 不含这些插座 ID，而新规则允许全部替换。结果既有错误归属，也有跨帧竞争导致的未分配，并非所有受影响点都被最终分给 ID2。GT65 盘子也存在 ID20→7 的2018条改票。证据见 [关键退化病例](reports/关键退化病例.txt) 与 [逐点诊断](evidence/diagnostics/room0_automatic_regressions.json)。

不能把局部小物体退化推广为所有点归属变差：自动组 final 正确归属覆盖率从 partial 的78.71%微升到78.78%，错误归属从12.96%降到12.86%，同时实例 F1/PQ/AP 下降。大物体点数改善和小物体实例损失可以同时出现。因此，“唯一的新 mask 候选”还不足以证明该点属于该对象；下一步应先验证点级替换资格、mask可靠性和小实例保护，不能仅凭零票损失为0就切换默认方法。

## 文件与复现边界

- [完整 HTML 报告](reports/逐帧逐点替换对比.html)：native/final、分阶段和各场景对照。
- [完整指标表](reports/metrics.csv)、[结果说明](reports/结果说明.txt)、[直接阶段对照](reports/直接阶段对照.txt)、[逐 GT 记录](evidence/per_GT.json)。
- `sources/`：本次服务器实际执行的21个 Python 文件和 config.json，逐字节保存；包含映射、评价、调度、诊断和报告代码。
- `evidence/`：16组来源锁、预测封存、协议、映射/评分验收以及汇总记录；大型 NPZ、逐帧事务和原图保留服务器，通过来源路径及哈希定位。
- `package_manifest.json`：仓库文件 SHA-256、来源路径，以及原交付压缩包哈希。`verify_package.py` 可离线核验归档完整性。

```bash
python -B experiments/onebyone_20261021/verify_package.py
# 需要 numpy；独立集合参考实现验证核心替换规则：
python -B experiments/onebyone_20261021/sources/test_point_replace.py
```

本次504个独立集合对照用例、16组完整回放、96条新阶段评分和64条同阶段旧方法评分全部通过。完整回放还核验了冻结原支持、相同新支持、增量/全量一致、精确回滚和原 E1 历史改票等价性。通过这些检查表示实现和比较口径可靠，不代表指标提升。

源码保留原服务器绝对路径：代码 `/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021`，结果 `/data/chenkejun/CVPR/results/frame_point_replacement_20261021`。**不要重跑 launcher 覆盖已有输出。** 完整重算需要服务器源数据和环境；另建 CVPR 代码/结果目录的路径适配副本并记录差异，再依次运行 `prepare.py`、16组 `replay.py`/`evaluate_v2.py`、同阶段旧方法 `score_direct_controls.py` 和汇总报告程序。group/scene 通过 `POINT_GROUP`、`POINT_SCENE` 选择，评价始终从指定 main 工作区执行。

归档保留第一次评价遇到的成对评分文件名错误记录；最终 `evaluate_v2.py` 读取 main 实际输出的 `paired_repair_metrics.json`，复用已核验评分后全部完成。`evaluate.py`、旧调度器与恢复记录仅用于审计；它们未改变映射算法或评分源码。
