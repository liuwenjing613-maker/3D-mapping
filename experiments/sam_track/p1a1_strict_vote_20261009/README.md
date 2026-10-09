# P1-A1：严格一帧一点一票，歧义弃权

本分支归档已完成的实现与 room0/room2 修复复验。从 `main@29304d8` 建立，统一使用最新 **Replica-CA-v3 / object_observed_repair / revision 2**。冻结 P1-A1、既有修复、SAM 追踪及评分器均未修改。

结论：完整追踪修复仍有质量收益；严格计票的额外收益较小，没有消除大部分新增冲突，且修复目标的完整度仍有损失。

## 实现

原规则只对同帧 `(表面点, 实例ID)` 去重。新规则先合并整帧的保留旧支持和新增支持，再归约生效票：

- 同一点只有一个 ID：一票；多个 Mask 支持相同 ID 也只是一票。
- 同一点出现不同 ID：全部弃权，不按 ID 大小或支持像素数选择赢家。
- 原始 Mask–点支持保留。撤回一个 Mask 后，若还有同 ID 支持，共享支持仍在；若消除了多 ID 歧义，原先弃权的旧票可以恢复。
- 修复以完整帧归约前后的票差更新账本；不能分别归约旧支持与新支持，也不能直接按 Mask 数加减票。

核心代码：[strict_vote_ops.py](code/strict_vote_ops.py)、[one_vote.py](code/one_vote.py)。实际运行脚本：[run_strict_repair.py](code/run_strict_repair.py)。本次选定的是 `abstain`；`one_vote.py` 中的其他策略仅为既有消融接口，未用于本次严格修复结果。

## 最新 R2 效果

两场景合并的最终结果。三状态统计 TSDF 未分配点；归属 coverage 统计固定 GT 参考顶点，二者不能混为同一计数单位。

| 计票规则 | 修复条件 | F1@50 | PQ | mCov | 三状态点 |
|---|---|---:|---:|---:|---:|
| 原规则 | 未修复 | 66.964% | 52.460% | 53.797% | 295,795 |
| 原规则 | 仅种子帧 | 66.964% | 52.466% | 53.804% | 296,085 |
| 原规则 | 完整追踪 | 69.388% | 54.907% | 58.353% | 331,201 |
| 严格一票＋弃权 | 未修复 | 66.964% | 52.522% | 53.864% | 293,928 |
| 严格一票＋弃权 | 仅种子帧 | 66.964% | 52.526% | 53.872% | 294,229 |
| 严格一票＋弃权 | 完整追踪 | 69.388% | 55.051% | 58.510% | 328,511 |

严格完整追踪相对严格未修复：F1 +2.423、PQ +2.530、mCov +4.646 个百分点；评分正确归属 +22,424、错误归属 −36,193 个参考顶点。正确增量中，标签改变位置净增 18,871 个；标签未变、仅评分一对一对齐变化的位置净增 3,553 个。

三状态净增 **34,583**：无生效票 U +416、证据不足 T +313、冲突 C +33,854。逐场景：

| 场景 | 严格未修复 U/T/C | 严格完整修复 U/T/C | 三状态净增 |
|---|---|---|---:|
| room0 | 75,842 / 10,174 / 58,491 | 76,118 / 10,298 / 59,337 | +1,246 |
| room2 | 77,936 / 17,780 / 53,705 | 78,076 / 17,969 / 86,713 | +33,337 |

扣除计票改变对未修复基线的影响，严格规则额外增加的修复 PQ 收益为 **0.082867 个百分点**，F1 无额外改善；修复净新增三状态仅少 **823 个（2.3245%）**。

45 个冻结修复目标的平均 best-IoU +12.602 个百分点，平均 best-GT recall −6.668 个百分点；20 个目标 IoU 提升超过 0.01、6 个下降超过 0.01。全部 124 个可评价对象平均 best-GT recall −2.319 个百分点；主结构 merge 31→29、split 35→38。不能将更少错误归属或更高 IoU 解读为完整度全面提升。

仅种子帧修复的 F1 未提升，正确归属略减。主要收益来自完整追踪；下一步应优先检查退化目标的跨帧历史身份关联。本次不替换冻结基线。

## 正确性与控制

- room0 最新 12 个案例、room2 Top30 中实际修复的 14 个案例；分别复验种子帧与完整追踪。人工修复使用额外 Mask/历史 RGB，不能解释为与自动未修复方法等监督预算。
- 10 项语义测试，包括 100 组随机撤销/重放；800 原始帧重新投影；802 个真实修复事务与冻结 Mask 编辑完全一致。
- 不导入生产计票算子的独立审计覆盖 1,600 个帧条件：每帧每点至多一票，从零累加等于增量票表，逐点总票不超过 400。
- 全量等于增量、精确回滚，冻结范围外的票、证据及最终标签不变；同 ID 共享支持保留。
- 预测在首次 GT 读取之前冻结，不以 GT 重选案例、身份或参数。全部 24 份原生/最终预测在相同 R2 GT、配置、几何对应和源码下评分。
- 8 个原规则场景控制与 R2 审核归档逐字段一致，3 个最终合并控制也一致；24 个参考归属分区均独立核验，评分前后输入/预测/协议哈希不变。

R2 评分源码 `cd260569e89de1e9525f82c45ee10842b80616b0`，审核归档 `d035f68770cefab27c9751e97062c72faa07367c`；配置 SHA256 `6668792cbbeff3b6a6b6f2302ad8baa1b202f463ae26306350d4617c07f601a0`。固定八场景 350-GT 范围保持不变，本次仅重评其中 room0 的 72 个和 room2 的 52 个对象。仍为开发协议，`frozen=false`。

过程中的两项诊断纠正已保留：

1. 原重放 receipt 的 `ambiguous_point_frames_before/after` 实际存储歧义候选键数量。真正 `(frame, point)` 碰撞次数请用 [独立审计](evidence/mapping/independent_correctness.json)。
2. 历史目标列表是 compact evaluation ID，不是 R2 物理 ID。最初为空的分组诊断已保留；随后按完全相同的参考顶点验证一一映射，校正了 28 份分组/归因报告。**目标分析以 [corrected_pairs](results/r2/analysis/corrected_pairs) 为准**；未改动预测、主指标或 GT 范围，不能对 compact ID 取模猜测对应关系。

## 文件与复核

- [完整中文报告](results/r2/严格一票修复复验结果_R2.txt)
- [完整指标 CSV](results/r2/comparison.csv)、[指标 JSON](results/r2/comparison.json)、[同规则修复增益及差中差](results/r2/effect_analysis.json)
- [校正目标编号和标签／评分对齐归因](results/r2/analysis/targets_and_attribution_complete.json)
- [预测冻结与源文件哈希](evidence/mapping/predictions_freeze.json)、[独立正确性审计](evidence/mapping/independent_correctness.json)
- [原始核验证据包](artifacts/严格一票修复复验_R2_证据包.zip)、[归档来源及逐文件 SHA256](publication_provenance.json)

运行便携语义测试（只需 NumPy）：

```bash
python -m unittest discover -s experiments/sam_track/p1a1_strict_vote_20261009/tests -p 'test_strict_vote.py' -v
```

测试只调整导入路径，十项测试正文与随机试验保持原样。`code/` 保存实际服务器运行快照；`code/dependencies/` 和 `code/evidence_replacement.py` 从 [冻结修复提交 42dc4f7](https://github.com/liuwenjing613-maker/3D-mapping/tree/42dc4f7760cffa864f926512f211778113616813/experiments/sam_track) 读取，并逐字节核对本次预测冻结记录的哈希。

服务器运行依赖 NumPy、SciPy、Pillow 与既有冻结输入。代码目录 `/home/chenkejun/CVPR/experiments/p1a1_strict_repair_20261009`，结果目录 `/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009`。冻结几何、Mask、追踪结果和大体积地图保留在服务器，路径与哈希均已归档。

复验顺序为：`run_strict_repair.py` → `independent_correctness.py` → `evaluate_strict_repair_r2.py` → `analyze_targets_and_attribution_r2.py` → `build_report_r2.py`。脚本为固定服务器路径的 as-run 快照，重新计算须选择新的输出目录，不覆盖已冻结结果。严格未修复控制来源于 `/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/<scene>/abstain`，其逐数组等价性已核验。`evaluate_strict_repair.py` 等非 R2 脚本仅保留作历史追溯，不用于当前评分。

当前评估入口：

```bash
python -m unified_eval.cli current-protocol
```

当前 main 的评分代码和配置保持原样；实验分支不携带另一套默认评分器。
