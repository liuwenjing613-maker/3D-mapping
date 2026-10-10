# P1-A1-strict-vote 无修复基线

此处补齐已经完成的严格一票无修复基线的生成代码与来源。选用 `abstain`：同帧同TSDF点只有一个候选身份时计一票，同身份多个支持去重，不同身份候选全部弃权。没有人工种子、SAM2修复或历史重关联。

## 实现与数据

- [one_vote.py](code/one_vote.py)、[run_ablation.py](code/run_ablation.py)：与评分预测冻结时的实际运行字节一致。
- `run_ablation.py`保留当时三种计票策略的消融入口；本分支当前基线只使用其中的`abstain`，不将其他策略结果混入。
- [预测冻结及代码哈希](evidence/predictions_freeze.json)、[运行等价性核验](evidence/runtime_equivalence.json)。
- [最新R2基线结果](results/summary_r2.json)：从已验证的当前R2比较数据提取，包含两个场景的原生/最终结果与三状态数量。历史消融的其他评估协议结果不作为当前分数。

| 场景 | 最终F1 | 最终PQ | U | T | C | 总灰点 |
|---|---:|---:|---:|---:|---:|---:|
| room0 | 67.19% | 54.46% | 75,842 | 10,174 | 58,491 | 144,507 |
| room2 | 66.67% | 49.94% | 77,936 | 17,780 | 53,705 | 149,421 |
| 合并 | 66.96% | 52.52% | 153,778 | 27,954 | 112,196 | 293,928 |

源预测：`/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/<scene>/abstain/final/instance_surface.npz`。原生预测和票据在同一`abstain`目录。冻结几何与原始身份关联不变。

生成脚本是固定服务器路径的实际运行快照，依赖冻结P1-A1输入；复算选择新的输出路径。后续评分只从main工作区使用当前默认 `object_observed_repair / revision 2`，不可直接使用历史消融评分结果。

## PLY

[完整实例导出脚本](exports/export_strict_baseline_ply.sh)与[已核验导出清单](exports/strict_baseline_ply_complete.json)；[仅三状态筛选脚本](exports/export_three_states_ply.py)与[数量及哈希](exports/three_states_export_complete.json)。仅三状态选择的是最终 `instance_id <= 0` 的点，保留其原生U/T/C分类。

三状态筛选脚本是本地交付工具的运行快照，相对路径以原 `outputs/p1a1_strict_repair_20261009` 为根；直接复制到该目录运行，或显式调整根目录。完整PLY及三状态PLY保留在CVPR结果目录，见[统一点云清单](../../p1a1_history_core_reassociation_20261009/exports/pointcloud_manifest.json)。

最新[重关联修复与三条件效果](../../p1a1_history_core_reassociation_20261009/README.md)在独立目录归档。
