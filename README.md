# 3D-mapping

## P1-A1-strict-vote 分支最新结果（2026-10-10）

本分支包含[严格一票无修复基线](experiments/sam_track/p1a1_strict_vote_20261009/baseline_materialization/README.md)、[既有严格追踪修复](experiments/sam_track/p1a1_strict_vote_20261009/README.md)和[最新核心区域历史重关联](experiments/sam_track/p1a1_history_core_reassociation_20261009/README.md)。room0、room2共26个案例使用同一R2开发协议评价；完整代码、三条件结果、全部案例图、退化分析及PLY来源均已归档。

| 两场景最终条件 | F1 | PQ | 灰点 |
|---|---:|---:|---:|
| 严格一票无修复 | 66.96% | 52.52% | 293,928 |
| 既有严格追踪修复 | 69.39% | 55.05% | 328,511 |
| 加核心区域历史重关联 | 71.02% | 56.37% | 279,833 |

历史重关联相对既有修复提高总分、减少冲突，同时存在同一地毯被多个修复目标ID拆分等问题。当前保存为独立开发实验，尚不替换冻结P1-A1。下一步优先验证修复目标身份一致性和混合Mask。

当前统一开发评价协议：**`object_observed_repair / revision 2`**，配置 `unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json`。评分源码固定为 `cd260569`，审核归档为 `d035f687`，八场景共 **350 个固定可评价 GT**。此前 main 更新只提升默认入口；本分支保留相同评分规则和配置字节。

用于固定 TSDF 的实例地图质量和成对标签修复；当前仍为 DEVELOPMENT、`frozen=false`。协议选择、配置 SHA256、评分源码和逐场景 GT 锁见 `unified_eval/configs/current_protocol.json`，完整依据见[revision 2 审核报告](docs/evaluation_reports/20261009_object_observed_repair_audit/README.md)。旧 v1/v2/v3 和 revision 1 仅供历史复现。

```bash
# 显示唯一当前协议及其来源锁
python -m unified_eval.cli current-protocol

# 固定 GT 对应完整原生表面；P1 与 OVI 导出均走此入口
python -m unified_eval.cli adapt-surface \
  --gt /data/chenkejun/CVPR/results/v3_object_observed_repair_20261009/mesh/room0/gt.npz \
  --surface /path/to/native_surface.npz \
  --fixed-correspondence /path/to/new_result/fixed_gt_to_tsdf.npz \
  --method-name MY_METHOD --method-commit METHOD_COMMIT \
  --out /path/to/new_result/adapter

# 不传 --config 即使用 audit_r2.json；固定 GT 与 revision 2 预测必须一致
python -m unified_eval.cli eval-scene \
  --gt /data/chenkejun/CVPR/results/v3_object_observed_repair_20261009/mesh/room0/gt.npz \
  --pred /path/to/new_result/adapter/canonical_prediction.npz \
  --out /path/to/new_result/evaluation

python -m unified_eval.cli eval-repair-pair \
  --gt /data/chenkejun/CVPR/results/v3_object_observed_repair_20261009/mesh/room0/gt.npz \
  --before /path/to/before/canonical_prediction.npz \
  --after /path/to/after/canonical_prediction.npz \
  --out /path/to/new_result/paired
```

同一 P1 TSDF 的所有标签版本复用同一个对应缓存。OVI 几何不同，使用自己的缓存。当前入口拒绝旧配置、改过参数的配置、不同 GT 文件和改变过的评分源码；旧预测在 revision 2 下也会被既有评分校验拒绝。历史复现须显式加 `--historical-protocol`，不能据此将旧结果当成当前基准。

## 历史实现与复现说明

历史P0版本：**baseline + 部分未分配 TSDF 点扩散处理**。详见 [P0 版本说明](docs/P0_BRANCH_VARIANTS_CN.md)。

V3 标签修复的新增开发入口见 [object_observed_repair profile](docs/OBJECT_OBSERVED_REPAIR_V3_CN.md)：固定 GT 资格、可信可观测表面和完整 TSDF 几何对应，保留历史 V3 结果。

八场景开发重评及改动清单见 [2026-10-09 验收报告](docs/evaluation_reports/20261009_object_observed_repair/README.md)。

Unified evaluation code for 3D instance maps. The frozen **Replica-CA-v1** is preserved for old results. **Replica-CA-v2** adds independent per-instance projection, F1, significant merge/split/duplicate diagnostics, macro scene averages, and causal online-prefix evaluation. Its significant-overlap thresholds are still pending, so v2 cannot produce formal scores yet. These are custom Replica metrics, not official ScanNet AP. See [v2 metric definitions](docs/UNIFIED_EVALUATOR_V2_CN.md).

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m unified_eval.cli --help
```

Only NumPy and SciPy are needed for the evaluator. The ConceptGraphs adapter reads a trusted local `pcd_*.pkl.gz` map. Python pickle files can execute code when loaded; do not use the adapter on untrusted downloads.

## Historical: evaluate a ConceptGraphs map

Prepare a Replica reference dataset, export its GT, and create an experiment manifest with exactly one entry matching the map and scene. Each entry needs `scene`, `map`, `cost.frames`, and `input.start`, `input.end`, `input.stride`. The frame count must equal `len(range(start, end, stride))`.

```bash
python -m unified_eval.cli export-replica-gt \
  --historical-protocol \
  --reference-root /path/to/reference --scene room0 --out /path/to/gt.npz

python -m unified_eval.cli adapt-conceptgraphs \
  --historical-protocol \
  --config unified_eval/configs/replica_ca_v1.json \
  --gt /path/to/gt.npz --map /path/to/pcd_map.pkl.gz \
  --experiment-manifest /path/to/experiment_manifest.json \
  --method-name my_method --method-commit COMMIT_SHA \
  --out /path/to/output/adapter

python -m unified_eval.cli eval-scene \
  --historical-protocol \
  --config unified_eval/configs/replica_ca_v1.json \
  --gt /path/to/gt.npz \
  --pred /path/to/output/adapter/canonical_prediction.npz \
  --out /path/to/output/evaluation
```

`metrics.json`, `overlap_matrix.npz`, and `manifest.json` are written under the evaluation output. The adapter writes `canonical_prediction.npz`, `adapter_stats.json`, and `adapter_manifest.json`. `eval-batch` accepts a JSON object with `scenes: [{"gt": "...", "prediction": "..."}]`.

The frozen config uses a maximum projection distance of 0.05 m and a minimum of 100 valid vertices per instance. Supply your own reference root at export time. The public config replaces server-specific provenance paths with generic descriptions, so its file hash differs from the original internal config while the metric parameters remain the same. `replica_ca_v1.pending.json` is an unfinished draft and is not the frozen protocol.

## V2 development use

Use `unified_eval/configs/replica_ca_v2.pending.json`. Its mapping distance, GT minimum size, and significant-overlap thresholds are all `null` until calibrated. Debug commands must supply `--debug-max-distance-m DELTA --debug-min-valid-instance-vertices M --debug-significant-min-vertices N --debug-significant-min-gt-fraction ALPHA`. Outputs are marked `DEBUG_ONLY / NON_OFFICIAL` and formal `summary.csv` is withheld. Re-adapt native maps under v2; v1 canonical predictions are intentionally rejected by a v2 protocol. The historical debug choice `DELTA=0.05` fails the real-GT oracle test and must not be used for ranking.

Run `scripts/replica_projection_calibration.py` on the existing reference mesh with `--scenes room0 room1 office0 --perturb-scenes room0 --debug-min-valid-instance-vertices 100` to reproduce GT-only distance, subset, noise, and density sensitivity. This is a diagnostic setting, not a frozen GT-size threshold.

`eval-online-prefix` needs an online JSON manifest with the original experiment manifest, its method, every source frame in order, per-frame depth/intrinsics/pose `.npy` paths, and ordered checkpoints with committed canonical predictions. Each prediction must record `committed_frame_id` and `max_input_frame_id`. The evaluator constructs observed GT from only the depth frames received through each checkpoint and writes `prefix_curve.csv` and provenance. An example and exact field definitions are in the v2 metric document.

`adapt-ovimap` accepts an OVI-MAP export with `xyz`, compact `instance` owner indices, and `native_instance_ids`, plus its export JSON and input-alignment JSON. It verifies the recorded frame list and maps each native instance independently under the same v2 protocol. The class labels are retained only as metadata; class-agnostic scores ignore them.

The `repair.py` module remains unchanged. Repair Success, False Repair, and Repair Delay are not yet implemented as metrics. Official ScanNet evaluation remains separate and unchanged.

## 历史统一 v3 评估（2026-10-07）

已核验 P1-A1、OVI-MAP、OVO 和 OpenVox 六种方法／版本共 48 组场景评估，补充 19 项覆盖、召回和结构指标。完整数值、逐场景数据、定义和核验见[历史评估报告](docs/evaluation_reports/20261007/README.md)。OVO、OpenVox 的原生输入预算与主比较不同，作为参考；全部保持 `DEBUG_ONLY / NON_OFFICIAL` 标记。这些历史分数不属于当前 revision 2，不能和 350-GT 当前结果直接混用。
