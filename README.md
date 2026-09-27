# 3D-mapping

Unified evaluation code for 3D instance maps. The frozen **Replica-CA-v1** is preserved for old results. **Replica-CA-v2** adds independent per-instance projection, F1, significant merge/split/duplicate diagnostics, macro scene averages, and causal online-prefix evaluation. Its significant-overlap thresholds are still pending, so v2 cannot produce formal scores yet. These are custom Replica metrics, not official ScanNet AP. See [v2 metric definitions](docs/UNIFIED_EVALUATOR_V2_CN.md).

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m unified_eval.cli --help
```

Only NumPy and SciPy are needed for the evaluator. The ConceptGraphs adapter reads a trusted local `pcd_*.pkl.gz` map. Python pickle files can execute code when loaded; do not use the adapter on untrusted downloads.

## Evaluate a ConceptGraphs map

Prepare a Replica reference dataset, export its GT, and create an experiment manifest with exactly one entry matching the map and scene. Each entry needs `scene`, `map`, `cost.frames`, and `input.start`, `input.end`, `input.stride`. The frame count must equal `len(range(start, end, stride))`.

```bash
python -m unified_eval.cli export-replica-gt \
  --reference-root /path/to/reference --scene room0 --out /path/to/gt.npz

python -m unified_eval.cli adapt-conceptgraphs \
  --config unified_eval/configs/replica_ca_v1.json \
  --gt /path/to/gt.npz --map /path/to/pcd_map.pkl.gz \
  --experiment-manifest /path/to/experiment_manifest.json \
  --method-name my_method --method-commit COMMIT_SHA \
  --out /path/to/output/adapter

python -m unified_eval.cli eval-scene \
  --config unified_eval/configs/replica_ca_v1.json \
  --gt /path/to/gt.npz \
  --pred /path/to/output/adapter/canonical_prediction.npz \
  --out /path/to/output/evaluation
```

`metrics.json`, `overlap_matrix.npz`, and `manifest.json` are written under the evaluation output. The adapter writes `canonical_prediction.npz`, `adapter_stats.json`, and `adapter_manifest.json`. `eval-batch` accepts a JSON object with `scenes: [{"gt": "...", "prediction": "..."}]`.

The frozen config uses a maximum projection distance of 0.05 m and a minimum of 100 valid vertices per instance. Supply your own reference root at export time. The public config replaces server-specific provenance paths with generic descriptions, so its file hash differs from the original internal config while the metric parameters remain the same. `replica_ca_v1.pending.json` is an unfinished draft and is not the frozen protocol.

## V2 development use

Use `unified_eval/configs/replica_ca_v2.pending.json`. Supply both significant-overlap thresholds explicitly while debugging, for example `--debug-significant-min-vertices N --debug-significant-min-gt-fraction ALPHA` on `adapt-conceptgraphs`, `eval-scene`, `eval-batch`, or `eval-online-prefix`. The placeholder `N` and `ALPHA` must be chosen for that development run; neither is a formal v2 value. Outputs are marked `DEBUG_ONLY / NON_OFFICIAL` and formal `summary.csv` is withheld. Re-adapt native maps under v2; v1 canonical predictions are intentionally rejected by a v2 protocol.

`eval-online-prefix` needs an online JSON manifest with the original experiment manifest, its method, every source frame in order, per-frame depth/intrinsics/pose `.npy` paths, and ordered checkpoints with committed canonical predictions. Each prediction must record `committed_frame_id` and `max_input_frame_id`. The evaluator constructs observed GT from only the depth frames received through each checkpoint and writes `prefix_curve.csv` and provenance. An example and exact field definitions are in the v2 metric document.

The `repair.py` module remains unchanged. Repair Success, False Repair, and Repair Delay are not yet implemented as metrics. Official ScanNet evaluation remains separate and unchanged.
