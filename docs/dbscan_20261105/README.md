# Guarded DBSCAN terminal processing

This update records the existing server implementation without changing algorithms,
thresholds, running experiments, frozen E1, or evaluator defaults.

## Processing
Voxelized DBSCAN support identifies merge candidates. Independent original-mask
co-visibility and independently accepted repair objects prohibit merging.
Every pair in a merged family must pass; transitive chains alone are insufficient.
Gray-point completion uses local geometry, normals and observation votes.
Historical completion only uses legitimate retained committed repair lineage.
ScanNet++ no-repair and final-only offline main do not fabricate historical cores.
No TSDF vertices are deleted, no new identities are invented, GT never drives
decisions, and terminal processing does not feed back into online mapping.
Native vote rule (at least 2 votes and 67%) and final B exactly once remain.

## Sources and entry points
Replica: experiments/unified_full_depth_20261101/ablations/dbscan_identity_completion_guarded_20261102.
Current co-visible entry: run_history_covisible.py. Earlier guarded/history variants
are preserved as dependencies and historical comparison implementations.
ScanNet++: experiments/scannetpp_v2_unified_dbscan_20261103/run_dbscan_scene.py VARIANT SCENE.
Variants: no_repair, main, online_A, online_B.
Use pool_native_variant.py VARIANT for paired two-scene pooled evaluation.
Do not call legacy dbscan_common.run(eval-batch); scannetpp_eval_batch.py provides
the current-main scope adaptation.

## Deployment prerequisites
These are server-bound research scripts, not a standalone portable installation.
Absolute paths intentionally match /home/chenkejun/CVPR and /data/chenkejun/CVPR.
They require existing native mapping snapshots, source_lock.json, plan.json,
source policy guards, frozen input receipts, scene-completion dependencies,
current main evaluator and original datasets. Runtime artifacts, datasets,
weights and results are not uploaded. The source manifest records byte-identical
copies of deployed files; this checkout alone does not recreate the environment.

ScanNet++ uses its locked 31+39=70 GT scope and native correspondence.
Replica uses separate current-main R2 350-GT scope. Evaluation remains frozen=false;
neither is an official ScanNet++ leaderboard result.
Do not overwrite completed outputs or use assigned-view PLYs for scoring.
