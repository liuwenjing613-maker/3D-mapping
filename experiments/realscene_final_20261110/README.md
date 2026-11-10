# ScanNet real-scene configuration snapshot

This branch records the actual 400-frame no-repair, offline main and online-v2 B results for scene0050_00, scene0011_00 and scene0518_00. The user requested configuration is frozen here; the evaluation GT is provisional and the locked Replica-CA-v3 protocol remains unfrozen.

- Offline main runs one actual repair after frame 400. Online-v2 B executes actual repairs at frames 100, 200 and 400.
- The filter is `split_new3500_inherit100`; the exact parameters and per-scene result/provenance hashes are in `frozen_config_and_results.json`.
- All reported outputs include final B and guarded DBSCAN merge/fill. The actual guarded DBSCAN source is preserved in `source/dbscan_guarded.py`; it differs from the historical ScanNet++ snapshot already in the repository.
- The R2 score uses IoU > 0.5. IoU > 0.4 is an additional sensitivity analysis only. The augmented GT contains assumed labels and RGB-D point-splat visibility; it is not the canonical Replica 350 GT.

`source_snapshot.tar.gz` preserves selected text source and configurations from the scene-specific execution trees; `source_manifest.json` records individual SHA-256 hashes. No RGB-D frames, GT, pretrained weights, environment, checkpoints, predictions or credentials are included. Existing absolute server paths and external dependencies remain, so this snapshot is not a standalone installation or a guarantee of reproducibility on another host.
