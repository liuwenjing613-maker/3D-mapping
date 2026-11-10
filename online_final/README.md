# online-final (2026-11-10)
User-approved frozen Replica ONLINE method. Offline main and R2 evaluator remain unchanged (evaluation frozen=false).
## Exact executed source
source/experiments/ppt5_staged_postid_full_depth_20261110 contains the actual 2241-file source snapshot, not a new implementation. The directory name is historical; all five rounds and all eight scenes use identical rules. Supporting helper, gate, scheduler, original mapping and frozen E1 sources are included. source_manifest.json records deployed paths and SHA256.
Sources intentionally retain validated server paths. Model repos SAM3, CropFormer and SAM2.1 (legacy dependencies), environments, datasets, full-depth TSDF and weights remain external on the authorized server. This is a frozen research deployment, not a portable cold-start installer.
## Verification
Run: python3 online_final/verify_frozen.py --deployed --audit-results
Deployment: /home/chenkejun/CVPR/experiments/ppt5_staged_postid_full_depth_20261110
Results: /data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110
## Executed entry points
run_first50.py SCENE performs first-round candidates/tracking/gate/repair/commit after certified first-50 unrepaired mapping preparation. controller.py SCENE waits for that commit and performs 100/200/300/400 causal mapping and actual repairs, then finalB and history DBSCAN. evaluate_and_compare.py uses current main R2; monitor.py reports status.
Do not rerun into frozen completed results. A new run requires isolated code/result roots, path relocation and initialization. The initial unrepaired 50-frame mapping was reused after verification; later mapping follows this method's own repaired state. This is not filter-only replay.
## Frozen rules and results
Dice strictly >0.25, inherited support >=100, new support >=3500. Unique-ID competition after final same-view deduplication. Uniform across all five rounds and eight scenes. SAM3, original B commit, finalB and PPT history DBSCAN.
40 actual commits, 8 final evaluations complete. Exact matched full-depth geometry. F1/AP50 0.6621067031463749 / 0.44021725007146384 versus PPT 0.6550335570469799 / 0.4281238250407318. AP/PQ improve; mCov 0.6249989767901398 -> 0.6213585848031344. TP244 ->242, FP151 ->139.
Five scenes improve; room1/office0/office4 regress. office4 F1/AP50 -6.58/-8.08pp. SAM3 versus original SAM2.1 means this is a full-method comparison, not a threshold-only ablation.
evidence/comparison.json contains pooled, scene and GT metrics. Historical audit decision default=PPT predates user adoption; frozen.json records subsequent approval.
