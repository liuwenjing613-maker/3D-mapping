# Replica paper final: online-final + trusted region
User approved this result as the final Replica method on 2026-11-10. This branch's current online method includes the trusted-region policy. It supersedes online-final@b7de6eea for Replica; offline main, other datasets and the R2 evaluator are unchanged.
## Frozen result
Eight scenes, final 400 frames, main object_observed_repair / revision 2, 350 GT, evaluation frozen=false. Algorithm frozen=true.
| Metric | Previous online-final | Paper final |
|---|---:|---:|
| F1 | 0.6621067031463749 | 0.671251719394773 |
| AP50 | 0.44021725007146384 | 0.44856475037424165 |
| AP | 0.23015774018346719 | 0.2346114452294036 |
| PQ | 0.5335282881166024 | 0.5397431056267806 |
| mCov | 0.6213585848031344 | 0.624668352880015 |
TP 242->244; FP139->133; FN108->106. Full scene/GT evidence is in evidence/trusted_final/comparison.json. office4 improves strongly; office3 regresses (F1 -3.85pp, AP50 -4.34pp). Do not describe every scene as improved.
## Complete method
Full-depth TSDF and checkpoints 50/100/200/300/400. SAM3 tracking. Uniform filter ppt5-full-depth-postid-uniform025-new3500-v1: Dice strictly >0.25, inherited support >=100, new support >=3500, unique ID competition recomputed after final same-view deduplication. Preserve the frozen original gate, partial repair, commit and finalB numerics.
From frame51 onward, use only previously committed repair regions. Query trusted owners within 1.5cm; significant support requires >=10 distinct points and >=5% of projected mask points. Multiple significant owners split a mask; one owner conflicting with the fitted ID abstains; local conflicts/ambiguity abstain locally; otherwise retain original association. Per-owner observations replace indivisible observations and rebuild the associator index before the next frame. Native confirmation remains >=2 votes and >=0.67 confidence.
After actual 400-frame repair/commit, run finalB plus the same history-protected DBSCAN. Terminal processing does not feed back into online mapping. No GT or future repair inputs enter online decisions.
## Executed source and entry points
source/experiments/online_final_trusted_20261110 is the exact executed method, including all eight scene branches. source_manifest.json maps every stored file to its deployed path and SHA256; deployment_symlinks records helper/config dependencies. The earlier uniform method and shared dependencies remain archived under source/experiments.
- branches/SCENE/trusted_online.py and trusted_partition.py: owner projection, partition/abstention and association feedback.
- map_continue.py, map_support.py, scene_continue.py, commit_continue.py in each branch: causal continuation, trusted prefix access and commits.
- controller.py SCENE: checkpoints100/200/300/400 after first50 commit; actual repair then finalB and terminal DBSCAN.
- run_first50.py SCENE: first repair after prepared initial mapping; not rerun in this experiment.
- terminal_dbscan.py: final processing with explicit retained-label provenance checks.
- evaluate_and_compare.py: current main R2 evaluation and exact old online-final baseline verification.
- audit_uniform_actual_commits.py --require-complete: all40 commit/rule/geometry/ID audit.
Actual deployment: /home/chenkejun/CVPR/experiments/online_final_trusted_20261110
Actual results: /data/chenkejun/CVPR/results/online_final_trusted_20261110
This experiment reused eight verified immutable first50 repair commits from b7de6eea and newly executed the subsequent32 commits. It is real online continuation, not filter replay and not40 freshly rerun repairs.
## Verify
Run python3 online_final/verify_frozen.py --deployed --audit-results from this repository on the authorized server.
The snapshot preserves validated absolute paths and is not a portable cold-start installer. External model repos, environments, weights, Replica RGB-D, TSDF geometry and initial/checkpoint caches remain on the server. A new run needs independent code/output roots and path relocation; never run entry points into immutable completed results.
## room1 validation fix
At400, repair removed ID48's current votes but preserved21 pre-existing spatial labels outside the repair region. The old all-final-labels-in-current-votes assertion incorrectly treated retained labels as new assignments. The fix verifies those labels exactly against pre-repair, committed and finalB surfaces outside the repair owner/core, retains every assigned point unchanged, and requires every newly filled label to have current votes. It changes no numeric labels or thresholds and invents no IDs. All eight scenes pass these invariants; only room1 has21 such retained points. Source patch, regression cases and final audits are archived.
## Provenance
Historical pre-adoption reports state default_method_updated=false; frozen.json is the subsequent explicit user adoption. The previous method and README are preserved under history/pre_trusted_b7de6eea and in commit b7de6eea.
