office0 Mask selection viewer — 2026-10-25

Local page: http://127.0.0.1:8801/office0_masks/index.html
Scope: office0, A/B, checkpoints 50/100/200/300/400.
The exporter only reads existing completed experiment outputs. No inference,
selection, tracking, geometry, or evaluation is rerun or modified.

Seven display stages:
1. Top20 ROI entries for each of CONFLICT/U/T and their views.
2. All original CropFormer / crop CropFormer / SAM candidates, including rejects.
3. Each occurrence's ROI intersection, source grouping and actual choice.
4. Real all_edge boundary and morphology filtering after grouping_r2 input.
5. Primary grouping, semantic, co-visible IoU, containment, skip repair and P/R.
6. Actual final seed masks, exact source aliases, auxiliary group members.
7. Recorded SAM2/Gate acceptance and persistent identities.

Final execution uses all_edge, not the abandoned earlier diagnostic branches.
Candidate occurrences, unique candidates, deduplicated masks, and primary groups
are different counting units. The page retains these distinctions.
Only window has the special SAM source policy; cabinet uses ordinary selection.
Window/cabinet P/R exemptions are preserved. skip_repair_B is a rule name,
not the experimental branch B.

export_selection.py runs on the existing authorized server with the numerical
Python environment. It writes image assets and compressed JSON without changing
the source artifacts, then checks the source hashes again. RGB label atlases
encode label = R + 256*G and are checked by a pixel-exact round trip.

install_page.py safely extracts the archive and verifies every file.
verify_extracted.py verifies an already extracted directory and final seed sets.
The initial long installation was interrupted after extraction; verification was
completed separately using verify_extracted.py. No numerical job was interrupted.

local_verification.json verifies all archive assets and ten final seed sets.
browser_verification.json records ten-run UI loading and representative pixel,
source competition, boundary, morphology, P/R, ROI navigation, and Gate checks.
The page's binary-mask download preserves all original foreground pixels.

The displayed data describe this completed experiment only. This viewer does not
resolve the previous original/optimized closed-loop equivalence limitation.
