from pathlib import Path
R=Path('/data/chenkejun/CVPR/results/e1_roi_top20_20261015');A=Path('/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015')
for p in [R/'render_annotation.log',R/'finalize_projections.log',R/'pipeline_error.json',R/'pipeline_complete.json',A/'score.log']:
 if p.exists():print(p.name, '\n'.join(p.read_text().splitlines()[-5:]))
for p in [R/'delivery_receipt.json',A/'audit.json']:
 if p.exists():print('READY',p.name)
