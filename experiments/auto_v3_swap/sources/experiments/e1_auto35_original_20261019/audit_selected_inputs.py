"""Check the complete prepared input against the user-confirmed 35 original masks."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/e1_auto35_original_20261019')
S=Path('/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
selection=read(C/'confirmed_selection.json');selected={x['original_track_id']:x for x in selection['included']};seen=[]
assert len(selected)==35 and selection['pr_filter_applied']
for row in read(R/'seed_diagnostics.json')['cases']:
    cfg=read(row['config_path']);ids={x['track_id'] for x in cfg['objects']}
    labels=np.asarray(Image.open(R/'cases'/row['case_uid']/'prepared_seed.png'))
    assert set(np.unique(labels))==ids|{0} and ids<=set(selected)
    for tid in ids:
        rec=selected[tid];p=S/rec['mask_relative_path']
        assert sha(p)==rec['mask_sha256'] and cfg['seed_frame']==rec['frame']
        np.testing.assert_array_equal(labels==tid,np.asarray(Image.open(p))>0)
        seen.append(tid)
assert sorted(seen)==sorted(selected)
report=dict(status='PASS',exact_original_seed_masks=35,cases=len(read(R/'seed_diagnostics.json')['cases']),original_ids_preserved=True,user_exceptions_applied=True,
            excluded_by_previous_containment=['G080','G081','G085'],user_retained_exceptions=['G017','G025'],excluded_this_round=selection['excluded_this_round'],selection_sha256=sha(C/'confirmed_selection.json'))
assert not {69,71,74,75,77,80,81,82,85,86}&set(selected) and {17,25}<=set(selected)
(R/'selected_input_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
