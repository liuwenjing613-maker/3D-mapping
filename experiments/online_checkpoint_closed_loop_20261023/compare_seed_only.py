from pathlib import Path
from collections import Counter
import json
from compare_tracking import seed_records
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
a,b=[seed_records(OUT/m,True) for m in ['reference','checkpoint']]
keys=lambda rows:Counter((r['frame'],r['mask_sha256']) for r in rows)
r=dict(status='PASS' if a==b else 'DIFFERENT',counts=[len(a),len(b)],ordered_seed_sources_exact=a==b,
       frame_mask_multisets_exact=keys(a)==keys(b),duplicate_counts=[sum(v-1 for v in keys(rows).values()) for rows in [a,b]])
(OUT/'seed_comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
