from pathlib import Path
import json,numpy as np
src=Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019/e1_repair');read=lambda p:json.loads(Path(p).read_text())
for c in read(src/'seed_diagnostics.json')['cases']:
 if any(x['track_id'] in [8,40] for x in c['objects']):
  cfg=read(c['config_path']);print(json.dumps(dict(case=c['case_uid'],old_family_ids=cfg['old_family_ids'],objects=[o for o in c['objects'] if o['track_id'] in [8,40]])))
