set -eu
/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/joint_mask_replacement_20261007/room2_mask14_mask24')
for name in ['tracking/status.json','tracking/failure.json','status.json','failure.json']:
    q=p/name
    if q.exists():
        d=json.loads(q.read_text())
        if 'traceback' not in d:d={k:v for k,v in d.items() if k in ['status','phase','completed_frames','total_frames','elapsed_seconds','completed_mapping_frames','accepted_mapping_frames','strict_success']}
        print(name,json.dumps(d,ensure_ascii=False))
PY
