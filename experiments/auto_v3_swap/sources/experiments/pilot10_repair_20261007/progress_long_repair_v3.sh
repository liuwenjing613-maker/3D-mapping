set -eu
/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/long_tracking_20261007/repair_v3_20261007/room2-14047feb0df3aaa5')
for name in ['status.json','failure.json']:
    f=p/name
    if f.exists():print(name,f.read_text())
if (p/'run.log').exists():
    lines=(p/'run.log').read_text(errors='replace').splitlines()
    print('LATEST LOG:\n'+'\n'.join(lines[-4:]))
PY
