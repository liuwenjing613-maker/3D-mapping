from pathlib import Path
import json, shutil, subprocess
B = Path('/data/chenkejun/CVPR/results')
for name in ['auto_v2_replica6_20261019/office1', 'auto_v2_replica6_20261019/office3', 'auto_v2_replica6_20261019/office4', 'room1_swap34_20261020']:
    r = B/name
    for f in ['v2/complete.json', 'e1_repair/evaluation_r2/comparison.json']:
        p=r/f
        v=json.loads(p.read_text())
        print(name, f, v.get('status'), p.stat().st_size)
print('disk_free_GB', round(shutil.disk_usage(B).free/1e9))
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
print(subprocess.run(['ps','-eo','pid,etimes,args'],capture_output=True,text=True).stdout[-1000:])
