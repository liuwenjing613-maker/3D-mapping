from pathlib import Path
import json,time
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
for scene in ['office1','office3','office4']:
    root=R/scene
    d=dict(scene=scene)
    for name in ['status.json','partial/progress.json']:
        p=root/name
        if p.exists():
            v=json.loads(p.read_text());d[name]={k:v[k] for k in ['status','stage','frames','total_frames','variant','seconds'] if k in v}
    for name in ['e1_repair/history_core.log','e1_repair/E1.log']:
        p=root/name
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-700));tail=f.read().decode(errors='replace')
            d[name]=tail.splitlines()[-1][-240:]
    print(json.dumps(d),flush=True)
