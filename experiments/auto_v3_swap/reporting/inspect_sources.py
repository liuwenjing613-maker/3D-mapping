from pathlib import Path
import json
B=Path('/data/chenkejun/CVPR/results')
for scene in ['room0','room2','office1']:
    r=B/'auto_v2_replica8_partial_20261019'/scene
    print(scene,[p.name for p in (r/'evaluation_r2').iterdir()])
    d=json.loads((r/'evaluation_r2/comparison.json').read_text())
    print('comparison',list(d),list(d['rows'][0]))
    print('freeze',json.dumps(json.loads((r/'prediction_freeze.json').read_text()))[:1200] if (r/'prediction_freeze.json').exists() else 'none')
    p=r/'whole_to_partial'/scene
    print('stage files', [x.name for x in (p/'final').iterdir()])
    print('native files',[x.name for x in (p/'native').iterdir()])
    p=r/'evaluation_r2/baseline_final/adapter_manifest.json'
    print('adapter keys',list(json.loads(p.read_text())))
print('available_RAM_GB',int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2)
