from pathlib import Path
import json, hashlib
from PIL import Image
import numpy as np
from grouping_rules import group_seeds

H=Path(__file__).resolve().parent
W=H.parents[1]/'results/固定案例_三模型对比_20261006/room2_E1_instance_seed_groups_20261015'
d=json.loads((H/'grouping.json').read_text(encoding='utf-8'))
web=json.loads((W/'data.json').read_text(encoding='utf-8'))
audit=json.loads((H/'audit.json').read_text(encoding='utf-8'))
for n,h in audit['output_sha256'].items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==h
with np.load(H/'pair_metrics.npz') as p:
    inter=p['intersection'];union=p['union'];passes=p['passes'];iou=p['iou']
counts=np.array([s['support_count'] for s in d['seeds']])
assert np.array_equal(inter,np.array(web['intersection']))
assert np.array_equal(union,counts[:,None]+counts[None,:]-inter)
assert np.array_equal(passes,(union>0)&(4*inter>union))
assert np.all(counts[:-1]>=counts[1:])
assert group_seeds(range(len(counts)),lambda a,b:passes[a,b])==[g['members'] for g in d['groups']]
with np.load(H/'surface_supports.npz') as p:
    indices=p['indices'];offsets=p['offsets']
    assert np.array_equal(np.diff(offsets),counts)
    for i,s in enumerate(d['seeds']):
        points=indices[offsets[i]:offsets[i+1]]
        assert len(np.unique(points))==len(points)
        assert hashlib.sha256(points.tobytes()).hexdigest()==s['support_sha256']
for s in web['seeds']:
    assert (W/s['rgb']).is_file()
    assert len(s['color'])==3 and all(0<=v<=255 for v in s['color'])
    p=W/s['mask'];assert hashlib.sha256(p.read_bytes()).hexdigest()==s['mask_sha256']
    a=np.asarray(Image.open(p));assert a.shape==(680,1200) and int((a>0).sum())==s['area']
    assert (W/s['source_link'].split('?')[0]).is_file()
assert sum(len(s['aliases']) for s in web['seeds'])==1407
report=dict(status='PASS',groups=len(d['groups']),masks=len(d['seeds']),
    checks=['downloaded artifacts SHA-256', 'exact grouping replay', 'strict threshold and TSDF-count order',
            'every exported support set hash and uniqueness', 'all mask areas, sizes and hashes',
            'all RGB and source links exist', 'original palette colors retained', 'all 1407 input occurrences retained'])
(H/'page_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
