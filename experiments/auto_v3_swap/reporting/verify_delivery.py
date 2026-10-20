from pathlib import Path
import colorsys,hashlib,json,urllib.request
import numpy as np
W=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
P=W/'results/固定案例_三模型对比_20261006/replica8_auto_E1_20261019'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=read(P/'swap34/data.json');assert d['status']=='PASS'
assert len(d['scenes'])==8 and sum(s['GT_count'] for s in d['scenes'])==d['GT_count']==350
assert sum(len(s['old']['rows']) for s in d['scenes'])==637
assert sum(s['new_count'] for s in d['scenes'])==196
assert sum(len(s['rows']) for s in d['scenes'])==80
assert read(O/'selection_audit.json')['status']=='PASS'
checks=[];files={};images=set()
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                ('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
for s in d['scenes']:
    assert s['status']=='PASS' and s['same_R2_geometry_GT_correspondence_verified']
    assert len(s['rows'])==10 and len(s['per_GT'])==s['GT_count']
    kept=set(s['selection']['retained_group_ids']);assert len(kept)==s['new_count']
    tids={r['original_track_id']:r['group_id'] for r in s['old']['rows']}
    actual={t for row in s['association']['objects'] for t in row['member_track_ids']}
    assert {tids[t] for t in actual}==kept
    for row in s['rows']:assert sum(row['gray_states'].values())==row['gray_total']
    for r in s['old']['rows']:
        for image in r['images'].values():assert (P/image).exists();images.add(image)
    for variant in ['baseline','new_E1','new_partial']:
        links=[x for x in s['ply'] if x['variant']==variant]
        assert len(links)==2 and {x['kind'] for x in links}=={'instances','three_states_black_yellow_red'}
    for item in s['ply']:
        p=P/item['url'];assert sha(p)==item['sha256'];files[item['url']]=item['sha256']
        if item['variant'] not in ['new_E1','new_partial']:continue
        with p.open('rb') as f:
            header=b''
            while not header.endswith(b'end_header\n'):header+=f.readline()
        count=int(next(x.split()[-1] for x in header.decode().splitlines() if x.startswith('element vertex')))
        assert count==item['vertices'] and p.stat().st_size==len(header)+count*24
        a=np.memmap(p,dtype=dtype,mode='r',offset=len(header),shape=count)
        assert np.all(a['surface_index'][1:]>a['surface_index'][:-1])
        colors=np.column_stack([a[n] for n in ['red','green','blue']])
        if item['kind']=='instances':
            assert np.all(a['instance_id']>0)
            for pid in np.unique(a['instance_id']):
                expected=(np.array(colorsys.hsv_to_rgb((int(pid)*.61803398875)%1,.67,.94))*255).astype(np.uint8)
                assert np.all(colors[a['instance_id']==pid]==expected)
        else:
            assert np.all(a['instance_id']<=0)
            metric=next(r for r in s['rows'] if r['variant']==item['variant'] and r['stage']=='final')
            for state,name,color in [(0,'U',[0,0,0]),(1,'T',[255,255,0]),(3,'C',[255,0,0])]:
                mask=a['state']==state
                assert int(mask.sum())==metric['gray_states'][name]
                assert np.all(colors[mask]==color)
    checks.append(dict(scene=s['scene'],selected=len(kept),metrics=10,GT_count=s['GT_count'],
        ID_changes=len(s['identity_changes']),PLY_links=len(s['ply']),status='PASS'))
for summary in d['summary']:
    rows=[next(r for r in s['rows'] if r['variant']==summary['variant'] and r['stage']==summary['stage']) for s in d['scenes']]
    for k in ['F1','PQ','AP','AP50','mCov']:assert abs(sum(r[k] for r in rows)/8-summary[k])<1e-14
    for k in ['TP','FP','FN','gray_total']:assert sum(r[k] for r in rows)==summary[k]
old=read(P/'previous_metrics_manifest.json');unchanged=[]
for name,digest in old['files'].items():
    if name=='index.html':continue
    assert sha(P/name)==digest,name
    unchanged.append(name)
url='http://127.0.0.1:8790/replica8_auto_E1_20261019/'
for name in ['replica8_swap34.html','swap34/data.json','swap34/metrics.csv',*files]:
    with urllib.request.urlopen(urllib.request.Request(url+name,method='HEAD'),timeout=30) as response:
        assert response.status==200,name
out=dict(status='PASS',scenes=checks,GT_count=350,all_metric_rows=80,v1_seeds=637,new_retained=196,
    image_assets_checked=len(images),PLY_files_checked=len(files),PLY_hashes=files,
    original_metric_files_unchanged=len(unchanged),web_links_HTTP_200=True)
(O/'delivery_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='PLY_hashes'},ensure_ascii=False,indent=2))
