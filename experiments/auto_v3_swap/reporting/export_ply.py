from pathlib import Path
import colorsys,hashlib,json
import numpy as np
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                ('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
assert dtype.itemsize==24
for scene in ['office1','office3','office4']:
    r=R/scene
    if not (r/'complete.json').exists():continue
    out=r/'ply';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        for row in read(out/'manifest.json')['exports']:assert sha(out/row['file'])==row['sha256']
        continue
    comparison=read(r/'comparison.json');assert comparison['status']=='PASS'
    reports=[];count=comparison['new_count']
    for variant,key,folder in [('E1','new_E1',r/'e1_repair/e1/E1_expanded_history'/scene),
                                ('partial_E1','new_partial',r/'partial/whole_to_partial'/scene)]:
        source=folder/'final/instance_surface.npz'
        assert sha(source)==comparison['source_locks'][key+'_final']['surface_sha256']
        z=dict(np.load(source));e=dict(np.load(folder/'native/surface_evidence.npz'))
        np.testing.assert_array_equal(z['xyz_m'],e['xyz_m']);np.testing.assert_array_equal(z['rgb'],e['rgb'])
        labels=z['instance_id'];state=e['state'];xyz=z['xyz_m']
        metric=next(x for x in comparison['rows'] if x['variant']==key and x['stage']=='final')
        assert int((labels<=0).sum())==metric['gray_total']
        for kind,sel in [('instances',labels>0),('three_states_black_yellow_red',labels<=0)]:
            idx=np.flatnonzero(sel);a=np.empty(len(idx),dtype)
            for j,n in enumerate(['x','y','z']):a[n]=xyz[idx,j]
            a['instance_id']=labels[idx];a['state']=state[idx];a['surface_index']=idx
            colors=np.zeros((len(idx),3),np.uint8)
            if kind=='instances':
                for pid in np.unique(labels[idx]):
                    colors[labels[idx]==pid]=np.array(colorsys.hsv_to_rgb((int(pid)*.61803398875)%1,.67,.94))*255
            else:
                assert set(np.unique(state[idx]))<={0,1,3}
                colors[state[idx]==1]=[255,255,0];colors[state[idx]==3]=[255,0,0]
            for j,n in enumerate(['red','green','blue']):a[n]=colors[:,j]
            path=out/f'{scene}_auto{count}_swap34_{variant}_{kind}.ply'
            header=('ply\nformat binary_little_endian 1.0\n'
                'comment same fixed TSDF; no resampling; original surface_index preserved\n'
                f'element vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\n'
                'property uchar state\nproperty int surface_index\nend_header\n').encode()
            if not path.exists():
                with path.open('xb') as f:f.write(header);a.tofile(f)
            b=np.memmap(path,dtype=dtype,mode='r',offset=len(header),shape=(len(a),))
            np.testing.assert_array_equal(a,b);assert path.stat().st_size==len(header)+len(a)*24
            reports.append(dict(file=path.name,vertices=len(a),sha256=sha(path),source=str(source),
                source_sha256=sha(source),variant=variant,kind=kind,readback_exact=True))
    (out/'manifest.json').write_text(json.dumps(dict(status='PASS',scene=scene,exports=reports),indent=2))
    print('EXPORTED_VERIFIED',scene,4,flush=True)
