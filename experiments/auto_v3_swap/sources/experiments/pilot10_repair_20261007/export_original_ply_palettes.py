"""Read the original P1-A1 PLY colors without changing any geometry or labels."""
from pathlib import Path
import json, hashlib, numpy as np
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
OUT=ROOT/'tracking_review'
manifest=json.loads((OUT/'ply_models/manifest.json').read_text())
types={'float':'<f4','double':'<f8','uchar':'u1','int':'<i4','uint':'<u4','short':'<i2','ushort':'<u2'}
scenes={}
for scene in sorted({c['scene'] for c in manifest['cases']}):
    p=BASE/f'final/{scene}/P1-A1/diffusion/{scene}_repaired_full.ply'
    with p.open('rb') as f:
        header=[]
        while True:
            line=f.readline().decode('ascii').strip();header.append(line)
            if line=='end_header':break
        props=[];count=None;vertex=False
        for line in header:
            parts=line.split()
            if parts[:2]==['element','vertex']:count=int(parts[2]);vertex=True
            elif parts and parts[0]=='element':vertex=False
            elif vertex and parts[0]=='property':props.append((parts[2],types[parts[1]]))
        a=np.fromfile(f,np.dtype(props),count=count)
    source=BASE/f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
    expected=next(c for c in manifest['cases'] if c['scene']==scene)['source_maps'][str(source)]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    with np.load(source) as z:
        np.testing.assert_array_equal(np.column_stack([a[k] for k in ['x','y','z']]),z['xyz_m'])
        np.testing.assert_array_equal(a['instance_id'],z['instance_id'])
    colors=np.column_stack([a[k] for k in ['red','green','blue']])
    ids,first,inverse=np.unique(a['instance_id'],return_index=True,return_inverse=True)
    np.testing.assert_array_equal(colors,colors[first][inverse])
    palette={str(int(pid)):colors[ix].tolist() for pid,ix in zip(ids,first) if pid>0}
    scenes[scene]={'source_PLY':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                   'geometry_and_labels_match_frozen_baseline':True,'id_colors':palette,
                   'unassigned_color':[89,89,89]}
    print(json.dumps({'scene':scene,'original_palette_instances':len(palette),'status':'PASS'}),flush=True)
data={'status':'PASS','GT_used':False,'original_maps_unchanged':True,'scenes':scenes}
(OUT/'instance_palettes.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
