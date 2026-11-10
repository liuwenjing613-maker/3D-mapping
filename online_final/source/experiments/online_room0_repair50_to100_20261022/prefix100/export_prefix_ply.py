"""Point-only exports with the same ID palette used in the recent eight-scene review."""
from online_common import *
import colorsys
import numpy as np

def main():
    with np.load(R/'baseline/final/instance_surface.npz') as z:
        xyz=z['xyz_m']; labels=z['instance_id']
    with np.load(R/'baseline/surface_evidence.npz') as z:
        np.testing.assert_array_equal(xyz,z['xyz_m']); state=z['state']
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                    ('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
    records=[]
    for name,selected in [('instances',labels>0),('three_states',labels<=0)]:
        idx=np.flatnonzero(selected); a=np.empty(len(idx),dtype)
        for j,field in enumerate(['x','y','z']):a[field]=xyz[idx,j]
        a['instance_id']=labels[idx];a['state']=state[idx];a['surface_index']=idx
        colors=np.zeros((len(idx),3),np.uint8)
        if name=='instances':
            ids,inverse=np.unique(labels[idx],return_inverse=True)
            palette=np.asarray([np.array(colorsys.hsv_to_rgb((int(k)*.61803398875)%1,.67,.94))*255 for k in ids],np.uint8)
            colors=palette[inverse]
        else:
            assert set(np.unique(state[idx]))<={0,1,3}
            colors[state[idx]==1]=[255,255,0];colors[state[idx]==3]=[255,0,0]
        for j,field in enumerate(['red','green','blue']):a[field]=colors[:,j]
        header=('ply\nformat binary_little_endian 1.0\ncomment room0 causal prefix100; original point indices and IDs\n'
            +f'element vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\n'
            +'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\n'
            +'property uchar state\nproperty int surface_index\nend_header\n').encode()
        p=R/'ply'/('room0_prefix100_'+name+'.ply')
        with p.open('wb') as stream:stream.write(header);a.tofile(stream)
        check=np.memmap(p,dtype=dtype,mode='r',offset=len(header),shape=(len(a),))
        np.testing.assert_array_equal(check,a)
        records.append(dict(file=p.name,points=len(a),sha256=sha(p),size=p.stat().st_size))
    write(R/'ply/manifest.json',dict(status='PASS',exports=records,geometry_points=len(xyz),
        frame_ids=list(FIDS),no_faces=True,no_normals=True,palette='HSV(id*0.61803398875,0.67,0.94)',
        three_states='U=black; T=yellow; C=red; only final instance_id <= 0',index_and_ID_verified=True))
    print('PLY_EXPORT_PASS',records,flush=True)

if __name__=='__main__':main()
