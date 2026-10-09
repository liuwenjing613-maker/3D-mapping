"""Export all frozen surface points with four aligned label states, without GT."""
from pathlib import Path
import sys,json
import numpy as np
import run_ablation as core


def main():
    sys.path.insert(0,str(core.LEGACY));import audit_and_render as render
    for scene in ['room0','room2']:
        engine=core.load_engine(scene);out=core.OUT/scene
        frozen=core.read(out/'predictions_frozen.json')
        assert frozen['status']=='PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT'
        ef=core.read(engine.ROOT/'evaluation_freeze.json')
        paths={'baseline':Path(ef['predictions']['baseline']['path']),
               'unrestricted':engine.ROOT/'validated/full_track/final/instance_surface.npz',
               **{name:Path(row['surface']) for name,row in frozen['conditions'].items()}}
        names=['baseline','unrestricted','gap0','gap1'];assert set(paths)==set(names)
        hashes={str(p):engine.r.sha(p) for p in paths.values()}
        for name in ['gap0','gap1']:assert hashes[str(paths[name])]==frozen['conditions'][name]['sha256']
        arrays={name:engine.f.load_arrays(paths[name]) for name in names}
        for name in names:engine.f.require_same_geometry(arrays['baseline'],arrays[name])
        xyz=arrays['baseline']['xyz_m'];n=len(xyz)
        dtype=np.dtype([(v,'<f4') for v in ['x','y','z']]+[(name,'<i4') for name in names])
        pack=np.empty(n,dtype)
        for i,key in enumerate(['x','y','z']):pack[key]=xyz[:,i]
        colors=dict(core.read(engine.ROOT/'instance_palettes.json')['scenes'][scene]['id_colors'])
        colors.update(core.read(engine.ROOT/'review/instance_colors.json')['id_colors'])
        positive=sorted(set(int(pid) for a in arrays.values() for pid in np.unique(a['instance_id']) if pid>0))
        fallback=render.palette(max(positive,default=0))
        omitted=[pid for pid in positive if str(pid) not in colors]
        for pid in omitted:colors[str(pid)]=fallback[pid].tolist()
        stats={}
        for name in names:
            ids=arrays[name]['instance_id'];pack[name]=ids;old=arrays['unrestricted']['instance_id']
            stats[name]={'unknown':int(np.sum(ids<=0)),'changed_vs_unrestricted':int(np.sum(ids!=old)),
              'new_unknown_vs_unrestricted':int(np.sum((ids<=0)&(old>0))),
              'restored_assigned_vs_unrestricted':int(np.sum((ids>0)&(old<=0))),
              'identity_swaps_vs_unrestricted':int(np.sum((ids>0)&(old>0)&(ids!=old))),
              'positive_instances':int(len(np.unique(ids[ids>0])))}
        path=out/'comparison_all_points.bin';temp=path.with_suffix('.bin.tmp');pack.tofile(temp)
        restored=np.memmap(temp,mode='r',dtype=dtype,shape=(n,))
        for key in dtype.names:np.testing.assert_array_equal(restored[key],pack[key])
        del restored;temp.replace(path);engine.check_hashes(hashes)
        identical={key:bool(np.array_equal(arrays['gap0'][key],arrays['gap1'][key]))
                   for key in set(arrays['gap0'])&set(arrays['gap1'])}
        info={'status':'PASS','scene':scene,'points':n,'stride_bytes':dtype.itemsize,'state_names':names,
          'path':path.name,'bytes':path.stat().st_size,'sha256':engine.r.sha(path),
          'native_bounds':[xyz.min(0).tolist(),xyz.max(0).tolist()], 'id_colors':colors,
          'surface_source_hashes':hashes,'stats':stats,'gap0_gap1_array_equality':identical,
          'roundtrip_geometry_and_labels_exact':True,'downsampled':False,'GT_used_for_rendering':False,
          'palette_fallback_for_omitted_IDs':omitted,
          'exporter_sha256':engine.r.sha(Path(__file__))}
        core.dump(out/'comparison_pack.json',info);print(scene,n,stats,identical,flush=True)


if __name__=='__main__':main()
