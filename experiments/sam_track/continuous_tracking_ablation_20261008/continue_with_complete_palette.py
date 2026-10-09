"""Resume unchanged, cached repairs; correct only the PLY color-table completeness."""
from pathlib import Path
import argparse,sys,time,traceback
import numpy as np
import run_ablation as core


def export_complete_palette(engine, scene, name, surface):
    sys.path.insert(0,str(core.LEGACY))
    import audit_and_render as render
    baseline_path=engine.ROOT/'instance_palettes.json'
    review_path=engine.ROOT/'review/instance_colors.json'
    protected={str(p):engine.r.sha(p) for p in [baseline_path,review_path,Path(render.__file__),Path(__file__)]}
    arrays=engine.f.load_arrays(surface);ids=arrays['instance_id']
    colors=dict(core.read(baseline_path)['scenes'][scene]['id_colors'])
    colors.update(core.read(review_path)['id_colors'])
    positive=sorted(map(int,np.unique(ids[ids>0])))
    missing=[pid for pid in positive if str(pid) not in colors]
    # Use the exact deterministic palette already used by the original export_review.
    # Existing explicit colors, including every human-selected identity, take precedence.
    fallback=render.palette(max(positive,default=0))
    for pid in missing:colors[str(pid)]=fallback[pid].tolist()
    rgb=np.full((len(ids),3),[105,105,105],np.uint8)
    for pid in positive:rgb[ids==pid]=colors[str(pid)]
    dtype=np.dtype([(n,'<f4') for n in ['x','y','z']]+[(n,'u1') for n in ['red','green','blue']]+[('instance_id','<i4')])
    cloud=np.empty(len(ids),dtype)
    for i,n in enumerate(['x','y','z']):cloud[n]=arrays['xyz_m'][:,i]
    for i,n in enumerate(['red','green','blue']):cloud[n]=rgb[:,i]
    cloud['instance_id']=ids
    header='\n'.join(['ply','format binary_little_endian 1.0',
      'comment Exact frozen TSDF geometry; original explicit colors and original palette function; no downsampling',
      f'element vertex {len(ids)}','property float x','property float y','property float z',
      'property uchar red','property uchar green','property uchar blue','property int instance_id','end_header','']).encode('ascii')
    path=core.OUT/scene/name/f'{scene}_{name}_full_instance.ply';path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.ply.tmp')
    with temp.open('wb') as stream:stream.write(header);cloud.tofile(stream)
    restored=np.memmap(temp,mode='r',dtype=dtype,offset=len(header),shape=(len(ids),))
    for n in dtype.names:np.testing.assert_array_equal(restored[n],cloud[n])
    del restored;temp.replace(path)
    engine.check_hashes(protected)
    receipt={'path':str(path),'sha256':engine.r.sha(path),'points':len(ids),'bytes':path.stat().st_size,
      'original_ID_colors_preserved':True,'geometry_unchanged':True,'downsampled':False,
      'palette_sources_sha256':protected,'previously_omitted_ID_colors':{str(pid):colors[str(pid)] for pid in missing},
      'palette_only_correction':True,'surface_sha256':engine.r.sha(surface)}
    core.dump(path.parent/'ply_export_receipt.json',receipt)
    return receipt


def main(scene):
    core.export_ply=export_complete_palette
    start=time.monotonic()
    outputs=core.run_scene(scene)
    core.dump(core.OUT/scene/'continuation_complete.json',{'status':'PASS','scene':scene,
      'unchanged_repair_core_sha256':core.load_engine(scene).r.sha(core.CODE/'run_ablation.py'),
      'continuation_sha256':core.load_engine(scene).r.sha(Path(__file__)),
      'correction':'PLY color-table completeness only; original renderer palette supplies restored IDs absent from tables',
      'completed_repairs_reused_after_original_hash_checks':True,'seconds':round(time.monotonic()-start,2),
      'conditions':outputs})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scene',choices=core.SOURCE,required=True);args=parser.parse_args()
    try:main(args.scene)
    except Exception:
        core.dump(core.OUT/args.scene/'continuation_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
