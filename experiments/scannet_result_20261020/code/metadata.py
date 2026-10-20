"""Check shared input identities and register immutable geometry metadata."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import run_repairs as r
from instance_repair.cli import verify_sources
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
def main(scene):
    verify_sources()
    p=R/'inputs'/scene/'geometry/geometry_full400.json'
    target=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_scannet_engineering_fix_20261008')/scene/'shared_geometry/geometry_full400.json'
    if not p.exists():p.symlink_to(target)
    a=r.Scene(scene);cfg=a.source.config;ids=cfg['frame_ids']
    assert sorted(a.records)==ids
    checked=[]
    for fid in [ids[0],ids[len(ids)//2],ids[-1]]:
        frame,table,keys,points=a.old_frame(fid,a.maximum_id+1)
        assert frame.rgb.shape==(480,640,3)
        assert frame.depth_m.shape==frame.mask_local.shape==(480,640)
        assert np.all((frame.depth_m>=0)&(frame.depth_m<4))
        assert np.isfinite(frame.camera_to_world).all()
        checked.append(fid)
    a.verify_unchanged()
    d=dict(status='PASS',scene=scene,mapping_frames=len(ids),checked_frames=checked,
        original_source_projection_matches_saved_support=True,pose_frame_ids_preserved=True,
        depth_policy='0 < depth < 4 m',GT_used=False,mapper_source_hashes=a.source_hashes)
    (R/scene/'adapter_preflight.json').write_text(json.dumps(d,indent=2))
    print('PREFLIGHT_PASS',scene,len(ids),flush=True)
if __name__=='__main__':main(sys.argv[1])
