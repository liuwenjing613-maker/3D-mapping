"""Posthoc seed-surface consistency diagnostics, never automatic map editing."""
from pathlib import Path
import json, time
import numpy as np
from PIL import Image

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/long_tracking_20261007')
UID='room2-14047feb0df3aaa5'

def intervals(frames):
    rows=[]
    for fid in frames:
        if rows and fid==rows[-1][1]+1:rows[-1][1]=fid
        else:rows.append([fid,fid])
    return rows

def main():
    start=time.monotonic()
    result=json.loads((ROOT/UID/'complete.json').read_text())
    visibility=json.loads((ROOT/'room2_mask24_visibility.json').read_text())
    cfg=result['source_config'];scene=Path(cfg['source']['scene_root']);cam=cfg['camera']
    poses=np.loadtxt(scene/cfg['source']['trajectory']).reshape(-1,4,4)
    pts=np.array(visibility['seed_points']);tid=visibility['track_id'];rows=[]
    for fid in visibility['depth_supported_frames']:
        pose=poses[fid];pc=(pts-pose[:3,3])@pose[:3,:3]
        front=(pc[:,2]>.05)&(pc[:,2]<10);pc=pc[front]
        u=np.rint(cam['fx']*pc[:,0]/pc[:,2]+cam['cx']).astype(int)
        v=np.rint(cam['fy']*pc[:,1]/pc[:,2]+cam['cy']).astype(int)
        inside=(u>=0)&(u<cam['width'])&(v>=0)&(v<cam['height']);u,v,pc=u[inside],v[inside],pc[inside]
        depth=np.array(Image.open(scene/cfg['source']['depth_pattern'].format(frame=fid))).astype(float)/cam['depth_png_units_per_meter']
        d=depth[v,u];valid=(d>0)&(np.abs(d-pc[:,2])<=visibility['depth_tolerance_m'])
        labels=np.array(Image.open(ROOT/UID/f'f{fid:06d}.png'))
        count=int(valid.sum());matched=int(np.sum(labels[v[valid],u[valid]]==tid))
        rows.append({'frame':fid,'visible_seed_samples':count,'covered_by_tracked_mask':matched,'fraction':matched/count if count else 0})
    fractions=np.array([r['fraction'] for r in rows])
    supported=[r['frame'] for r in rows if r['fraction']>=.5]
    objects=[]
    for obj in result['objects_metadata']:
        areas=[r['areas'][str(obj['track_id'])] for r in result['frames']]
        objects.append({**obj,'nonempty_frames':sum(a>0 for a in areas),'empty_frames':sum(a==0 for a in areas),'min_area':min(areas),'max_area':max(areas)})
    data={'status':'PASS','case_uid':UID,'objects':8,'tracked_frames':2000,'tracking_seconds':result['elapsed_seconds'],
          'diagnostic_seconds':round(time.monotonic()-start,2),'GT_used':False,'map_updated':False,
          'mask24':{'geometric_candidate_frames':len(visibility['candidate_frames']),'depth_supported_frames':len(rows),
                    'mask_covers_at_least_half_visible_seed_samples_frames':len(supported),'coverage_threshold':.5,
                    'median_seed_sample_coverage':float(np.median(fractions)),
                    'fraction_quantiles':np.quantile(fractions,[0,.25,.5,.75,1]).tolist(),
                    'consistent_intervals':intervals(supported),'rows':rows},'objects_stats':objects,
          'diagnostic_is_heuristic_not_pixel_ground_truth':True,
          'long_run_has_not_been_used_to_replace_votes_or_change_PLY':True}
    (ROOT/'long_tracking_diagnostics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ['mask24','objects_stats']}))
    print(json.dumps({k:v for k,v in data['mask24'].items() if k not in ['rows','consistent_intervals']}))

if __name__=='__main__':main()
