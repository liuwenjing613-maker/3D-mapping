"""Conservative pose/frustum candidates for all human masks; depth-check demo."""
from pathlib import Path
import itertools, json, time
import numpy as np
from PIL import Image

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
INPUT=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')
OUT=ROOT/'long_tracking_20261007'

def intervals(frames):
    result=[]
    for fid in frames:
        if result and fid==result[-1][1]+1:result[-1][1]=fid
        else:result.append([fid,fid])
    return result

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    seeds=json.loads((ROOT/'human_seeds_20ad6139511b/seed_manifest.json').read_text())
    cases=[];demo=None
    for c in seeds['seeds']:
        if c['status']!='READY':continue
        uid,scene=c['case_uid'],c['scene']
        cfg=json.loads((INPUT/scene/'configs/raw.json').read_text());source=cfg['source'];cam=cfg['camera']
        root=Path(source['scene_root']);poses=np.loadtxt(root/source['trajectory']).reshape(-1,4,4)
        with np.load(BASE/f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as z:xyz=z['xyz_m']
        with np.load(ROOT/'repair'/uid/'seed_projected_support.npz') as z:point,track=z['surface_point_index'],z['track_id']
        objects=[];union=set()
        for obj in c['objects']:
            p=np.unique(point[track==obj['track_id']]);pts=xyz[p]
            lo,hi=pts.min(0)-.20,pts.max(0)+.20
            corners=np.array(list(itertools.product(*zip(lo,hi))))
            candidates=[]
            for fid,pose in enumerate(poses):
                pc=(corners-pose[:3,3])@pose[:3,:3]
                x,y,z=pc.T
                # Reject only boxes completely beyond one frustum plane.
                planes=[z-.05,10-z,cam['fx']*x+cam['cx']*z,(cam['width']-1-cam['cx'])*z-cam['fx']*x,
                        cam['fy']*y+cam['cy']*z,(cam['height']-1-cam['cy'])*z-cam['fy']*y]
                if all(np.any(v>=0) for v in planes):candidates.append(fid)
            row={**obj,'candidate_frames':candidates,'candidate_count':len(candidates),'intervals':intervals(candidates),
                 'seed_surface_points':len(p),'seed_bbox_margin_m':.20,'geometry_candidate_is_true_visibility':False}
            objects.append(row);union.update(candidates)
            if scene=='room2' and obj['native_mask_id']==24:
                sampled=pts[np.linspace(0,len(pts)-1,min(len(pts),2048),dtype=int)]
                depth_counts={}
                for fid in candidates:
                    pose=poses[fid];pc=(sampled-pose[:3,3])@pose[:3,:3]
                    front=(pc[:,2]>.05)&(pc[:,2]<10)
                    pc=pc[front]
                    u=np.rint(cam['fx']*pc[:,0]/pc[:,2]+cam['cx']).astype(int)
                    v=np.rint(cam['fy']*pc[:,1]/pc[:,2]+cam['cy']).astype(int)
                    inside=(u>=0)&(u<cam['width'])&(v>=0)&(v<cam['height'])
                    u,v,pc=u[inside],v[inside],pc[inside]
                    depth=np.array(Image.open(root/source['depth_pattern'].format(frame=fid))).astype(float)/cam['depth_png_units_per_meter']
                    d=depth[v,u]
                    gap=np.abs(d-pc[:,2])
                    depth_counts[str(fid)]=int(np.sum((d>0)&(gap<=.03)))
                visible=[int(f) for f,n in depth_counts.items() if n>=10]
                demo={'case_uid':uid,'native_mask_id':24,'track_id':obj['track_id'],'seed_frame':c['source_choice']['frame'],
                      'raw_frames':len(poses),'candidate_frames':candidates,'depth_supported_frames':visible,
                      'depth_supported_intervals':intervals(visible),'sampled_seed_surface_points':len(sampled),
                      'depth_tolerance_m':.03,'minimum_support_points':10,'depth_support_counts':depth_counts,
                      'seed_points':sampled.tolist(),'source_scene_root':str(root),'source_config':str(INPUT/scene/'configs/raw.json'),
                      'not_a_guarantee_of_all_object_visibility':True}
                (OUT/'room2_mask24_visibility.json').write_text(json.dumps(demo,ensure_ascii=False,indent=2)+'\n')
        case={'case_uid':uid,'scene':scene,'ROI':c['ROI'],'raw_frames':len(poses),'objects':objects,
              'union_candidate_frames':sorted(union),'union_count':len(union)}
        cases.append(case)
        print(json.dumps({'scene':scene,'ROI':c['ROI'],'objects':len(objects),'union_frames':len(union)}),flush=True)
    summary={'status':'PASS','GT_used':False,'cases':cases,'objects':sum(len(c['objects']) for c in cases),
             'seconds':round(time.monotonic()-start,2),'candidate_frames_are_conservative_heuristics':True,
             'sampling_does_not_change_v3_or_repair':True,'demo':{k:v for k,v in demo.items() if k not in ['seed_points','depth_support_counts']}}
    (OUT/'visibility_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','seconds':summary['seconds'],'demo_candidates':len(demo['candidate_frames']),
                      'demo_depth_supported_frames':len(demo['depth_supported_frames'])}),flush=True)

if __name__=='__main__':main()
