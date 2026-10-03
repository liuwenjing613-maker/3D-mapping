"""Prepare all previously selected RGB crops, with frozen full-frame controls."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
from PIL import Image
H=Path('/home/chenkejun/CVPR/experiments/p1a1_roi_cropformer_zoom_20261003')
R=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003')
PREV=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002')
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/p1a1_room0_roi_review_20261002')
from select_history import project_visible
from analyse_conflicts import box
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
R.mkdir(exist_ok=True);(R/'inputs').mkdir(exist_ok=True);(R/'inference').mkdir(exist_ok=True)
cfg=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/configs/raw.json').read_text());c=cfg['camera'];scene=Path(cfg['source']['scene_root']);maskroot=Path(cfg['source']['mask_root'])
review=json.loads((PREV/'review_evidence.json').read_text());selected=json.loads((PREV/'review_selection.json').read_text());representatives=set(selected['distinct_component_representatives']+selected['additional_U_T_examples'])
diagnosis=json.loads((PREV/'conflict_diagnosis/conflict_cases.json').read_text());diagnosis={r['rank']:r for r in diagnosis}
source_meta={r['frame']:r for r in map(json.loads,(maskroot/'frames.jsonl').read_text().splitlines())}
ranked=json.loads((PREV/'roi_ranked.json').read_text());poses=np.loadtxt(scene/'traj.txt').reshape(-1,4,4)
with np.load('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as z:xyz=z['xyz_m']
m=np.load(PREV/'roi_membership.npz');rp=m['context_point_index'];ro=m['context_offsets'];cp=m['core_point_index'];co=m['core_offsets']
jobs=[];cases=[];framecache={};contentjobs={}
def frame(fid):
 if fid in framecache:return framecache[fid]
 im=scene/f'results/frame{fid:06d}.jpg';raw=maskroot/f'frame{fid:06d}.png';metadata=source_meta[fid]
 assert sha(im)==metadata['input_sha256'] and sha(raw)==metadata['mask_sha256']
 rgb=np.asarray(Image.open(im).convert('RGB'));baseline=np.asarray(Image.open(raw));depth=np.asarray(Image.open(scene/f'results/depth{fid:06d}.png'),np.float32)/c['depth_png_units_per_meter']
 framecache[fid]=(rgb,baseline,depth);return framecache[fid]
def addjob(key,rgb,baseline,extra):
 rgbfile=R/'inputs'/f'{key}_rgb.png';basefile=R/'inputs'/f'{key}_baseline.png'
 assert not rgbfile.exists()
 Image.fromarray(rgb).save(rgbfile);Image.fromarray(baseline).save(basefile)
 digest=hashlib.sha256(rgb.tobytes()+str(rgb.shape).encode()).hexdigest();duplicate=contentjobs.get(digest)
 job={'key':key,'input_file':str(rgbfile),'baseline_file':str(basefile),'shape':list(rgb.shape[:2]),'RGB_file_sha256':sha(rgbfile),'baseline_file_sha256':sha(basefile),'same_RGB_as':duplicate,**extra}
 if duplicate is None:contentjobs[digest]=key
 jobs.append(job);return job
for rec in review:
 rank=rec['rank'];roi=ranked[rank-1];k=roi['candidate_index'];context=rp[ro[k]:ro[k+1]];core=cp[co[k]:co[k+1]]
 for view in rec['views']:
  fid=view['frame_id'];rgb,baseline,depth=frame(fid);vn=view['view_number'];row={'roi_id':rec['roi_id'],'rank':rank,'view':vn,'frame':fid,'representative':rank in representatives,'core_counts':roi['core_counts'],'crops':{}}
  for strategy in ['tight']+(['wide'] if rank in representatives else []):
   if strategy=='tight':crop=view['crop_xyxy']
   elif vn==1:crop=diagnosis[rank]['crop_xyxy']
   else:
    visible,uv=project_visible(xyz[context],poses[fid],depth,c);crop=box(uv,visible,c)
   x0,y0,x1,y1=crop;assert 0<=x0<x1<=1200 and 0<=y0<y1<=680
   cam=(xyz[core]-poses[fid,:3,3])@poses[fid,:3,:3];uv=np.column_stack((cam[:,0]*c['fx']/cam[:,2]+c['cx']-x0,cam[:,1]*c['fy']/cam[:,2]+c['cy']-y0))
   inside=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<x1-x0)&(uv[:,1]>=0)&(uv[:,1]<y1-y0)
   anchor=np.median(uv[inside],axis=0).tolist() if inside.any() else [(x1-x0)/2,(y1-y0)/2]
   key=f'{rec["roi_id"]}_v{vn}_{strategy}'
   job=addjob(key,rgb[y0:y1,x0:x1],baseline[y0:y1,x0:x1],{'kind':'ROI_crop','strategy':strategy,'ROI':rec['roi_id'],'rank':rank,'view':vn,'frame':fid,'crop_xyxy':crop,'alarm_projection_anchor_uv':anchor})
   row['crops'][strategy]=key
  cases.append(row)
for fid in sorted(framecache):
 rgb,baseline,_=framecache[fid];addjob(f'full_f{fid:06d}',rgb,baseline,{'kind':'full_frame_control','frame':fid,'source_RGB_sha256':source_meta[fid]['input_sha256']})
manifest={'scene':'room0','selected_ROIs':len(review),'representative_ROIs':len(representatives),'selected_ROI_views':len(cases),'tight_jobs':sum(j.get('strategy')=='tight' for j in jobs),'wide_jobs':sum(j.get('strategy')=='wide' for j in jobs),'full_frame_controls':len(framecache),'jobs':jobs,'cases':cases,'no_GT_used':True,'no_depth_geometry_refinement':True,'map_unchanged':True,'crop_policy':{'tight':'exact previous selected RGB crop','wide':'exact displayed diagnostic crop for primary; same 440x320 minimum measured-depth ROI window for auxiliary views'},'source_evidence_manifest':str(PREV/'review_evidence.json')}
(R/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in {'jobs','cases'}}),flush=True)
