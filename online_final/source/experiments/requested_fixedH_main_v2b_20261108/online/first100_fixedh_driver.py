from pathlib import Path
import os,sys,json,time,traceback,importlib.util,subprocess
import numpy as np
from PIL import Image
s=sys.argv[1];R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online')/s/'100/B';E=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online');V=R/'candidates/visibility';V.mkdir(parents=True,exist_ok=True);start=time.time()
def put(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
try:
 assert json.loads((R/'candidates/background070/status.json').read_text())['status']=='PASS'
 cfg=json.loads((R/'mapping/config.json').read_text());cam=cfg['camera'];src=Path(cfg['source']['scene_root']);frames=list(range(cfg['frame_selection']['start'],cfg['frame_selection']['stop_exclusive'],cfg['frame_selection']['stride']));assert len(frames)==100
 poses=np.loadtxt(src/cfg['source']['trajectory']).reshape(-1,4,4)
 with np.load(R/'candidates/baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'].copy()
 cp=E/'branches'/s/'selection_code/100/B/catalog/common.py';spec=importlib.util.spec_from_file_location('projection',cp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 caches=R/'candidates/overmerge';existing=[f for f in frames if (caches/f'frame{f:06d}_visible.npz').exists()];assert existing
 f=existing[0];depth=np.asarray(Image.open(src/cfg['source']['depth_pattern'].format(frame=f)),np.float32)/cam['depth_png_units_per_meter'];vis,uv=m.project_visible(xyz,poses[f],depth,cam);idx=np.flatnonzero(vis).astype(np.int32)
 with np.load(caches/f'frame{f:06d}_visible.npz') as z:np.testing.assert_array_equal(idx,z['vertex_index']);np.testing.assert_array_equal(uv[idx],z['uv'])
 for i,f in enumerate(frames,1):
  target=V/f'f{f:06d}.npz';cached=caches/f'frame{f:06d}_visible.npz'
  if not target.exists():
   if cached.exists():os.symlink(cached,target)
   else:
    depth=np.asarray(Image.open(src/cfg['source']['depth_pattern'].format(frame=f)),np.float32)/cam['depth_png_units_per_meter'];vis,uv=m.project_visible(xyz,poses[f],depth,cam);idx=np.flatnonzero(vis).astype(np.int32);tmp=target.with_name(target.stem+'.tmp.npz');np.savez_compressed(tmp,vertex_index=idx,uv=uv[idx]);tmp.replace(target)
  if i%10==0:put(V/'status.json',{'status':'RUNNING','frames_done':i,'frames_total':len(frames),'elapsed_seconds':time.time()-start})
 put(V/'status.json',{'status':'PASS','frames':frames,'cached_frames':len(existing),'cache_parity_verified':True,'elapsed_seconds':time.time()-start,'GT_used':False,'tracking_or_repair':False})
 put(R/'fixedH_driver_status.json',{'status':'RUNNING','stage':'FIXED_H_SCREENING_ONLY'})
 with (R/'fixedH_geometry.log').open('ab') as f:rc=subprocess.call(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(E/'first100_fixedh_geometry.py'),s],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=os.environ)
 if rc:raise RuntimeError('fixed-H failed: fixedH_geometry.log')
 put(R/'fixedH_driver_status.json',{'status':'PASS','stage':'FIXED_H_SCREENING_ONLY','GT_used':False,'tracking_or_repair':False,'elapsed_seconds':time.time()-start})
except BaseException:
 put(R/'fixedH_driver_status.json',{'status':'FAILED','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
