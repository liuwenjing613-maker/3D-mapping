"""Read only past-view availability for already known prefix candidates."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
n=int(sys.argv[1]);assert n in [20,50]
R=Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021');C=Path(f'/home/chenkejun/CVPR/experiments/online_room0_prefix{n}_20261021')
read=lambda p:json.loads(Path(p).read_text())
cfg=read(C/'config.json');poses=np.loadtxt(Path(cfg['source']['scene_root'])/cfg['source']['trajectory']).reshape(-1,4,4)
fids=list(range(0,5*n,5));sys.path.insert(0,str(C/'catalog'));from common import project_visible
with np.load(R/'baseline/final/instance_surface.npz') as z:xyz=z['xyz_m']
with np.load(R/'v1/surface_supports.npz') as z:allidx=z['indices'];off=z['offsets']
man=read(R/'v1/seed_manifest.json');kept=set(read(R/'v2/selection.json')['retained_group_ids'])
groups={g['id']:g for g in man['groups'] if g['id'] in kept}
supports={gid:allidx[off[g['primary']['index']]:off[g['primary']['index']+1]] for gid,g in groups.items()}
result={gid:[] for gid in groups}
for fid in fids:
 cache=R/f'overmerge/frame{fid:06d}_visible.npz'
 if cache.exists():
  with np.load(cache) as z:idx=z['vertex_index'];uv=z['uv']
 else:
  dep=np.asarray(Image.open(Path(cfg['source']['scene_root'])/cfg['source']['depth_pattern'].format(frame=fid)),float)/cfg['camera']['depth_png_units_per_meter']
  valid,alluv=project_visible(xyz,poses[fid],dep,cfg['camera']);idx=np.flatnonzero(valid);uv=alluv[idx]
 assert np.all(idx[1:]>=idx[:-1])
 for gid,s in supports.items():
  common,_,positions=np.intersect1d(s,idx,assume_unique=True,return_indices=True);fraction=len(common)/len(s)
  if fraction<.8:continue
  xy=uv[positions];lo=xy.min(0);hi=xy.max(0)
  if min(lo[0],lo[1],1199-hi[0],679-hi[1])<5:continue
  result[gid].append(dict(frame=fid,visible_known_support_fraction=fraction,projected_bbox=[int(lo[0]),int(lo[1]),int(hi[0]+1),int(hi[1]+1)]))
rows=[]
for gid,views in result.items():
 center=xyz[supports[gid]].mean(0);frames=[v['frame'] for v in views]
 if len(frames)>1:
  rays=np.array([poses[f,:3,3]-center for f in frames]);rays/=np.linalg.norm(rays,axis=1,keepdims=True)
  angles=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1)));aa,bb=np.unravel_index(np.argmax(angles),angles.shape);maximum=float(angles[aa,bb]);pair=[frames[aa],frames[bb]]
 else:maximum=0.;pair=frames
 rows.append(dict(gid=gid,geometrically_eligible_past_views=len(views),max_angle_deg=maximum,most_diverse_frames=pair,views=views))
out=dict(status='PASS',prefix=n,cutoff=fids[-1],causal_prefix_only=True,meaning='Past frames with >=80% of CURRENT KNOWN primary surface depth-visible and 5px image border clearance. Does not prove full object visibility or valid segmentation.',new_segmentation_not_run=True,rows=rows)
(R/'diagnostics/trigger_exploration_20261021/view_bank.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('VIEW_BANK_COMPLETE',n,[(r['gid'],r['geometrically_eligible_past_views'],round(r['max_angle_deg'],1),r['most_diverse_frames']) for r in rows],flush=True)
