"""ROI-local Euclidean radius expansion on fixed TSDF vertices; read-only inputs."""
from pathlib import Path
import collections,hashlib,json,time,zipfile
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

R=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/intersection');W=R/'web';(W/'assets').mkdir(parents=True,exist_ok=True)
SRC=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/frontend');FILTER=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/overmerge')
BASE=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/baseline')
GEOM=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/geometry/geometry_prefix20.json')
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 start=time.time();d=read(SRC/'review/viewer_data.json');a=read(FILTER/'web/data.json');man=read(SRC/'source/room0/manifest.json')
 paths=[BASE/'final/instance_surface.npz',BASE/'surface_evidence.npz',SRC/'source/room0/membership.npz',FILTER/'web/data.json',GEOM]
 before={str(p):sha(p) for p in paths};assert before[str(paths[0])]==a['source_hashes'][str(paths[0])]
 assert before[str(paths[1])]==a['source_hashes'][str(paths[1])];assert before[str(paths[2])]==man['fixed_membership_sha256']
 v=float(read(GEOM)['voxel_size_m']);radius=2*v;assert v==.01
 with np.load(paths[0]) as z:xyz=z['xyz_m'].astype(np.float64);final=z['instance_id'].copy()
 with np.load(paths[1]) as z:assert np.array_equal(xyz,z['xyz_m']);state=z['state'].copy()
 with np.load(paths[2]) as z:cores={str(uid):z['core_point_index'][z['core_offsets'][i]:z['core_offsets'][i+1]].copy() for i,uid in enumerate(z['uid'])}
 jobs={j['key']:j for j in a['jobs']};out=dict(schema='roi-tsdf-radius-intersection-v1',rule=dict(voxel_m=v,radius_m=radius,seed='all original C/U/T anomaly vertices of the current ROI',metric='Euclidean distance on fixed TSDF surface',condition='expanded_intersection_points >= 1',ranking_by_intersection=False),source_hashes=before,cases={},frames={},crops={})
 stats=collections.Counter();sample_checks=0;unique_pairs=set();unique_new_pairs=set();eligible_rois=set();expindices={};hashes=read(FILTER/'input_hashes.json');verified=set()
 for case in d['cases']:
  uid=case['uid'];core=cores[uid];assert np.all((final[core]<=0)&np.isin(state[core],[0,1,3]))
  assert len(np.unique(core))==len(core);coremask=np.zeros(len(xyz),bool);coremask[core]=True
  tree=cKDTree(xyz[core]);dist,_=tree.query(xyz,k=1,distance_upper_bound=np.nextafter(radius,np.inf),workers=4)
  expanded=dist<=radius;assert np.all(expanded[core]);expindices[uid]=np.flatnonzero(expanded).astype(np.int32)
  # Independent exact nearest-distance checks, including both sides of the boundary.
  rng=np.random.default_rng(51);probes=np.unique(np.r_[rng.choice(len(xyz),80,replace=False),core[:10],np.flatnonzero(expanded&~coremask)[:20]])
  for idx in probes:
   exact=np.sqrt(np.square(xyz[core]-xyz[idx]).sum(1)).min();assert bool(expanded[idx])==bool(exact<=radius);sample_checks+=1
  out['cases'][uid]=dict(core_points=len(core),expanded_points=int(expanded.sum()),added_points=int(expanded.sum()-len(core)),aliases=case['aliases'])
  byframe=collections.defaultdict(list)
  for view in case['views']:
   for key in view['variants']:
    if key not in byframe[view['frame']]:byframe[view['frame']].append(key)
  for frame,keys in byframe.items():
   with np.load(FILTER/f'frame{frame:06d}_visible.npz') as z:idx=z['vertex_index'];uv=z['uv']
   assert len(np.unique(idx))==len(idx);insidecore=coremask[idx];insideplus=expanded[idx]
   rgb=np.zeros((680,1200,4),np.uint8);add=uv[insideplus&~insidecore];rgb[add[:,1],add[:,0]]=[0,212,238,255]
   cp=uv[insidecore];rgb[cp[:,1],cp[:,0]]=0
   rel=f'assets/{uid}_f{frame:06d}_expansion.png';Image.fromarray(rgb).save(W/rel)
   fk=uid+':'+str(frame);out['frames'][fk]=dict(expansion=rel,core_visible=int(insidecore.sum()),expanded_visible=int(insideplus.sum()),added_visible=int((insideplus&~insidecore).sum()))
   cache={}
   for key in keys:
    job=jobs[key];x0,y0,x1,y1=job['box'];in_crop=(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
    rec=dict(frame_key=fk,models=[])
    for mi,sid in enumerate(job['sources']):
     source=a['sources'][sid]
     if source['path'] not in verified:assert sha(Path(source['path']))==hashes[source['path']];verified.add(source['path'])
     if sid not in cache:
      labels=np.asarray(Image.open(source['path']),np.int32);sx0,sy0,sx1,sy1=source['box'];valid=(uv[:,0]>=sx0)&(uv[:,0]<sx1)&(uv[:,1]>=sy0)&(uv[:,1]<sy1);vv=np.flatnonzero(valid);l=labels[uv[vv,1]-sy0,uv[vv,0]-sx0];n=int(labels.max())+1
      hist=[np.bincount(l[sel[vv]],minlength=n) for sel in [np.ones(len(idx),bool),insidecore,insideplus]]
      cache[sid]=(vv,l,hist)
     vv,l,hist=cache[sid];crop_core=np.bincount(l[insidecore[vv]&in_crop[vv]],minlength=len(hist[0]));crop_plus=np.bincount(l[insideplus[vv]&in_crop[vv]],minlength=len(hist[0]))
     members=set(job['members'][mi]);rows={};eligible=[];excluded=[]
     for lid,cid in source['mapping'].items():
      if cid not in members:continue
      lid=int(lid);cand=a['candidates'][cid];N,c,p=[int(h[lid]) for h in hist];assert N==cand['N'] and 0<=c<=p<=N
      decision='overmerge' if cand['decision']=='reject' else ('intersects' if p else 'no_intersection')
      rows[str(lid)]=dict(N=N,core=c,expanded=p,added=p-c,core_in_crop=int(crop_core[lid]),expanded_in_crop=int(crop_plus[lid]),decision=decision,candidate_id=cand['id'])
      stats[decision]+=1
      if decision=='intersects':
       eligible.append(lid);eligible_rois.add(uid);unique_pairs.add((uid,cand['id']));stats['expansion_newly_included']+=int(c==0)
       if c==0:unique_new_pairs.add((uid,cand['id']))
      elif decision=='no_intersection':excluded.append(lid)
      # Direct vertex-membership recount for all masks near the one-point threshold.
      if p<=2 or (p>0 and c==0):
       chosen=vv[l==lid];assert int(expanded[idx[chosen]].sum())==p;assert int(coremask[idx[chosen]].sum())==c;sample_checks+=1
     rec['models'].append(dict(labels=rows,eligible_ids=eligible,no_intersection_ids=excluded))
    out['crops'][key]=rec
  print('ROI',case['ROI'],'core',len(core),'expanded',int(expanded.sum()),'crops',len(out['crops']),flush=True)
 assert len(out['cases'])==len(d['cases']) and len(out['crops'])==len(a['jobs'])
 assert sum(stats[k] for k in ['intersects','no_intersection','overmerge'])==a['stats']['occurrences'];assert stats['overmerge']==sum(len(c['occ']) for c in a['candidates'] if c['decision']=='reject')
 out['stats']=dict(stats,ROI_count=len(d['cases']),crops=len(a['jobs']),ROI_with_candidates=len(eligible_rois),unique_ROI_candidate_pairs=len(unique_pairs),newly_included_unique_ROI_candidate_pairs=len(unique_new_pairs))
 np.savez_compressed(R/'expanded_vertex_indices.npz',**expindices)
 assert before=={str(p):sha(p) for p in paths};save(W/'intersection.json',out)
 audit=dict(status='PASS',stats=out['stats'],rule=out['rule'],independent_checks=sample_checks,source_files_unchanged=True,no_GT_used=True,no_map_write=True,no_new_inference=True,seconds=round(time.time()-start,2));save(R/'audit.json',audit);save(W/'audit.json',audit)
 files={p.relative_to(W).as_posix():sha(p) for p in W.rglob('*') if p.is_file()};save(W/'files_sha256.json',files)
 with zipfile.ZipFile(R/'intersection_web.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in W.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(W).as_posix())
 save(R/'receipt.json',dict(sha256=sha(R/'intersection_web.zip'),bytes=(R/'intersection_web.zip').stat().st_size));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
