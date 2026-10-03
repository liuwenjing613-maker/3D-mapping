"""Read-only, GT-free ROI proposal generation on P1-A1 final room0."""
from pathlib import Path
import sys,json,time,hashlib,csv
from dataclasses import replace,asdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import save_npz,load_npz
from scipy.sparse.csgraph import connected_components,dijkstra
from plyfile import PlyData

CODE=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
sys.path.insert(0,str(CODE/'revisable_instance_map/src'))
from revisable_instance_map.offline_surface_assignment import AssignmentSettings,surface_edges
SRC=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
OUT=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002')
NATIVE=SRC/'native/room0/P1-A1/surface_p0/surface_evidence.npz'
FINAL=SRC/'final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
FULL=SRC/'final/room0/P1-A1/diffusion/room0_repaired_full.ply'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)

def packed(arrays):
 sizes=np.array([len(a) for a in arrays],np.int64)
 return np.r_[0,np.cumsum(sizes)],np.concatenate(arrays).astype(np.int32)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 hashes={'native_evidence':sha(NATIVE),'final_map':sha(FINAL),'surface_ply':sha(FULL)}
 with np.load(NATIVE) as e:e={k:e[k] for k in e.files}
 with np.load(FINAL) as f:xyz=f['xyz_m'];inst=f['instance_id'];rgb=f['rgb']
 vertex=PlyData.read(FULL)['vertex'].data
 assert np.array_equal(xyz,e['xyz_m'])
 assert np.array_equal(xyz,np.column_stack([vertex[x] for x in ['x','y','z']]))
 assert np.array_equal(inst,vertex['instance_id'])
 normals=np.column_stack([vertex[x] for x in ['nx','ny','nz']])
 normals=normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
 state=e['state'];n=len(xyz);seed_mask=(inst<=0)&np.isin(state,[0,1,3])
 assert int(seed_mask.sum())==145379
 tree=cKDTree(xyz)
 settings=replace(AssignmentSettings(),normal_cos=.20,plane_tolerance_scale=.80,
                  color_cost_weight=0.,normal_cost_weight=0.,query_chunk=25000)
 graph_file=OUT/'review_surface_graph_n02_p08.npz'
 if graph_file.exists():
  g=load_npz(graph_file);scale=np.load(OUT/'sampling_scale.npy')
 else:
  log('Building sampling scale and surface graph')
  dist=tree.query(xyz,k=5,workers=8)[0][:,4]
  base=float(np.median(dist));scale=np.clip(dist,.75*base,1.25*base)
  g,_=surface_edges(tree,xyz,normals,rgb,scale,np.arange(n,dtype=np.int32),settings)
  g=g.maximum(g.T).tocsr();g.sort_indices()
  save_npz(graph_file,g);np.save(OUT/'sampling_scale.npy',scale)
 log('Graph',n,g.nnz,'sampling median',float(np.median(scale)))
 seeds=np.flatnonzero(seed_mask)
 closed=np.unique(np.r_[seeds,g[seeds].indices])
 nc,labels=connected_components(g[closed][:,closed],directed=False)
 counts=np.bincount(labels[seed_mask[closed]],minlength=nc)
 keep=np.flatnonzero(counts)
 log('Anomaly components',len(keep),'largest seed counts',np.sort(counts)[-15:].tolist())
 groups=[];cores=[];split_parents=0
 order=np.argsort(labels,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(labels,minlength=nc))]
 for cid in keep:
  pts=closed[order[offset[cid]:offset[cid+1]]];local_seed=seed_mask[pts]
  core=pts[local_seed]
  if np.ptp(xyz[core],axis=0).max()<=.60:
   cores.append(core);groups.append(int(cid));continue
  split_parents+=1
  sub=g[pts][:,pts]
  # Farthest-point seeds on the existing graph; every alarm retained exactly once.
  core_local=np.flatnonzero(local_seed)
  closest=np.full(len(pts),np.inf);owner=np.full(len(pts),-1,np.int32)
  first=int(core_local[np.argmax(np.linalg.norm(xyz[core]-xyz[core].mean(0),axis=1))])
  centers=[];center=first
  while True:
   centers.append(center)
   distances=dijkstra(sub,directed=False,indices=center,limit=.30)
   better=distances<closest;closest[better]=distances[better];owner[better]=len(centers)-1
   far=int(core_local[np.argmax(closest[core_local])])
   if closest[far]<=.30:break
   center=far
  for k in range(len(centers)):
   part=pts[local_seed&(owner==k)]
   if len(part):cores.append(part);groups.append(int(cid))
 log('Bounded review patches',len(cores),'oversized parents split',split_parents)
 # 1cm equal-area proxy avoids bias from the irregular density of TSDF edge vertices.
 vox=np.floor(xyz/.01).astype(np.int32)
 _,voxel=np.unique(vox,axis=0,return_inverse=True)
 voxel=voxel.astype(np.int32)
 core_to=np.full(n,-1,np.int32);contexts=[];density_regions=[];rows=[]
 for k,core in enumerate(cores):
  assert np.all(core_to[core]<0);core_to[core]=k
  # Three graph rings for the simple reference density baseline, with a metric
  # floor so an isolated graph vertex cannot manufacture 100% local density.
  small=core
  for _ in range(3):small=np.unique(np.r_[small,g[small].indices])
  small=np.union1d(small,np.concatenate(tree.query_ball_point(xyz[core],.03,workers=8))).astype(np.int32)
  # Metric bounded geodesic halo; the wider region includes stable points.
  lo=xyz[core].min(0)-.101;hi=xyz[core].max(0)+.101
  center=(lo+hi)/2;radius=float(np.linalg.norm(hi-lo)/2)
  cand=np.array(tree.query_ball_point(center,radius),np.int32)
  cand=cand[np.all((xyz[cand]>=lo)&(xyz[cand]<=hi),axis=1)];cand.sort()
  sub=g[cand][:,cand]
  ds=dijkstra(sub,directed=False,indices=np.searchsorted(cand,core),min_only=True,limit=.10)
  context=cand[np.isfinite(ds)]
  # Adjacent faces can disconnect at sharp edges or flipped normals. Include a
  # 6cm spatial comparison halo as well; it does not merge anomaly cores or
  # declare those faces to belong to the same instance.
  spatial=np.unique(np.concatenate(tree.query_ball_point(xyz[core],.06,workers=8))).astype(np.int32)
  context=np.union1d(context,spatial).astype(np.int32)
  context=np.union1d(context,core).astype(np.int32)
  contexts.append(context);density_regions.append(small)
  vc,iv=np.unique(voxel[small],return_inverse=True)
  # Count only this component's alarms; neighbouring independent cores are context.
  weight=np.zeros(len(small),np.float64)
  member=np.isin(small,core,assume_unique=True)
  weight[member]=np.choose(state[small[member]],[1.,2.,0.,3.])
  denominator=np.bincount(iv);avg=np.bincount(iv,weights=weight)/denominator
  density=float(avg.sum()/(3*len(vc)))
  stabilized=density*len(vc)/(len(vc)+9.)
  rawscore=float((3*np.sum(state[core]==3)+2*np.sum(state[core]==1)+np.sum(state[core]==0))/(3*len(small)))
  cs={name:int(np.sum(state[core]==s)) for s,name in [(0,'U'),(1,'T'),(3,'CONFLICT')]}
  cidx=core[state[core]==3]
  pairs=np.column_stack([e['top1_instance_id'][cidx],e['top2_instance_id'][cidx]])
  pairs=np.sort(pairs,axis=1)
  if len(pairs):
   up,pc=np.unique(pairs,axis=0,return_counts=True);pp=np.argsort(-pc)[:4]
   competition=[{'ids':up[p].tolist(),'points':int(pc[p])} for p in pp]
  else:competition=[]
  stable=context[inst[context]>0]
  stable_ids,stable_counts=np.unique(inst[stable],return_counts=True)
  ss=np.argsort(-stable_counts)[:8]
  rows.append({'candidate_index':k,'parent_component':groups[k],
    'score_density':density,'score_priority':stabilized,'score_reference_raw':rawscore,
    'score_stability_factor':float(len(vc)/(len(vc)+9.)),
    'core_points':len(core),'core_counts':cs,'context_points':len(context),
    'ranking_neighbourhood_points':len(small),'ranking_voxels_1cm':len(vc),
    'stable_context_points':len(stable),'inferred_context_points':int(np.sum((inst[context]>0)&(state[context]!=2))),
    'bbox_min':xyz[context].min(0).tolist(),'bbox_max':xyz[context].max(0).tolist(),
    'core_bbox_min':xyz[core].min(0).tolist(),'core_bbox_max':xyz[core].max(0).tolist(),
    'center':xyz[core].mean(0).tolist(),'extent_m':np.ptp(xyz[context],axis=0).tolist(),
    'dominant_competing_ids':competition,
    'context_instances':[{'id':int(stable_ids[j]),'points':int(stable_counts[j])} for j in ss],
    'median_source_frame_votes':float(np.median(e['total_frame_votes'][core])),
    'conflict_margin_median':float(np.median(e['margin'][cidx])) if len(cidx) else None,
    'conflict_frame_votes_median':float(np.median(e['total_frame_votes'][cidx])) if len(cidx) else None})
  if (k+1)%100==0:log('Expanded',k+1,'/',len(cores))
 assert np.all(core_to[seed_mask]>=0) and np.all(core_to[~seed_mask]==-1)
 co,cp=packed(cores);ro,rp=packed(contexts);so,sp=packed(density_regions)
 np.savez_compressed(OUT/'roi_membership.npz',core_offsets=co,core_point_index=cp,
    context_offsets=ro,context_point_index=rp,density_offsets=so,density_point_index=sp,
    point_core_candidate=core_to,voxel_id_1cm=voxel)
 ranked=sorted(rows,key=lambda r:(-r['score_priority'],-(r['conflict_frame_votes_median'] or 0),r['candidate_index']))
 for rank,r in enumerate(ranked,1):r['rank_density_only']=rank
 (OUT/'roi_candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 manifest={'method':'P1-A1','scene':'room0','scope':'final unassigned three-state PLY (145379 alarm points)',
  'source_files':{'native_evidence':str(NATIVE),'final_map':str(FINAL),'surface_ply':str(FULL)},
  'source_hashes':hashes,'trigger_counts':dict(zip(['U','T','CONFLICT'],[int(np.sum(seed_mask&(state==s))) for s in [0,1,3]])),
  'all_alarm_points_retained_once':True,'minimum_alarm_point_threshold':None,
  'initial_components':len(keep),'oversized_components_split':split_parents,'roi_count':len(rows),
  'settings':{'graph':asdict(settings),'closure_graph_rings':1,'oversized_core_extent_m':.60,
    'oversized_core_geodesic_partition_radius_m':.30,'rank_density_graph_rings':3,
    'repair_context_geodesic_halo_m':.10,'repair_context_spatial_comparison_halo_m':.06,
    'rank_density_metric_radius_floor_m':.03,'density_equal_area_voxel_m':.01,'score_smoothing_equivalent_voxels':9},
  'ranking':'100 * area-normalized density * n_voxels/(n_voxels+9); density = mean per occupied 1cm voxel of (3C + 2T + U)/(3*surface points) in three-ring/3cm neighbourhood; numerator only current core alarms',
  'raw_reference_score_retained':True,'GT_used':False,'map_labels_written':False,
  'context_overlap_policy':'nearby alarms joined before expansion; halo overlap alone never transitively merges cores',
  'limits':['priority measures suspicion, not proven errors','10cm halo is an initial review area, not a guarantee of whole-object coverage','three-state triggers can miss stable wrong assignments']}
 assert sha(NATIVE)==hashes['native_evidence'] and sha(FINAL)==hashes['final_map']
 manifest['input_maps_unchanged']=True
 (OUT/'detector_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 log('COMPLETE',json.dumps({k:manifest[k] for k in ['trigger_counts','initial_components','roi_count','input_maps_unchanged']}))
 log('TOP_DENSITY',json.dumps([{k:r[k] for k in ['candidate_index','score_priority','score_density','core_points','core_counts','context_points','stable_context_points','extent_m']} for r in ranked[:12]],ensure_ascii=False))

if __name__=='__main__':main()
