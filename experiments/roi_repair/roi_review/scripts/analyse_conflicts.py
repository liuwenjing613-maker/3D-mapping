"""Read-only per-ROI evidence diagnosis; GT assists names after ROI selection."""
from pathlib import Path
import json,time,hashlib,re,zipfile
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from plyfile import PlyData
from detect_rois import sha
from select_history import project_visible

SRC=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
ROI=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002')
OUT=ROI/'conflict_diagnosis';OUT.mkdir(exist_ok=True)
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0')
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def classname(classes,i):
 v=classes.get(str(i),classes.get(i,str(i))) if isinstance(classes,dict) else classes[i-1] if 1<=i<=len(classes) else 'unlabeled'
 return v.get('name',v.get('class_name',str(v))) if isinstance(v,dict) else str(v)
def box(uv,valid,c):
 a=uv[valid]
 lo=a.min(0);hi=a.max(0);mid=(lo+hi)/2;size=np.maximum((hi-lo+1)*1.55,[440,320])
 low=np.maximum([0,0],np.floor(mid-size/2)).astype(int);high=np.minimum([c['width'],c['height']],np.ceil(mid+size/2)).astype(int)
 return [int(low[0]),int(low[1]),int(high[0]),int(high[1])]
def main():
 ranked=json.loads((ROI/'roi_ranked.json').read_text());sel=json.loads((ROI/'review_selection.json').read_text())
 ranks=sel['distinct_component_representatives']+list(set(sel['additional_U_T_examples'])-set(sel['distinct_component_representatives']))
 evidence_file=SRC/'native/room0/P1-A1/surface_p0/surface_evidence.npz';final_file=SRC/'final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
 source_hashes={'evidence':sha(evidence_file),'final':sha(final_file)}
 with np.load(evidence_file) as z:e={k:z[k] for k in z.files}
 with np.load(final_file) as z:xyz=z['xyz_m'];inst=z['instance_id']
 v=PlyData.read(SRC/'final/room0/P1-A1/room0_final_reference_v2_full.ply')['vertex'].data
 colors=np.column_stack([v[k] for k in ['red','green','blue']]);color_by_id={int(i):colors[np.flatnonzero(inst==i)[0]].tolist() for i in np.unique(inst[inst>0])}
 tree=cKDTree(xyz);assigned=np.flatnonzero(inst>0);assigned_tree=cKDTree(xyz[assigned])
 m=np.load(ROI/'roi_membership.npz');co=m['core_offsets'];cp=m['core_point_index'];ro=m['context_offsets'];rp=m['context_point_index'];core_to=m['point_core_candidate']
 # Semantic names are independent, post-hoc diagnostic annotations. These GT
 # labels never participate in detection, ranks, vote computation or map writes.
 with np.load(BASE/'ground_truth/gt.npz') as z:
  gtxyz=z['xyz_ref'];gtsem=z['semantic_id'];gtinst=z['instance_id'];gtmeta=json.loads(str(z['metadata_json']))
 classes=json.loads(Path(gtmeta['source_manifest']).read_text())['semantic_classes']
 log('Semantic dictionary',str(classes)[:220])
 dist,near=assigned_tree.query(gtxyz,workers=8);mapped=inst[assigned[near]];valid=(dist<=.01)&(gtsem>=0)
 label_info={}
 for i in np.unique(inst[inst>0]):
  ids=np.flatnonzero(valid&(mapped==i));sem,n=np.unique(gtsem[ids],return_counts=True);order=np.argsort(-n)
  label_info[int(i)]={'semantic_reference':[{'semantic_id':int(sem[j]),'name':classname(classes,int(sem[j])),'GT_vertices':int(n[j]),'fraction':float(n[j]/max(1,len(ids)))} for j in order[:3]],'point_count':int(np.sum(inst==i))}
 label_trees={};records=[];targets=[]
 for rank in ranks:
  r=ranked[rank-1];k=r['candidate_index'];core=cp[co[k]:co[k+1]];context=rp[ro[k]:ro[k+1]]
  cf=core[e['state'][core]==3];t=core[e['state'][core]==1]
  assert len(cf)==r['core_counts']['CONFLICT'] and len(t)==r['core_counts']['T']
  nd,nn=assigned_tree.query(xyz[core],workers=8);nearest_ids=inst[assigned[nn]]
  near_ids,near_n=np.unique(nearest_ids[nd<=.06],return_counts=True);order=np.argsort(-near_n)
  nearest=[{'id':int(near_ids[j]),'alarms_with_nearest_assigned_ID_within_6cm':int(near_n[j])} for j in order[:5]]
  pairs=[]
  if len(cf):
   pair=np.sort(np.column_stack([e['top1_instance_id'][cf],e['top2_instance_id'][cf]]),axis=1)
   up,cnt=np.unique(pair,axis=0,return_counts=True)
   for j in np.argsort(-cnt)[:3]:
    a,b=map(int,up[j]);points=cf[np.all(pair==up[j],axis=1)]
    av=np.where(e['top1_instance_id'][points]==a,e['top1_votes'][points],e['top2_votes'][points]);bv=np.where(e['top1_instance_id'][points]==b,e['top1_votes'][points],e['top2_votes'][points])
    meda=float(np.median(av));medb=float(np.median(bv));p=int(points[np.argmin(abs(av-meda)+abs(bv-medb))])
    pinfo={'ids':[a,b],'conflict_points':int(cnt[j]),'share_of_conflicts':float(cnt[j]/len(cf)),
     'vote_medians':[meda,medb],'sample_point_index':p,'sample_xyz_m':xyz[p].tolist(),
     'sample_top1_id':int(e['top1_instance_id'][p]),'sample_top1_votes':int(e['top1_votes'][p]),
     'sample_top2_id':int(e['top2_instance_id'][p]),'sample_top2_votes':int(e['top2_votes'][p]),
     'sample_total_votes':int(e['total_frame_votes'][p]),'sample_confidence':float(e['confidence'][p]),
     'sample_other_votes':int(e['total_frame_votes'][p]-e['top1_votes'][p]-e['top2_votes'][p])}
    targets.append((len(records),len(pairs),p));pairs.append(pinfo)
  tids,tn=np.unique(e['top1_instance_id'][t],return_counts=True);to=np.argsort(-tn)
  tentative=[{'id':int(tids[j]),'points':int(tn[j]),'votes_per_point_unique':np.unique(e['total_frame_votes'][t[e['top1_instance_id'][t]==tids[j]]]).tolist()} for j in to[:4]]
  ids=set(i for p in pairs for i in p['ids'])|set(p['id'] for p in nearest)|set(p['id'] for p in r['context_instances'])|set(p['id'] for p in tentative)
  current=[]
  for i in sorted(ids):
   info=label_info.get(i,{'semantic_reference':[],'point_count':0}).copy();info['id']=i;info['RGB']=color_by_id.get(i,[89,89,89])
   selpoints=np.flatnonzero(inst==i)
   if len(selpoints):
    if i not in label_trees:label_trees[i]=cKDTree(xyz[selpoints])
    d=label_trees[i].query(xyz[core],workers=8)[0]
    info['minimum_distance_to_alarm_m']=float(d.min());info['median_distance_to_alarm_m']=float(np.median(d))
   else:info['minimum_distance_to_alarm_m']=None;info['median_distance_to_alarm_m']=None
   info['assigned_points_in_ROI']=int(np.sum(inst[context]==i));current.append(info)
  records.append({'roi_id':r['roi_id'],'rank':rank,'counts':r['core_counts'],'context_instances':r['context_instances'],
   'nearest_instances':nearest,'competition_pairs':pairs,'tentative_candidates':tentative,'instance_details':current,
   'primary_frame':r['selected_views'][0]['frame_id'],'candidate_index':k})
 # Reconstruct each representative point's actual frame/ID votes from immutable
 # per-frame supports, rather than interpreting a colored map as vote evidence.
 lookup={}
 for line in (SRC/'native/room0/P1-A1/association/associations.jsonl').open():
  row=json.loads(line);lookup.setdefault(row['frame_id'],{})[row['mask_local_id']]=row['instance_id']
 target_points=np.array(sorted(set(p for _,_,p in targets)),np.int32);target_lut=np.full(len(xyz),-1,np.int32);target_lut[target_points]=np.arange(len(target_points))
 events=defaultdict(list)
 support=SRC/'raw_inputs/room0/surface_regions/frame_region_support'
 for fid in range(0,2000,5):
  with np.load(support/f'f{fid:06d}.npz') as z:pi=z['surface_point_index'];local=z['mask_local_id']
  hit=target_lut[pi]>=0
  if hit.any():
   for p,lid in set(zip(pi[hit].tolist(),local[hit].tolist())):events[int(p)].append({'frame':fid,'mask_local_id':int(lid),'id':int(lookup[fid][lid])})
  if fid%400==0:log('Trace supports',fid,'/ 1995')
 for ri,pair_no,p in targets:
  info=records[ri]['competition_pairs'][pair_no];info['source_support']={}
  sets=[]
  for i in info['ids']:
   ev=[x for x in events[p] if x['id']==i];frames=sorted(set(x['frame'] for x in ev));sets.append(set(frames))
   expected=info['sample_top1_votes'] if i==info['sample_top1_id'] else info['sample_top2_votes']
   assert len(frames)==expected,(p,i,len(frames),expected)
   info['source_support'][str(i)]={'unique_source_frames':len(frames),'first_frame':min(frames),'last_frame':max(frames),'events':ev}
  info['same_source_frame_both_ids_count']=len(sets[0]&sets[1])
 log('Native vote reconstruction PASS',len(targets),'representative points')
 cfg=json.loads((BASE/'configs/raw.json').read_text());c=cfg['camera'];scene=Path(cfg['source']['scene_root']);masks=Path(cfg['source']['mask_root']);poses=np.loadtxt(scene/'traj.txt').reshape(-1,4,4)
 frames={}
 def frame(fid):
  if fid in frames:return frames[fid]
  rgb=np.asarray(Image.open(scene/f'results/frame{fid:06d}.jpg').convert('RGB'));depth=np.asarray(Image.open(scene/f'results/depth{fid:06d}.png'),np.float32)/c['depth_png_units_per_meter']
  raw=np.asarray(Image.open(masks/f'frame{fid:06d}.png'))
  frames[fid]=(rgb,depth,raw);return frames[fid]
 for rec in records:
  k=rec['candidate_index'];core=cp[co[k]:co[k+1]];context=rp[ro[k]:ro[k+1]];fid=rec['primary_frame'];rgb,depth,raw=frame(fid);pose=poses[fid]
  visible,uv=project_visible(xyz[context],pose,depth,c);crop=box(uv,visible,c);rec['crop_xyxy']=crop
  x0,y0,x1,y1=crop;vv,uu=np.indices((y1-y0,x1-x0));vv+=y0;uu+=x0;dep=depth[y0:y1,x0:x1]
  valid=(dep>0)&(dep<10);z=dep[valid];pc=np.column_stack([(uu[valid]-c['cx'])*z/c['fx'],(vv[valid]-c['cy'])*z/c['fy'],z]);world=pc@pose[:3,:3].T+pose[:3,3]
  d,idx=tree.query(world,workers=8);pmap=np.full(dep.shape,-1,np.int32);pmap.ravel()[np.flatnonzero(valid)[d<=.02]]=idx[d<=.02];good=pmap>=0
  imap=np.full(dep.shape,-1,np.int32);imap[good]=inst[pmap[good]]
  smap=np.full(dep.shape,255,np.uint8);in_core=np.zeros(dep.shape,bool);in_core[good]=core_to[pmap[good]]==k;smap[in_core]=e['state'][pmap[in_core]]
  t1=np.full(dep.shape,-1,np.int32);t2=t1.copy();t1[in_core]=e['top1_instance_id'][pmap[in_core]];t2[in_core]=e['top2_instance_id'][pmap[in_core]]
  np.savez_compressed(OUT/f'{rec["roi_id"]}_pixels.npz',rgb=rgb[y0:y1,x0:x1],instance_map=imap,state_map=smap,top1_map=t1,top2_map=t2,
     reference_colors=colors[np.maximum(0,pmap)],mapped=good)
  # Main conflict pair: one actual source mask supporting each opposing ID.
  source_views=[]
  if rec['competition_pairs']:
   pair=rec['competition_pairs'][0];p=pair['sample_point_index']
   for i in pair['ids']:
    ev=min(pair['source_support'][str(i)]['events'],key=lambda x:(abs(x['frame']-fid),x['frame'],x['mask_local_id']))
    sf=ev['frame'];lid=ev['mask_local_id'];srgb,sdepth,sraw=frame(sf);spose=poses[sf]
    vv,uu=np.indices(sraw[::2,::2].shape);v=vv*2;u=uu*2;dep=sdepth[::2,::2];eligible=(sraw[::2,::2]==lid)&(dep>0)&(dep<10)
    pu=u[eligible];pv=v[eligible];dz=dep[eligible];pc=np.column_stack([(pu-c['cx'])*dz/c['fx'],(pv-c['cy'])*dz/c['fy'],dz]);ww=pc@spose[:3,:3].T+spose[:3,3]
    distances,nn=tree.query(ww,workers=8);selected=(nn==p)&(distances<.015)
    assert selected.any(),(rec['roi_id'],i,sf,lid,p)
    at=np.flatnonzero(selected)[np.argmin(distances[selected])];ux=int(pu[at]);vy=int(pv[at]);crop=[max(0,ux-220),max(0,vy-160),min(c['width'],ux+220),min(c['height'],vy+160)]
    a,b,cc,dd=crop;np.savez_compressed(OUT/f'{rec["roi_id"]}_source_ID{i}.npz',rgb=srgb[b:dd,a:cc],mask=(sraw[b:dd,a:cc]==lid),source_pixel=np.array([ux-a,vy-b],np.int32))
    source_views.append({'id':i,'frame_id':sf,'mask_local_id':lid,'support_votes':pair['source_support'][str(i)]['unique_source_frames'],'source_pixel_uv':[ux,vy]})
  rec['source_views']=source_views;log('Rendered diagnostic arrays',rec['roi_id'])
 (OUT/'conflict_cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 assert sha(evidence_file)==source_hashes['evidence'] and sha(final_file)==source_hashes['final']
 audit={'status':'PASS','ROI_count':len(records),'representative_vote_points_verified':len(targets),
   'vote_counts_reconstructed_from_all_400_support_caches':True,'source_pixel_witnesses_exactly_reproject_to_representative_surface_point':True,
   'semantic_names':'GT semantic reference via 1cm geometry support; diagnostic annotations, not new benchmark scores',
   'GT_does_not_affect_ROI_detection_ranking_or_map':True,'map_hashes_unchanged':True,'source_hashes':source_hashes,'new_PLY_generated':False}
 (OUT/'diagnosis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 with zipfile.ZipFile(ROI/'conflict_diagnosis_arrays.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for p in OUT.iterdir():
   if p.is_file():z.write(p,p.name)
 log('COMPLETE',json.dumps(audit))
 log('PAIRS',json.dumps([{k:r[k] for k in ['roi_id','counts','instance_details']} for r in records],ensure_ascii=False))

if __name__=='__main__':main()
