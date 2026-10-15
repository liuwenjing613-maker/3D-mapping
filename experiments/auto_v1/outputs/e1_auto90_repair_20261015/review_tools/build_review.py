"""Build full tracking playback and paired seed-view audit after E1 freezes."""
import sys
from pathlib import Path
UI=Path(__file__).resolve().parent
sys.path.insert(0,str(UI.parent))
from bootstrap import C,R,S,digest
from pathlib import Path
import json,collections,shutil,colorsys,zipfile,statistics
import numpy as np
from PIL import Image
from plyfile import PlyData,PlyElement
from observation_ops import visible_samples
import run_repairs as r
O=R/'review';read=lambda p:json.loads(Path(p).read_text());load=lambda p:dict(np.load(p))
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 assert read(R/'pipeline_complete.json')['status']=='PASS'
 assert read(R/'evaluation_complete.json')['status']=='PASS'
 assert read(R/'review_assets_status.json')['status']=='PASS'
 conflict_report=read(R/'conflict_assets_complete.json');assert conflict_report['status']=='PASS'
 seed_conflicts=collections.defaultdict(list)
 for record in conflict_report['rows']:
  for pair in record['seed_conflicts']:seed_conflicts[pair['track_id']].append(pair)
 e1=R/'e1/E1_expanded_history/room2';beforepath=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/final/instance_surface.npz');z0=load(beforepath);z1=load(e1/'final/instance_surface.npz')
 np.testing.assert_array_equal(z0['xyz_m'],z1['xyz_m']);np.testing.assert_array_equal(z0['rgb'],z1['rgb']);xyz=z0['xyz_m'];before=z0['instance_id'];after=z1['instance_id']
 decisions={x['frame']:x for x in read(R/'validated/full_track/frame_decisions.json')['frames']};assert len(decisions)==400
 ids=read(R/'seed_id_map.json')['objects'];display={x['track_id']:x for x in read(C/'display_metadata.json')['objects']}
 assoc={x['canonical_track_id']:x for x in read(R/'global_association.json')['objects']}
 regions={x['persistent_id']:x for x in read(R/'e1/region_report.json')['targets']};cores={x['track_id']:x for x in read(R/'e1/objects.json')};profiles=[];scene=r.Scene('room2')
 for row in ids:
  tid=row['track_id'];p=row['source'];fid=row['frame'];uid=row['case_uid'];pid=assoc[tid]['persistent_id'];d=display[tid];assert row['group_id']==d['group_id'] and row['mask_id']==d['mask_id']
  frames=[];reasons=collections.Counter()
  for f,decision in sorted(decisions.items()):
   check=decision['objects'][str(tid)];initial=decision['initial_objects'][str(tid)];accept=tid in decision['accepted_track_ids']
   assert bool(check['accepted'])==accept
   if not accept:reasons.update(check['reasons'])
   frames.append({'frame':f,'accepted':accept,'reasons':check['reasons'],'visible_samples':check['visible_seed_samples'],'visible_fraction':check['visible_seed_fraction'],'coverage':check['tracked_coverage'],'bbox_fraction':check['new_surface_bbox_fraction'],'initial_accepted':initial['accepted'],'initial_coverage':initial['tracked_coverage'],'initial_reasons':initial['reasons'],'overlap_rechecked':check.get('rechecked_after_overlap_abstention',False)})
  support=load(R/'cases'/uid/'seed_projected_support.npz');points=np.unique(support['surface_point_index'][support['track_id']==tid]);mask=np.asarray(Image.open(S/p['mask']))>0
  fr=scene.source.load_frame(fid);rgb=fr.rgb.copy();color=np.asarray(d['color']);seedrgb=rgb.copy();seedrgb[mask]=(rgb[mask]*.30+color*.70).astype(np.uint8)
  ys,xs=np.where(mask);box=(max(0,int(xs.min())-35),max(0,int(ys.min())-35),min(1200,int(xs.max())+36),min(680,int(ys.max())+36));vis,u,v=visible_samples(xyz,fr.camera_to_world,fr.camera,fr.depth_m,.02)
  local=np.flatnonzero(vis&(u>=box[0])&(u<box[2])&(v>=box[1])&(v<box[3]));images={}
  for name,arr in [('seed',seedrgb),('before',rgb.copy()),('after',rgb.copy())]:
   if name!='seed':
    labels=before if name=='before' else after;a=labels[local];colors=np.tile(np.array([242,146,54],np.uint8),(len(local),1));colors[a==pid]=[24,109,224];colors[a<=0]=[163,169,180]
    # Depth-valid 3D points only, sorted far to near for deterministic overlap.
    cam=(xyz[local]-fr.camera_to_world[:3,3])@fr.camera_to_world[:3,:3];order=np.argsort(-cam[:,2],kind='stable')
    ii=local[order];colors=colors[order];xx=u[ii];yy=v[ii]
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
     x=np.minimum(xx+dx,1199);y=np.minimum(yy+dy,679);arr[y,x]=colors
   im=Image.fromarray(arr).crop(box);im.thumbnail((720,500));path=O/'views'/f'{tid:03d}_{name}.jpg';path.parent.mkdir(exist_ok=True);im.save(path,quality=92);images[name]=str(path.relative_to(O))
  item={**d,'frame':fid,'case_uid':uid,'persistent_id':pid,'association':assoc[tid]['association'],'area':p['area'],'selection_support':p['support_count'],'E1_support':len(points),
   'accepted_frames':sum(x['accepted'] for x in frames),'initial_pass_frames':sum(x['initial_accepted'] for x in frames),'overlap_rejected_frames':sum(x['initial_accepted'] and not x['accepted'] and x['overlap_rechecked'] for x in frames),'seed_conflicts':sorted(seed_conflicts[tid],key=lambda x:-x['pixels']),'rejection_reasons':dict(reasons),'frames':frames,'images':images,'core_points':regions[pid]['core_points'],'expanded_points':regions[pid]['final_points'],'core_profile':cores[tid]['profile'],
   'before_own_seed_points':int((before[points]==pid).sum()),'after_own_seed_points':int((after[points]==pid).sum()),'before_gray_seed_points':int((before[points]<=0).sum()),'after_gray_seed_points':int((after[points]<=0).sum()),
   'before_global_own_points':int((before==pid).sum()),'after_global_own_points':int((after==pid).sum())}
  profiles.append(item);print('REVIEW',tid,item['accepted_frames'],item['core_points'],item['expanded_points'],flush=True)
 counts=read(R/'validated/full_track/complete.json')['counts'];summary=read(R/'evaluation_complete.json');changed=before!=after
 stats={'seeds':90,'tracking_cases':59,'mapping_frames':400,'raw_tracking_frames_per_case':2000,'seed_pixels_preserved':True,'retained_source_policy':'SAM only for windows/cabinets',
  'seeds_with_accepted_frames':sum(x['accepted_frames']>0 for x in profiles),'zero_accepted_track_ids':[x['track_id'] for x in profiles if x['accepted_frames']==0],
  'empty_core_track_ids':[x['track_id'] for x in profiles if x['core_points']==0],'accepted_object_frame_events':sum(x['accepted_frames'] for x in profiles),'updated_mapping_frames':counts['updated_frames'],
  'changed_final_points_vs_prerepair':int(changed.sum()),'unassigned_to_assigned':int(((before<=0)&(after>0)).sum()),'assigned_to_unassigned':int(((before>0)&(after<=0)).sum()),
  'assigned_identity_changed':int(((before>0)&(after>0)&changed).sum()),'surface_points':len(xyz),'region':{k:v for k,v in read(R/'e1/region_report.json').items() if k not in ('targets','objects')},'evaluation':summary}
 timings=[]
 for case in read(R/'seed_diagnostics.json')['cases']:
  report=read(R/'cases'/case['case_uid']/'tracking/complete.json');assert report['status']=='PASS'
  timings.append({'case_uid':case['case_uid'],'objects':len(case['objects']),'seconds':report['elapsed_seconds'],'frames':report['total_frames']})
 groups=[]
 for n in sorted({x['objects'] for x in timings}):
  values=[x['seconds'] for x in timings if x['objects']==n]
  groups.append({'objects':n,'tasks':len(values),'min_seconds':min(values),'median_seconds':statistics.median(values),'max_seconds':max(values)})
 stats['runtime']={'GPU_model':'NVIDIA RTX 5880 Ada Generation','GPU_memory_MiB':49140,'tracking_by_object_count':groups,'tracking_tasks':timings,'pipeline_stages':read(R/'pipeline_complete.json')['stages'],'scope':'offline full sequence; includes initialization and IO; concurrent GPU jobs; not online latency'}
 stats['overlap_rejected_object_frame_events']=sum(x['overlap_rejected_frames'] for x in profiles)
 stats['lost_all_frames_after_overlap_track_ids']=[x['track_id'] for x in profiles if x['initial_pass_frames'] and not x['accepted_frames']]
 data={'stats':stats,'objects':profiles,'per_GT':read(R/'evaluation_r2/per_GT.json'),'source':{'baseline':str(beforepath),'result':str(e1/'final/instance_surface.npz'),'seed_manifest':str(S/'seed_manifest.json'),'seed_manifest_sha256':digest(S/'seed_manifest.json')},'checks':read(e1/'complete.json')['checks']}
 dump(O/'data.json',data);dump(O/'summary.json',stats)
 # Complete fixed surface, preserving its index and identity as PLY properties.
 color=np.empty((len(after),3),np.uint8)
 for pid in np.unique(after):color[after==pid]=[153,153,153] if pid<=0 else np.asarray(colorsys.hsv_to_rgb((int(pid)*.61803398875)%1,.67,.94))*255
 array=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),('surface_index','<i4')])
 for j,n in enumerate(['x','y','z']):array[n]=xyz[:,j]
 for j,n in enumerate(['red','green','blue']):array[n]=color[:,j]
 array['instance_id']=after;array['surface_index']=np.arange(len(xyz));ply=O/'room2_auto90_E1_instances.ply';PlyData([PlyElement.describe(array,'vertex')],text=False,byte_order='<').write(ply);back=PlyData.read(ply)['vertex'].data
 for n in array.dtype.names:np.testing.assert_array_equal(array[n],back[n])
 dump(O/'export_receipt.json',{'sha256':digest(ply),'points':len(array),'all_fields_roundtrip_exact':True,'full_surface':True})
 for src,name in [(R/'e1/policy.json','E1_policy.json'),(R/'input_freeze.json','input_freeze.json'),(R/'e1/E1_expanded_history/predictions_freeze.json','predictions_freeze.json'),(R/'evaluation_r2/protocol_freeze.json','protocol_freeze.json'),(C/'adapter_provenance.json','adapter_provenance.json'),(R/'tests_complete.json','tests_complete.json')]:shutil.copyfile(src,O/name)
 for name in ['review.html','review.css','review.js']:shutil.copyfile(UI/name,O/('index.html' if name=='review.html' else name))
 dump(O/'review_provenance.json',{'code':{str(p):digest(p) for p in UI.glob('*') if p.is_file()},'conflict_assets_receipt_sha256':digest(R/'conflict_assets_complete.json'),'repair_algorithms_unchanged':True})
 dump(R/'report_complete.json',{'status':'PASS','seeds':90,'all_mapping_frames_playable':True,'PLY_sha256':digest(ply),'summary':stats})
 archive=R/'review.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in sorted(O.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(O).as_posix())
 dump(R/'review_archive.json',{'file':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'files':len([p for p in O.rglob('*') if p.is_file()])})
 print('REVIEW_COMPLETE',json.dumps(stats),flush=True)
if __name__=='__main__':main()
