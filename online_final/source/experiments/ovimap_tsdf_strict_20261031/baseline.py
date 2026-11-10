from pathlib import Path
import argparse,hashlib,json,os,sys,time
from dataclasses import asdict
import numpy as np
import cv2
C=Path(__file__).resolve().parent
SNAP=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map')
sys.path.insert(0,str(SNAP/'src'))
from revisable_instance_map.frame_io import Camera,Frame
from revisable_instance_map.observations import extract_frame_observations
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.surface_evidence import reduce_surface_votes,CONFIRMED
from strict_votes import IncrementalVotes,full_replay,BASE
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,d):
 p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def save(path,**d):
 p=Path(path);q=p.with_suffix('.tmp.npz');np.savez_compressed(q,**d);q.replace(p)
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sample(frame,lookup):
 mask=frame.mask_local[::2,::2];depth=frame.depth_m[::2,::2]
 valid=(mask>0)&np.isfinite(depth)&(depth>0)&(depth<10)
 y,x=np.nonzero(valid);z=depth[valid].astype(np.float64);cam=frame.camera
 xyz=np.column_stack(((x*2-cam.cx)*z/cam.fx,(y*2-cam.cy)*z/cam.fy,z))
 world=xyz@frame.camera_to_world[:3,:3].T+frame.camera_to_world[:3,3]
 table=np.zeros(int(frame.mask_local.max())+1,np.int64)
 for local,pid in lookup.items():table[local]=pid
 ids=table[mask[valid]];assert np.all(ids>0)
 return {'frame_id':frame.frame_id,'world':world,'ids':ids}
def get_frame(scene,fid,row,config):
 source=config['source'];root=Path(source['scene_root']);mroot=Path(source['mask_root'])
 rgbpath=root/source['rgb_pattern'].format(frame=fid)
 depthpath=root/source['depth_pattern'].format(frame=fid);maskpath=mroot/source['mask_pattern'].format(frame=fid)
 rgb=cv2.cvtColor(cv2.imread(str(rgbpath)),cv2.COLOR_BGR2RGB)
 depth=cv2.imread(str(depthpath),cv2.IMREAD_UNCHANGED).astype(np.float32)/config['camera']['depth_png_units_per_meter']
 mask=cv2.imread(str(maskpath),cv2.IMREAD_UNCHANGED)
 assert digest(depth)==row['depth_decoded_sha256'],('depth_mismatch',fid)
 assert digest(mask.astype(np.int32))==row['mask_decoded_sha256']
 camera=Camera(**{k:v for k,v in config['camera'].items() if k!='trajectory_matrix'})
 np.testing.assert_array_equal(camera.intrinsic,np.asarray(row['intrinsics']))
 # OVI actually integrates float32 poses. Use exactly those numeric values.
 pose=np.asarray(row['pose'],np.float64)
 frame=Frame(fid,rgb,depth,mask,pose,camera)
 return frame,sha(maskpath),{str(p):sha(p) for p in [rgbpath,depthpath,maskpath]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--count',type=int,required=True)
 p.add_argument('--native-root',type=Path,required=True);p.add_argument('--tag',default='main')
 p.add_argument('--verify-all',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
 a=p.parse_args();out=O/'ours'/(a.scene+'_'+str(a.count)+'_'+a.tag)
 out.mkdir(parents=True,exist_ok=True)
 for d in ['history','maps','checkpoints','decisions']: (out/d).mkdir(exist_ok=True)
 config=json.loads((Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs')/(a.scene+'.json')).read_text())
 config['frame_selection']['stop_exclusive']=a.count*5
 write(out/'config.json',config)
 history=[];history_hashes={};cache=IncrementalVotes()
 associator=OnlineVoxelAssociator(allow_multiple_observations_per_instance_per_frame=True,improved_association=False,
  valid_first_sampling=False,association_mode='probabilistic',relative_weight_temperature=.1,null_candidate_score=.35)
 start=0
 if a.resume:
  commit=json.loads((out/'checkpoint.json').read_text());start=commit['processed_frames']
  assert sha(out/commit['association'])==commit['association_sha256']
  assert sha(out/commit['votes'])==commit['votes_sha256']
  history_hashes=commit['history_sha256']
  for name,digest_value in history_hashes.items():assert sha(out/name)==digest_value
  associator=OnlineVoxelAssociator.from_checkpoint(out/commit['association'])
  cache=IncrementalVotes.load(out/commit['votes'])
  for count in range(1,start+1):
   row=load(out/'history'/('f%06d.npz'%((count-1)*5)))
   row['frame_id']=int(row['frame_id'][0]);history.append(row)
  assert cache.frames==tuple(r['frame_id'] for r in history)
  oldrows=[json.loads(line) for line in (out/'audit.jsonl').read_text().splitlines()]
  kept=[r for r in oldrows if r['processed_frames']<=start]
  assert [r['processed_frames'] for r in kept]==list(range(1,start+1))
  for r in kept:
   for name,digest_value in r['input_hashes'].items():assert sha(name)==digest_value
  (out/'audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kept))
 elif (out/'checkpoint.json').exists():raise RuntimeError('Existing checkpoint requires --resume')
 for count in range(start+1,a.count+1):
  fid=(count-1)*5;started=time.perf_counter()
  while True:
   f=a.native_root/'prefix_audit.jsonl'
   rows=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
   row=next((r for r in rows if r['frame_id']==fid),None)
   if row is not None:break
   if (a.native_root/'FAILED.json').exists():raise RuntimeError('Native producer failed')
   if time.perf_counter()-started>1800:raise TimeoutError('Native prefix missing')
   time.sleep(2)
  xyz=load(a.native_root/'geometry'/('f%06d.npz'%fid))['xyz_m']
  assert digest(xyz)==row['geometry_sha256']
  frame,masksha,inputs=get_frame(a.scene,fid,row,config)
  obs=extract_frame_observations(frame,'Replica',a.scene,masksha)
  decisions=associator.process_frame(frame,obs)
  assert len(decisions)==len(obs)
  write(out/'decisions'/('f%06d.json'%fid),decisions)
  sample_row=sample(frame,{d['mask_local_id']:d['instance_id'] for d in decisions});history.append(sample_row)
  save(out/'history'/('f%06d.npz'%fid),frame_id=np.array([fid],np.int32),world=sample_row['world'],ids=sample_row['ids'])
  history_hashes['history/f%06d.npz'%fid]=sha(out/'history'/('f%06d.npz'%fid))
  keys,votes=cache.refresh(xyz,history)
  verify=a.verify_all or count in [1,2,5,10,20,50,100,200,300,400]
  if verify:
   k,v=full_replay(xyz,history);np.testing.assert_array_equal(keys,k);np.testing.assert_array_equal(votes,v)
  if len(xyz):
   evidence=reduce_surface_votes(keys,votes,len(xyz),BASE,2,.67)
   labels=np.where(evidence['state']==CONFIRMED,evidence['top1_instance_id'],-1).astype(np.int32)
  else:labels=np.empty(0,np.int32);evidence={}
  save(out/'maps'/('f%06d.npz'%fid),xyz_m=xyz,instance_id=labels,native_instance_ids=np.unique(labels[labels>0]))
  report={'processed_frames':count,'frame_id':fid,'geometry_sha256':digest(xyz),'points':len(xyz),
   'assigned':int(np.sum(labels>0)),'published_ids':len(np.unique(labels[labels>0])),
   'valid_votes':int(votes.sum()),'strict_vote_verified':True,'full_replay_verified':verify,
   'input_hashes':inputs,'seconds':time.perf_counter()-started,**cache.stats}
  with (out/'audit.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
  write(out/'progress.json',report)
  if verify or count==a.stop_after:
   rel='checkpoints/f%06d'%fid
   associator.save_checkpoint(out/(rel+'_association.npz'));cache.save(out/(rel+'_votes.npz'))
   write(out/'checkpoint.json',{'processed_frames':count,'association':rel+'_association.npz','votes':rel+'_votes.npz','association_sha256':sha(out/(rel+'_association.npz')),'votes_sha256':sha(out/(rel+'_votes.npz')),'history_sha256':history_hashes})
  print('OURS_PREFIX',a.scene,count,len(xyz),report['published_ids'],round(report['seconds'],2),flush=True)
  if count==a.stop_after:
   write(out/'paused_test.json',report);return
 write(out/'complete.json',{'status':'PASS','scene':a.scene,'processed_frames':a.count,
  'no_repair':True,'no_hole_completion':True,'GT_read':False,'geometry_shared_with_OVI':True,'config':associator.checkpoint_parameters()})
if __name__=='__main__':main()
