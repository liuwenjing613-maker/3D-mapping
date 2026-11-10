from pathlib import Path
import os,sys,json,time,hashlib,traceback
sys.path += ['/home/chenkejun/beauty/conceptgraphs/envs/cg-main/lib/python3.10/site-packages','/home/chenkejun/beauty/conceptgraphs/envs/cg-ali/lib/python3.10/site-packages']
import numpy as np
from PIL import Image
import torch
from sam3.model_builder import build_sam3_video_predictor
root=Path(sys.argv[1]);mid=sys.argv[2];track=root/'tracking_sam3';out=track/mid;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
def write(d):q=out/'status.tmp';q.write_text(json.dumps(d,indent=2)+'\n');q.replace(out/'status.json')
try:
 sel=json.loads((root/'candidates/fixedH/result.json').read_text());r=next(x for x in sel['selected_records'] if x['mask_id']==mid);path=root/'candidates/v1/masks'/f'{mid}.png';digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==r['mask_sha256'];mask=np.asarray(Image.open(path))>0;assert mask.shape==(680,1200) and mask.any();fid=int(r['frame']);ids=json.loads((track/'frame_ids.json').read_text())['frame_ids'];anchor=ids.index(fid);n=len(ids);video=track/'rgb';assert len(list(video.glob('*.jpg')))==n
 (out/'masks').mkdir(exist_ok=True);Image.fromarray(mask.astype(np.uint8)*255).save(out/'masks'/f'{fid:06d}.png');torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.7,0);t=time.monotonic();p=build_sam3_video_predictor(checkpoint_path='/data/chenkejun/CVPR/models/sam3.pt',gpus_to_use=[0],async_loading_frames=False,apply_temporal_disambiguation=False,compile=False);model_seconds=time.monotonic()-t;stats={}
 for direction in ('forward','backward'):
  t=time.monotonic();sid=p.handle_request({'type':'start_session','resource_path':str(video),'offload_video_to_cpu':True,'offload_state_to_cpu':True})['session_id'];p.handle_request({'type':'add_mask','session_id':sid,'frame_index':anchor,'obj_id':1,'mask':mask});count=0
  for v in p.handle_stream_request({'type':'propagate_in_video','session_id':sid,'propagation_direction':direction,'start_frame_index':anchor,'force_tracker_propagation':True,'evict_cached_frame_outputs':True}):
   i=int(v['frame_index'])
   if i==anchor:continue
   assert (direction=='forward' and i>anchor) or (direction=='backward' and i<anchor)
   outputs=v['outputs'];oid=np.asarray(outputs['out_obj_ids']);bm=np.asarray(outputs['out_binary_masks']);assert bm.shape[1:]==mask.shape and len(oid)==len(bm);hit=np.flatnonzero(oid==1);assert len(hit)<=1;bitmap=bm[hit[0]] if len(hit) else np.zeros(mask.shape,bool);Image.fromarray(bitmap.astype(np.uint8)*255).save(out/'masks'/f'{ids[i]:06d}.png');count+=1
   if count%25==0:write({'status':'RUNNING','stage':'SAM3_TRACKING_ONLY','direction':direction,'count':count,'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'elapsed_seconds':time.monotonic()-start})
  p.handle_request({'type':'close_session','session_id':sid});expected=n-1-anchor if direction=='forward' else anchor;assert count==expected,(direction,count,expected);stats[direction]={'count':count,'seconds':time.monotonic()-t}
 p.shutdown();assert {x.name for x in (out/'masks').glob('*.png')}=={f'{i:06d}.png' for i in ids};write({'status':'PASS','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','mask_id':mid,'seed_frame':fid,'seed_sha256':digest,'model':'SAM3','gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'model_seconds':model_seconds,'directions':stats,'elapsed_seconds':time.monotonic()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/2**30,'frame_count':n,'GT_used':False})
except BaseException:write({'status':'FAILED','stage':'SAM3_TRACKING_ONLY','mask_id':mid,'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start});raise
