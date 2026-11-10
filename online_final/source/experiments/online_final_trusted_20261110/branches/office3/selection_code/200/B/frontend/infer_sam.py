"""Identical crop RGB, official SAM2.1 AMG defaults, raw proposals retained."""
from pathlib import Path
import argparse, hashlib, json, os, inspect, time, shutil
import numpy as np, torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from prepare_inputs import sha,write,link_verified

ROOT=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110/online/office3/200/B/candidates/frontend')
CHECKPOINT=Path('/data/chenkejun/CVPR/models/sam3.pt')
PARAMS=dict(points_per_side=32,points_per_batch=64,pred_iou_thresh=.8,stability_score_thresh=.95,
 stability_score_offset=1.,mask_threshold=0.,box_nms_thresh=.7,crop_n_layers=0,crop_nms_thresh=.7,
 crop_overlap_ratio=512/1500,crop_n_points_downscale_factor=1,min_mask_region_area=0,
 output_mode='binary_mask',use_m2m=False,multimask_output=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args()
 manifest=json.loads((ROOT/'input_manifest.json').read_text())
 assert sha(CHECKPOINT)=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
 np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 model=build_sam3_image_model(device='cuda',checkpoint_path=str(CHECKPOINT),load_from_HF=False,enable_inst_interactivity=True)
 processor=Sam3Processor(model,device='cuda');pred=model.inst_interactive_predictor
 pred._transforms.max_hole_area=0.;pred._transforms.max_sprinkle_area=0.
 class PredictorAdapter:
  def __init__(self):self.device=pred.device;self._transforms=pred._transforms
  def set_image(self,image):
   state=processor.set_image(Image.fromarray(image));backbone=state['backbone_out']['sam2_backbone_out'];_,features,_,_=pred.model._prepare_backbone_features(backbone);features[-1]=features[-1]+pred.model.no_mem_embed
   features=[f.permute(1,2,0).view(1,-1,*size) for f,size in zip(features[::-1],pred._bb_feat_sizes[::-1])][::-1]
   pred._features={'image_embed':features[-1],'high_res_feats':features[:-1]};pred._is_image_set=True;pred._orig_hw=[image.shape[:2]]
  def reset_predictor(self):pred.reset_predictor()
  def _predict(self,*args,**kwargs):return pred._predict(*args,**kwargs)
 generator=SAM2AutomaticMaskGenerator(pred.model,**PARAMS);generator.predictor=PredictorAdapter()
 write(ROOT/'sam2_1/model_provenance.json',dict(model='SAM3 image interactivity',checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),parameters=PARAMS,generator_sha256=sha(inspect.getfile(SAM2AutomaticMaskGenerator)),code_sha256=sha(__file__),GT_used=False,GPU=torch.cuda.get_device_name(),GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),compatibility_directory='sam2_1 legacy path only; actual model SAM3'))
 jobs=sorted(manifest['jobs'],key=lambda j:(j['kind']!='control',j['key']))
 if args.limit:jobs=jobs[:args.limit]
 done={};rows=[];beg=time.perf_counter()
 for idx,job in enumerate(jobs,1):
  key=job['key'];png=ROOT/'sam2_1'/(key+'.png');npz=png.with_suffix('.npz');recpath=png.with_suffix('.json')
  if recpath.exists():
   rec=json.loads(recpath.read_text());assert sha(png)==rec['mask_sha256'];assert sha(npz)==rec['proposals_sha256'];assert rec['RGB_sha256']==job['RGB_sha256']
  else:
   assert sha(job['input_file'])==job['RGB_sha256'];t=time.perf_counter();duplicate=job['same_RGB_as']
   cached_duplicate=ROOT/'sam2_1'/(str(duplicate)+'.json')
   if duplicate and (duplicate in done or cached_duplicate.exists()):
    old=done.get(duplicate) or json.loads(cached_duplicate.read_text())
    assert old['RGB_sha256']==job['RGB_sha256'] and sha(old['mask_file'])==old['mask_sha256'] and sha(old['proposal_file'])==old['proposals_sha256']
    link_verified(old['mask_file'],png,old['mask_sha256']);link_verified(old['proposal_file'],npz,old['proposals_sha256']);rec=dict(old);rec['reuse_from']=duplicate
   else:
    rgb=np.array(Image.open(job['input_file']).convert('RGB'),copy=True)
    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):proposals=generator.generate(rgb)
    for p in proposals:p['content_hash']=hashlib.sha256(np.packbits(p['segmentation']).tobytes()).hexdigest()
    proposals.sort(key=lambda p:(-int(p['area']),float(p['predicted_iou']),float(p['stability_score']),p['content_hash']))
    masks=np.stack([p['segmentation'] for p in proposals]).astype(bool) if proposals else np.zeros((0,*rgb.shape[:2]),bool)
    labels=np.zeros(rgb.shape[:2],np.uint16)
    for lid,m in enumerate(masks,1):labels[m]=lid
    assert np.array_equal(labels>0,masks.any(0))
    np.savez_compressed(npz,masks=masks,scores=np.asarray([p['predicted_iou'] for p in proposals],np.float32),
     stability=np.asarray([p['stability_score'] for p in proposals],np.float32),area=np.asarray([p['area'] for p in proposals],np.int64))
    Image.fromarray(labels).save(png)
    rec=dict(raw_proposals=len(proposals),network_input_wh=[1008,1008],reuse_from=None,visible_mask_count=int(len(np.unique(labels[labels>0]))))
   labels=np.asarray(Image.open(png));assert list(labels.shape)==job['shape']
   with np.load(npz) as z:original_masks=z['masks']
   for lid in np.unique(labels[labels>0]):assert np.all(original_masks[int(lid)-1][labels==lid])
   del original_masks
   rec.update(key=key,mask_file=str(png),proposal_file=str(npz),mask_sha256=sha(png),proposals_sha256=sha(npz),
    RGB_sha256=job['RGB_sha256'],seconds=time.perf_counter()-t,coverage_fraction=float((labels>0).mean()))
   write(recpath,rec)
  done[key]=rec;rows.append(rec)
  if idx%10==0 or idx==1:print('SAM',idx,len(jobs),key,round(time.perf_counter()-beg,1),flush=True)
  write(ROOT/'sam2_1/progress.json',dict(status='RUNNING',done=idx,total=len(jobs),last=key,seconds=time.perf_counter()-beg))
 audit=dict(status='PASS' if len(rows)==len(manifest['jobs']) else 'SMOKE',processed=len(rows),total=len(manifest['jobs']),
  raw_proposals=sum(r['raw_proposals'] for r in rows),union_preserved=True,each_partition_pixel_in_original_proposal=True,
  manual_prompts=False,GT_used=False,seconds=time.perf_counter()-beg)
 write(ROOT/'sam2_1/audit.json',audit);write(ROOT/'sam2_1/progress.json',audit);print('SAM_COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':main()
