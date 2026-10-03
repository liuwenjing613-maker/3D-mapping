"""Same native CropFormer config/weights/threshold as frozen room0 masks."""
from pathlib import Path
import os,sys,time,json,hashlib,argparse
import numpy as np,torch,cv2
from scipy.optimize import linear_sum_assignment
R=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003')
C=Path('/data/chenkejun/ovimap_runtime_20260908/detectron2/projects/CropFormer')
sys.path.insert(0,str(C));sys.path.insert(0,str(C/'demo_cropformer'))
from demo_from_dirs import setup_cfg
from predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def match_quality(a,b):
 ai=np.unique(a);bi=np.unique(b);nx=len(ai);ny=len(bi)
 joint=np.bincount(np.searchsorted(ai,a.ravel())*ny+np.searchsorted(bi,b.ravel()),minlength=nx*ny).reshape(nx,ny)
 union=joint.sum(1)[:,None]+joint.sum(0)[None,:]-joint
 iou=joint/np.maximum(1,union);rr,cc=linear_sum_assignment(-iou)
 agree=int(joint[rr,cc].sum())/a.size
 return {'best_one_to_one_pixel_agreement':agree,'matched_mask_mean_IoU':float(iou[rr,cc].mean()),'identical_label_image':bool(np.array_equal(a,b))}
np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
manifest=json.loads((R/'input_manifest.json').read_text())
weights=Path('/data/chenkejun/CVPR/models/CropFormer_hornet_3x_03823a.pth');config=C/'configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml'
cfg=setup_cfg(argparse.Namespace(config_file=str(config),opts=['MODEL.WEIGHTS',str(weights)]));model=VisualizationDemo(cfg)
(R/'model_config.yaml').write_text(cfg.dump())
settings={'weights':str(weights),'weights_sha256':sha(weights),'config_path':str(config),'config_sha256':sha(config),'predictor_source_sha256':sha(C/'demo_cropformer/predictor.py'),'MIN_SIZE_TEST':cfg.INPUT.MIN_SIZE_TEST,'MAX_SIZE_TEST':cfg.INPUT.MAX_SIZE_TEST,'ENTITY_CROP_AREA_RATIO':cfg.ENTITY.CROP_AREA_RATIO,'ENTITY_CROP_STRIDE_RATIO':cfg.ENTITY.CROP_STRIDE_RATIO,'ENTITY_CROP_SAMPLE_NUM_TEST':cfg.ENTITY.CROP_SAMPLE_NUM_TEST,'score_threshold':.5,'winner_policy':'all masks >=0.5; ascending score painting, highest score owns overlap; same as frozen cache','input':'lossless native-resolution RGB crop; standard predictor resize supplies effective zoom','device':torch.cuda.get_device_name(0),'GPU_visible':os.environ.get('CUDA_VISIBLE_DEVICES')}
(R/'model_provenance.json').write_text(json.dumps(settings,indent=2)+'\n')
# Controls first establish frozen-cache compatibility, then representative crops.
jobs=sorted(manifest['jobs'],key=lambda j:(j['kind']!='full_frame_control',not j.get('rank') in {1,23,35,38,48,49,58,61,63,67,73,74,226},j['key']))
records=[];outputs={}
for no,job in enumerate(jobs,1):
 key=job['key'];target=R/'inference'/f'{key}.png';record_file=R/'inference'/f'{key}.json'
 imgpath=Path(job['input_file']);assert sha(imgpath)==job['RGB_file_sha256'];image=read_image(str(imgpath),format='BGR')
 if target.exists() and record_file.exists():
  rec=json.loads(record_file.read_text());assert sha(target)==rec['mask_file_sha256'] and sha(imgpath)==rec['RGB_file_sha256']
  outputs[key]=(target,rec);records.append(rec);continue
 assert not target.exists() and not record_file.exists(),f'partial evidence needs inspection {key}'
 torch.cuda.synchronize();start=time.perf_counter();reuse=job['same_RGB_as']
 if reuse and reuse in outputs:
  src,source_rec=outputs[reuse];result=cv2.imread(str(src),cv2.IMREAD_UNCHANGED);scores=source_rec['selected_scores'];counts=source_rec['selected_mask_areas_before_paint'];infer_shape=source_rec['model_resized_full_image_shape'];reused_from=reuse
 else:
  pred=model.run_on_image(image)['instances'];keep=pred.scores>=.5;score=pred.scores[keep];masks=pred.pred_masks[keep]
  result=np.zeros(image.shape[:2],dtype=np.uint16);assert len(score)<65536
  for ix in torch.argsort(score):result[masks[ix].cpu().numpy()==1]=int(ix)+1
  scores=score.cpu().numpy().tolist();counts=masks.flatten(1).sum(1).cpu().numpy().tolist();reused_from=None
  # The same official shortest-edge transform used by the full-frame branch.
  transform=model.predictor.generate_img_augs()[0][0].get_transform(image);infer_shape=list(transform.apply_image(image).shape[:2])
  del pred,masks,score
 assert result.shape==tuple(job['shape']);assert cv2.imwrite(str(target),result)
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 rec={'key':key,'kind':job['kind'],'frame':job['frame'],'ROI':job.get('ROI'),'view':job.get('view'),'strategy':job.get('strategy'),'shape':list(result.shape),'visible_mask_count':int(len(np.unique(result[result>0]))),'selected_scores':scores,'selected_mask_areas_before_paint':counts,'seconds':seconds,'model_resized_full_image_shape':infer_shape,'RGB_file_sha256':sha(imgpath),'mask_file_sha256':sha(target),'reused_identical_RGB_prediction':reused_from}
 if job['kind']=='full_frame_control':
  baseline=cv2.imread(job['baseline_file'],cv2.IMREAD_UNCHANGED);rec['frozen_cache_comparison']=match_quality(baseline,result)
 record_file.write_text(json.dumps(rec,indent=2)+'\n');outputs[key]=(target,rec);records.append(rec)
 with (R/'inference_records.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 print(time.strftime('%H:%M:%S'),f'INFER {no}/{len(jobs)}',key,rec['visible_mask_count'],round(seconds,3),rec.get('frozen_cache_comparison',''),flush=True)
controls=[r['frozen_cache_comparison'] for r in records if r['kind']=='full_frame_control']
audit={'status':'PASS','jobs':len(records),'ROI_crop_jobs':sum(r['kind']=='ROI_crop' for r in records),'full_frame_controls':len(controls),'minimum_full_frame_frozen_pixel_agreement':min(r['best_one_to_one_pixel_agreement'] for r in controls),'all_selected_inputs_processed':len(records)==len(manifest['jobs']),'no_depth_refinement':True,'no_GT_used':True,'new_PLY_generated':False}
(R/'inference_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('COMPLETE',json.dumps(audit),flush=True)
