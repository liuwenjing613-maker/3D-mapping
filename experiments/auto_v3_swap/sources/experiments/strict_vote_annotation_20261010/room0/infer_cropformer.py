"""Run native CropFormer and compare full-frame controls to frozen caches."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, shutil
import numpy as np, torch, cv2
from scipy.optimize import linear_sum_assignment

ROOT=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/room0')
CODE=Path('/home/chenkejun/CVPR/repos/CropFormer')
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'demo_cropformer'))
from demo_from_dirs import setup_cfg
from predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
from prepare_inputs import sha,write

def agreement(a,b):
 ai=np.unique(a);bi=np.unique(b);n=len(bi)
 joint=np.bincount(np.searchsorted(ai,a.ravel())*n+np.searchsorted(bi,b.ravel()),minlength=len(ai)*n).reshape(len(ai),n)
 union=joint.sum(1)[:,None]+joint.sum(0)[None,:]-joint
 iou=joint/np.maximum(1,union);rr,cc=linear_sum_assignment(-iou)
 return dict(pixel_agreement=float(joint[rr,cc].sum()/a.size),matched_mean_IoU=float(iou[rr,cc].mean()),identical_labels=bool(np.array_equal(a,b)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args()
 manifest=json.loads((ROOT/'input_manifest.json').read_text());weights=Path('/data/chenkejun/CVPR/models/CropFormer_hornet_3x_03823a.pth')
 config=CODE/'configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml'
 assert sha(weights)=='71915b7e9dc63fc662b32c59dff513530a06221c3b22f6de7ff4d6ef82df1128'
 cfg=setup_cfg(argparse.Namespace(config_file=str(config),opts=['MODEL.WEIGHTS',str(weights)]))
 np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
 model=VisualizationDemo(cfg)
 (ROOT/'cropformer/model_config.yaml').write_text(cfg.dump())
 write(ROOT/'cropformer/model_provenance.json',dict(weights=str(weights),weights_sha256=sha(weights),
  config=str(config),config_sha256=sha(config),predictor_sha256=sha(CODE/'demo_cropformer/predictor.py'),
  MIN_SIZE_TEST=cfg.INPUT.MIN_SIZE_TEST,MAX_SIZE_TEST=cfg.INPUT.MAX_SIZE_TEST,
  ENTITY_CROP_AREA_RATIO=cfg.ENTITY.CROP_AREA_RATIO,ENTITY_CROP_STRIDE_RATIO=cfg.ENTITY.CROP_STRIDE_RATIO,
  ENTITY_CROP_SAMPLE_NUM_TEST=cfg.ENTITY.CROP_SAMPLE_NUM_TEST,threshold=.5,seed=0,
  partition='ascending score painting; highest score owns overlap; selected model IDs retained',GPU=torch.cuda.get_device_name(),
  GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),code_sha256=sha(__file__)))
 jobs=sorted(manifest['jobs'],key=lambda j:(j['kind']!='control',j['key']))
 if args.limit:jobs=jobs[:args.limit]
 done={};rows=[];beg=time.perf_counter()
 for idx,job in enumerate(jobs,1):
  key=job['key'];png=ROOT/'cropformer'/(key+'.png');recpath=png.with_suffix('.json')
  if recpath.exists():
   rec=json.loads(recpath.read_text());assert sha(png)==rec['mask_sha256'];assert rec['RGB_sha256']==job['RGB_sha256']
  else:
   assert sha(job['input_file'])==job['RGB_sha256']
   rgb=read_image(job['input_file'],format='BGR');t=time.perf_counter();duplicate=job['same_RGB_as']
   cached_duplicate=ROOT/'cropformer'/(str(duplicate)+'.json')
   if duplicate and (duplicate in done or cached_duplicate.exists()):
    source_rec=done.get(duplicate) or json.loads(cached_duplicate.read_text())
    assert source_rec['RGB_sha256']==job['RGB_sha256'] and sha(source_rec['mask_file'])==source_rec['mask_sha256']
    shutil.copyfile(source_rec['mask_file'],png);rec=dict(source_rec);rec.pop('frozen_cache_comparison',None);rec['reuse_from']=duplicate
   else:
    instances=model.run_on_image(rgb)['instances'];selected=instances.scores>=.5
    scores=instances.scores[selected];masks=instances.pred_masks[selected]
    labels=np.zeros(rgb.shape[:2],np.uint16)
    for ix in torch.argsort(scores):labels[masks[ix].cpu().numpy().astype(bool)]=int(ix)+1
    assert cv2.imwrite(str(png),labels)
    transform=model.predictor.generate_img_augs()[0][0].get_transform(rgb);resized=transform.apply_image(rgb)
    # Save actual full-image-branch input once for every crop; internal CropFormer crops stay native.
    input_preview=ROOT/'cropformer'/(key+'_network_input.jpg');cv2.imwrite(str(input_preview),resized)
    rec=dict(scores=scores.cpu().tolist(),areas=masks.flatten(1).sum(1).cpu().tolist(),network_input_wh=[resized.shape[1],resized.shape[0]],network_input_preview=str(input_preview),reuse_from=None)
    del instances,scores,masks
   torch.cuda.synchronize()
   labels=cv2.imread(str(png),cv2.IMREAD_UNCHANGED);assert list(labels.shape)==job['shape']
   rec.update(key=key,mask_file=str(png),mask_sha256=sha(png),RGB_sha256=job['RGB_sha256'],seconds=time.perf_counter()-t,
    visible_mask_count=int(len(np.unique(labels[labels>0]))),effective_zoom=job['effective_zoom'])
   if job['kind']=='control':
    rec['frozen_cache_comparison']=agreement(cv2.imread(job['baseline_file'],cv2.IMREAD_UNCHANGED),labels)
   write(recpath,rec)
  if job['kind']=='control':
   rec['frozen_cache_comparison']=agreement(cv2.imread(job['baseline_file'],cv2.IMREAD_UNCHANGED),cv2.imread(str(png),cv2.IMREAD_UNCHANGED))
   write(recpath,rec)
  done[key]=rec;rows.append(rec)
  if idx%10==0 or job['kind']=='control':print('CROP',idx,len(jobs),key,round(time.perf_counter()-beg,1),rec.get('frozen_cache_comparison',''),flush=True)
  write(ROOT/'cropformer/progress.json',dict(status='RUNNING',done=idx,total=len(jobs),last=key,seconds=time.perf_counter()-beg))
 controls=[r['frozen_cache_comparison'] for r in rows if 'frozen_cache_comparison' in r]
 audit=dict(status='PASS' if len(rows)==len(manifest['jobs']) else 'SMOKE',processed=len(rows),total=len(manifest['jobs']),
  control_min_pixel_agreement=min(c['pixel_agreement'] for c in controls),controls=len(controls),RGB_coordinates_preserved=True,seconds=time.perf_counter()-beg)
 assert audit['control_min_pixel_agreement']>=.999,'Frozen full-frame controls are incompatible'
 write(ROOT/'cropformer/audit.json',audit);write(ROOT/'cropformer/progress.json',audit)
 print('CROP_COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':main()
