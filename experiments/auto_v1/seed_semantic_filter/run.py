"""Infer whole-frame semantics once per RGB frame and screen unchanged final seeds."""
from pathlib import Path
import argparse
import collections
import hashlib
import importlib.metadata
import json
import time
import numpy as np
from PIL import Image
from rules import resolve_labels, summarize_mask, validate_records, filtered_manifest


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def palette_for(labels):
    # Stable display colors, not ADE annotation IDs; red/orange/blue/purple identify excluded classes.
    import colorsys
    colors=np.array([[int(255*v) for v in colorsys.hsv_to_rgb((i*.61803398875)%1,.60,.86)] for i in range(len(labels))],dtype=np.uint8)
    custom={'wall':[233,79,92],'floor':[235,158,48],'ceiling':[78,139,236],'rug':[181,91,215]}
    for i,name in labels.items():
        if name in custom: colors[i]=custom[name]
    return colors


def preview_frame(rgb,pred,colors,out,fid):
    for name,arr in [('rgb',rgb),('semantic',colors[pred])]:
        im=Image.fromarray(arr); im.thumbnail((960,544))
        im.save(out/'previews'/f'f{fid:06d}_{name}.jpg',quality=90)


def preview_seed(rgb,mask,pred,conf,colors,excluded_ids,cfg,out,gid):
    ys,xs=np.nonzero(mask); x0,x1=int(xs.min()),int(xs.max()+1); y0,y1=int(ys.min()),int(ys.max()+1)
    pad=max(20,round(.08*max(x1-x0,y1-y0)))
    box=[max(0,x0-pad),max(0,y0-pad),min(rgb.shape[1],x1+pad),min(rgb.shape[0],y1+pad)]
    context=rgb.copy(); context[mask]=(rgb[mask]*.6+np.array([28,223,184])*.4).astype(np.uint8)
    masked=np.full_like(rgb,38); masked[mask]=rgb[mask]
    semantic=(rgb*.25).astype(np.uint8); semantic[mask]=colors[pred[mask]]
    confidence=(rgb*.25).astype(np.uint8)
    high=conf >= cfg['pixel_confidence']
    excluded=np.isin(pred,excluded_ids)
    confidence[mask & high & excluded]=[240,75,80]
    confidence[mask & high & ~excluded]=[27,188,158]
    confidence[mask & ~high]=[246,195,71]
    images={}
    for name,arr in [('context',context),('masked',masked),('semantic',semantic),('confidence',confidence)]:
        im=Image.fromarray(arr).crop(box); im.thumbnail((500,310))
        rel=f'previews/{gid}_{name}.jpg'; im.save(out/rel,quality=92); images[name]=rel
    return images,box


def main(args):
    started=time.perf_counter()
    import torch
    import torch.nn.functional as F
    from transformers import SegformerConfig, SegformerImageProcessor, SegformerForSemanticSegmentation
    cfg_path=Path(args.config).resolve(); cfg=json.loads(cfg_path.read_text())
    manifest_path=Path(args.manifest).resolve(); source=json.loads(manifest_path.read_text()); root=manifest_path.parent
    raw_path=Path(args.raw_config).resolve(); raw=json.loads(raw_path.read_text())['source']
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'classification.json').exists(): raise FileExistsError('Frozen result exists; use a new result directory')
    for name in ['previews','semantic_cache']: (out/name).mkdir(exist_ok=True)
    model_root=Path(cfg['model_directory']); weights=model_root/'pytorch_model.bin'
    assert sha(weights)==cfg['weight_sha256'],'Weight SHA differs from official NVIDIA file'
    downloads=json.loads((model_root/'download_manifest.json').read_text())
    assert downloads['revision']==cfg['model_revision'] and downloads['model_id']==cfg['model_id']
    for name,entry in downloads['files'].items(): assert sha(model_root/name)==entry['sha256']
    config=SegformerConfig.from_pretrained(str(model_root),local_files_only=True)
    labels,excluded_ids=resolve_labels(config.id2label,cfg); assert len(labels)==150
    code_hashes={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    protected={str(p):sha(p) for p in [manifest_path,raw_path,cfg_path,weights,model_root/'config.json',model_root/'preprocessor_config.json',model_root/'download_manifest.json']}
    freeze={'status':'FROZEN_BEFORE_INFERENCE','config':cfg,'source_hashes':protected,'code_hashes':code_hashes,'GT_used':False,'human_selection_used':False}
    save(out/'input_freeze.json',freeze)
    assert torch.cuda.is_available(),'Explicit GPU run required'
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    processor=SegformerImageProcessor.from_pretrained(str(model_root),local_files_only=True)
    model=SegformerForSemanticSegmentation(config)
    state=torch.load(str(weights),map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True); del state
    model=model.float().eval().cuda()
    model_meta={'model_id':cfg['model_id'],'revision':cfg['model_revision'],'weights_sha256':cfg['weight_sha256'],
                'torch_version':torch.__version__,'transformers_version':importlib.metadata.version('transformers'),
                'device':torch.cuda.get_device_name(),'precision':cfg['precision'],'processor':processor.to_dict(),
                'excluded_id_to_label':{i:labels[i] for i in excluded_ids},'id2label':labels}
    basis={'model':model_meta,'postprocess':cfg['postprocess'],'preprocessor_file_sha256':protected[str(model_root/'preprocessor_config.json')]}
    namespace=hashlib.sha256(json.dumps(basis,sort_keys=True).encode()).hexdigest()
    setup_seconds=time.perf_counter()-started
    groups_by_frame=collections.defaultdict(list)
    for original_id,g in enumerate(source['groups'],1): groups_by_frame[g['primary']['frame']].append((original_id,g))
    records=[]; frame_records=[]; inputs={}; infer_seconds=0; newly_inferred=0; cache_hits=0; checked_argmax=False
    colors=palette_for(labels)
    for fi,(fid,groups) in enumerate(sorted(groups_by_frame.items()),1):
        rgb_path=Path(raw['scene_root'])/raw['rgb_pattern'].format(frame=fid)
        rgb_sha=sha(rgb_path); inputs[str(rgb_path)]=rgb_sha
        rgb=np.array(Image.open(rgb_path).convert('RGB')); h,w=rgb.shape[:2]
        key=hashlib.sha256((namespace+rgb_sha+str((h,w))).encode()).hexdigest()
        npz_path=out/'semantic_cache'/f'{key}.npz'; meta_path=npz_path.with_suffix('.json')
        if npz_path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text()); assert meta['namespace']==namespace and meta['RGB_sha256']==rgb_sha
            assert sha(npz_path)==meta['npz_sha256']
            with np.load(npz_path,allow_pickle=False) as data: pred=data['labels']; conf=data['confidence']
            cache_hits+=1; elapsed=0
        else:
            tensors=processor(images=Image.fromarray(rgb),return_tensors='pt')
            assert tuple(tensors['pixel_values'].shape)==(1,3,640,640)
            torch.cuda.synchronize(); t=time.perf_counter()
            with torch.inference_mode():
                logits=model(pixel_values=tensors['pixel_values'].cuda()).logits.float()
                up=F.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
                maxprob,ids=up.softmax(dim=1).max(dim=1)
                if not checked_argmax:
                    official=processor.post_process_semantic_segmentation(type('O',(),{'logits':logits})(),target_sizes=[(h,w)])[0]
                    assert torch.equal(ids[0],official),'Postprocessing differs from official processor'
                    checked_argmax=True
                pred=ids[0].cpu().numpy().astype(np.uint8); conf=maxprob[0].cpu().numpy().astype(np.float32)
                del logits,up,maxprob,ids
            torch.cuda.synchronize(); elapsed=time.perf_counter()-t; infer_seconds+=elapsed; newly_inferred+=1
            np.savez_compressed(npz_path,labels=pred,confidence=conf)
            save(meta_path,{'namespace':namespace,'RGB_sha256':rgb_sha,'shape':[h,w],'npz_sha256':sha(npz_path)})
        assert pred.shape==conf.shape==(h,w)
        Image.fromarray(pred).save(out/'semantic_cache'/f'f{fid:06d}_label_ids.png')
        preview_frame(rgb,pred,colors,out,fid)
        frame_records.append({'frame':fid,'RGB_path':str(rgb_path),'RGB_sha256':rgb_sha,'shape':[h,w],
                              'cache':str(npz_path.relative_to(out)),'cache_sha256':sha(npz_path),'inference_seconds':elapsed,
                              'groups':[g['id'] for _,g in groups],'rgb_preview':f'previews/f{fid:06d}_rgb.jpg',
                              'semantic_preview':f'previews/f{fid:06d}_semantic.jpg'})
        for original_id,g in groups:
            p=g['primary']; mask_path=root/p['mask']; assert mask_path.resolve().is_relative_to(root.resolve())
            mask_sha=sha(mask_path); assert mask_sha==p['mask_sha256']; inputs[str(mask_path)]=mask_sha
            mask=np.asarray(Image.open(mask_path))>0; assert int(mask.sum())==p['area'] and mask.shape==(h,w)
            info=summarize_mask(pred,conf,mask,labels,cfg)
            images,box=preview_seed(rgb,mask,pred,conf,colors,excluded_ids,cfg,out,g['id'])
            records.append({'group_id':g['id'],'original_track_id':original_id,'mask_id':p['id'],'frame':fid,
                            'model':p['representative']['model'],'mask_sha256':mask_sha,'mask_relative_path':p['mask'],
                            'RGB_sha256':rgb_sha,'images':images,'preview_crop_box':box,'frame_cache':str(npz_path.relative_to(out)),**info})
        print('FRAME',fi,'/',len(groups_by_frame),'frame',fid,'seeds',len(groups),'GPU_seconds',round(elapsed,3),flush=True)
    records.sort(key=lambda r:r['original_track_id']); validate_records(source,records)
    for path,expected in {**protected,**code_hashes,**inputs}.items(): assert sha(path)==expected,'Source changed: '+path
    stats={'total':len(records),'unique_frames':len(frame_records),'newly_inferred_frames':newly_inferred,'cached_frames':cache_hits,
           'suggest_exclude':sum(r['exclude'] for r in records),'retain':sum(r['decision']=='retain' for r in records),
           'review':sum(r['decision']=='review' for r in records),'operationally_kept':sum(not r['exclude'] for r in records),
           'sources':dict(collections.Counter(str(r['model']) for r in records)),
           'setup_seconds':setup_seconds,'inference_seconds':infer_seconds,'total_seconds':time.perf_counter()-started,
           'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'official_postprocess_agreement_checked':checked_argmax}
    result={'schema':cfg['schema'],'status':'PASS','config':cfg,'model':model_meta,'palette':colors.tolist(),
            'records':records,'frames':frame_records,'stats':stats,'source_manifest':str(manifest_path),
            'source_manifest_sha256':sha(manifest_path),'GT_used':False,'human_selection_used':False,
            'input_hashes':{**protected,**inputs},'code_hashes':code_hashes,'input_freeze_sha256':sha(out/'input_freeze.json')}
    save(out/'classification.json',result)
    filtered=filtered_manifest(source,records,sha(manifest_path),sha(out/'classification.json'),cfg)
    link=out/'masks'
    if link.exists(): assert link.is_symlink() and link.resolve()==(root/'masks').resolve()
    else: link.symlink_to(root/'masks',target_is_directory=True)
    save(out/'seed_manifest.json',filtered)
    save(out/'seed_id_map.json',{'source_manifest_sha256':sha(manifest_path),'objects':[
        {k:r[k] for k in ['original_track_id','group_id','mask_id','frame','mask_sha256']} for r in records if not r['exclude']]})
    save(out/'filter_complete.json',{'status':'PASS','stats':stats,'classification_sha256':sha(out/'classification.json'),
                                   'filtered_manifest_sha256':sha(out/'seed_manifest.json'),'all_primary_pixels_unchanged':True,
                                   'original_track_ids_preserved':True,'GT_used':False,'human_selection_used':False})
    print('COMPLETE',json.dumps(stats),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--raw-config',required=True)
    p.add_argument('--out',required=True); p.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    main(p.parse_args())
