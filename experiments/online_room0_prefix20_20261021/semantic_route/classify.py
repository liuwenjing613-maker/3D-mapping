"""Classify every existing SAM partition before any downstream selection gates."""
from pathlib import Path
import collections,hashlib,json,time,sqlite3,os,sys,importlib.metadata
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
import open_clip
from safetensors.torch import load_file

CODE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/semantic_route')
SOURCE=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/frontend')
FILTER=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/overmerge')

def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def masked_square(rgb,mask,gray):
    yy,xx=np.nonzero(mask);assert len(xx)>0
    x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
    crop=rgb[y0:y1,x0:x1].copy();crop[~mask[y0:y1,x0:x1]]=gray
    height,width=crop.shape[:2];side=max(height,width)
    square=np.full((side,side,3),gray,np.uint8);left=(side-width)//2;top=(side-height)//2
    square[top:top+height,left:left+width]=crop
    assert np.array_equal(square[top:top+height,left:left+width][mask[y0:y1,x0:x1]],rgb[mask])
    return square,[x0,y0,x1,y1]

def main():
    start=time.time();ROOT.mkdir(parents=True,exist_ok=True);config=read(CODE/'config.json')
    assert torch.cuda.is_available(),'CUDA unavailable; no silent slow CPU fallback'
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weight_source=Path(config['weight_source']);assert weight_source.is_file()
    model_dir=Path('/data/chenkejun/CVPR/models/openclip');model_dir.mkdir(exist_ok=True)
    weights=model_dir/'ViT-H-14-laion2b_s32b_b79k.safetensors'
    if not weights.exists():weights.symlink_to(weight_source)
    assert weights.resolve()==weight_source.resolve()
    weight_sha=sha(weights);pc=open_clip.get_pretrained_cfg(config['model_name'],config['pretrained_tag'])
    model,_,preprocess=open_clip.create_model_and_transforms(config['model_name'],pretrained=None,device='cuda',precision='fp16',image_mean=pc['mean'],image_std=pc['std'])
    state=load_file(str(weights),device='cpu');model.load_state_dict(state,strict=True);del state;model.eval()
    tokenizer=open_clip.get_tokenizer(config['model_name'])
    terms=[x for family in config['targets'].values() for x in family]+config['controls']
    families=[name for name,words in config['targets'].items() for _ in words]+['other']*len(config['controls'])
    prompts=[config['prompt_template'].format(x) for x in terms]
    with torch.inference_mode():text_features=functional.normalize(model.encode_text(tokenizer(prompts).cuda()).float(),dim=-1)
    model_meta=dict(model_name=config['model_name'],pretrained_tag=config['pretrained_tag'],weights=str(weights),weights_sha256=weight_sha,open_clip_version=importlib.metadata.version('open-clip-torch'),torch_version=torch.__version__,device=torch.cuda.get_device_name(),precision='fp16 model; float32 normalization and cosine',preprocess=str(preprocess))
    cache_config=dict(model=model_meta,gray=config['gray'],masked_bbox_square=True,terms=terms,prompts=prompts)
    namespace=hashlib.sha256(json.dumps(cache_config,sort_keys=True).encode()).hexdigest()
    conn=sqlite3.connect(ROOT/'feature_cache.sqlite');conn.execute('create table if not exists features (key text primary key, vector blob not null)')
    f=read(FILTER/'web/data.json');hashes=read(FILTER/'input_hashes.json');man=read(SOURCE/'input_manifest.json');jobs={j['key']:j for j in man['jobs']}
    source_inputs={}
    for job in f['jobs']:
        sid=job['sources'][2];original=jobs[job['key']]
        if sid in source_inputs:assert source_inputs[sid]['RGB_sha256']==original['RGB_sha256']
        else:source_inputs[sid]=original
    protected={str(p):sha(p) for p in [FILTER/'web/data.json',FILTER/'input_hashes.json',SOURCE/'input_manifest.json']}
    cache={};records={};pending={};encoded=0;reused=0;verified=0
    def flush():
        nonlocal encoded
        if not pending:return
        keys=list(pending);batch=torch.stack([pending[k] for k in keys]).to(device='cuda',dtype=model.visual.conv1.weight.dtype)
        with torch.inference_mode():vectors=functional.normalize(model.encode_image(batch).float(),dim=-1).cpu().numpy()
        assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
        for key,vector in zip(keys,vectors):
            cache[key]=vector;conn.execute('insert or replace into features values (?,?)',(key,vector.astype('<f4').tobytes()))
        conn.commit();encoded+=len(keys);pending.clear()
    for si,(sid,job) in enumerate(sorted(source_inputs.items()),1):
        src=f['sources'][sid];path=Path(src['path']);rgb_path=Path(job['input_file'])
        assert sha(path)==hashes[str(path)];assert sha(rgb_path)==job['RGB_sha256'];verified+=1
        labels=np.asarray(Image.open(path),np.int32);rgb=np.asarray(Image.open(rgb_path).convert('RGB'));assert labels.shape==rgb.shape[:2]
        for lid,cid in src['mapping'].items():
            lid=int(lid);cand=f['candidates'][cid];mask=labels==lid;assert int(mask.sum())==cand['pixels']>0
            square,bbox=masked_square(rgb,mask,config['gray'])
            mask_sha=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            key=hashlib.sha256(namespace.encode()+str(square.shape).encode()+square.tobytes()).hexdigest()
            if key in cache or key in pending:reused+=1
            else:
                found=conn.execute('select vector from features where key=?',(key,)).fetchone()
                if found:cache[key]=np.frombuffer(found[0],dtype='<f4').copy();reused+=1
                else:pending[key]=preprocess(Image.fromarray(square))
            records[cand['id']]=dict(candidate_id=cand['id'],source_index=sid,label=lid,area=cand['pixels'],bbox=bbox,mask_sha256=mask_sha,RGB_sha256=job['RGB_sha256'],source_mask_sha256=hashes[str(path)],feature_cache_key=key)
            if len(pending)>=32:flush()
        if si%25==0:
            flush();print('CLASSIFY',si,'/',len(source_inputs),'candidates',len(records),'encoded',encoded,'reused',reused,flush=True)
            save(ROOT/'progress.json',dict(stage='classification',sources=si,total_sources=len(source_inputs),candidates=len(records),encoded=encoded,seconds=round(time.time()-start,2)))
    flush();conn.close();text_vectors=text_features.cpu().numpy();n_target=len(terms)-len(config['controls']);counts=collections.Counter()
    for rec in records.values():
        scores=cache[rec['feature_cache_key']]@text_vectors.T;ti=int(np.argmax(scores[:n_target]));oi=n_target+int(np.argmax(scores[n_target:]));best=int(np.argmax(scores))
        margin=float(scores[ti]-scores[oi]);hit=margin>=config['margin_threshold']
        rec.update(scores=[float(v) for v in scores],target_family=families[ti],target_term=terms[ti],target_score=float(scores[ti]),control_term=terms[oi],control_score=float(scores[oi]),margin=margin,hit=bool(hit),top_term=terms[best])
        counts['hit_'+families[ti] if hit else 'not_hit']+=1
    assert len(records)==sum(c['model']==2 for c in f['candidates'])
    assert protected=={p:sha(Path(p)) for p in protected}
    result=dict(schema=config['schema'],config=config,model=model_meta,terms=terms,prompts=prompts,cache_namespace=namespace,candidates=records,stats=dict(counts,total_candidates=len(records),unique_image_encodings=encoded,cached_or_duplicate_candidates=reused,verified_sources=verified,seconds=round(time.time()-start,2)),source_sha256=protected)
    save(ROOT/'semantic.json',result);save(ROOT/'progress.json',dict(stage='COMPLETE',stats=result['stats']));print('COMPLETE',json.dumps(result['stats']),flush=True)

if __name__=='__main__':main()
