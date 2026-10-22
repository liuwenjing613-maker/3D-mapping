"""Exact raw-model reuse by RGB bytes and normalized, frozen inference implementation."""
from common_continue import *
import ast,copy
from PIL import Image
import numpy as np
SC=CONT/'selection_code'/str(CHECKPOINT)/GROUP
def source_pairs():
 result=[(BASEC/'prefix100',TOP/'prefix100'),(BASEC/'prefix100_new50',TOP/'prefix100_new50')]
 for n in [200,300]:
  if n>CHECKPOINT:continue
  for g in ['A','B']:
   root=TOP/'continuation300'/str(n)/g/'candidates'
   if root!=CANDIDATE_ROOT:result.append((CONT/'selection_code'/str(n)/g,root))
 return result
def link(src,dst,h):
 src=Path(src);dst=Path(dst);assert digest(src)==h;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(src.resolve())
 assert digest(dst)==h
def main():
 started=time.perf_counter();manifest=read(CANDIDATE_ROOT/'frontend/input_manifest.json');rows=[];proofs=[];counts={}
 for model,script in [('cropformer','infer_cropformer.py'),('sam2_1','infer_sam.py')]:
  target_source=(SC/'frontend'/script).read_text();index={}
  for code,root in source_pairs():
   pp=root/'frontend'/model/'model_provenance.json';mp=root/'frontend/input_manifest.json'
   if not pp.exists() or not mp.exists():continue
   p=read(pp);source=code/'frontend'/script
   assert digest(source)==p['code_sha256']
   norm=source.read_text().replace(str(code),str(SC)).replace(str(root),str(CANDIDATE_ROOT))
   assert ast.dump(ast.parse(norm),include_attributes=False)==ast.dump(ast.parse(target_source),include_attributes=False)
   weight=Path(p.get('weights',p.get('checkpoint')));wh=p.get('weights_sha256',p.get('checkpoint_sha256'));assert digest(weight)==wh
   if model=='cropformer':
    assert digest(p['config'])==p['config_sha256'] and digest('/home/chenkejun/CVPR/repos/CropFormer/demo_cropformer/predictor.py')==p['predictor_sha256']
    assert p['seed']==0 and p['threshold']==.5
    config_files=list(Path('/home/chenkejun/CVPR/repos/CropFormer/configs').rglob('*.yaml'))
   else:
    assert digest('/home/chenkejun/CVPR/repos/sam2_1/sam2/automatic_mask_generator.py')==p['generator_sha256']
    assert p['seed']==0 and p['apply_postprocessing']==False
    config_files=list(Path('/home/chenkejun/CVPR/repos/sam2_1/sam2/configs').rglob('*.yaml'))
   assert config_files and all(q.stat().st_mtime<=pp.stat().st_mtime for q in config_files)
   proofs.append(dict(source=str(root),model=model,source_provenance_sha256=digest(pp),weights_sha256=wh,normalized_inference_code_identical=True,config_tree_predates_original_model_initialization=True,config_tree_sha256={str(q):digest(q) for q in config_files}))
   for j in read(mp)['jobs']:
    if j['frame']>CURRENT_RAW:continue
    record=root/'frontend'/model/(j['key']+'.json')
    if record.exists():index.setdefault((j['RGB_sha256'],tuple(j['shape'])),(j,record))
  hits=0
  for j in manifest['jobs']:
   match=index.get((j['RGB_sha256'],tuple(j['shape'])))
   if not match:continue
   old,record=match;rec=read(record)
   assert digest(j['input_file'])==digest(old['input_file'])==j['RGB_sha256']==rec['RGB_sha256']
   assert np.array_equal(np.asarray(Image.open(j['input_file'])),np.asarray(Image.open(old['input_file'])))
   target=CANDIDATE_ROOT/'frontend'/model/(j['key']+'.png');link(rec['mask_file'],target,rec['mask_sha256'])
   nr=copy.deepcopy(rec);nr.update(key=j['key'],mask_file=str(target),RGB_sha256=j['RGB_sha256'],seconds=0.,
    reuse_from='exact-prefix:'+old['key'],cache_source_record=str(record),cache_source_record_sha256=digest(record),
    cache_proof='identical RGB PNG sha/pixels, model weights/config/parameters, normalized inference code')
   if model=='sam2_1':
    dest=target.with_suffix('.npz');link(rec['proposal_file'],dest,rec['proposals_sha256']);nr['proposal_file']=str(dest)
   else:
    nr['effective_zoom']=j['effective_zoom'];nr.pop('frozen_cache_comparison',None)
    if rec.get('network_input_preview'):
     src=Path(rec['network_input_preview']);dest=target.with_name(j['key']+'_network_input.jpg');link(src,dest,digest(src));nr['network_input_preview']=str(dest)
   dump(target.with_suffix('.json'),nr);hits+=1
   rows.append(dict(model=model,target_key=j['key'],source_record=str(record),frame=j['frame'],RGB_sha256=j['RGB_sha256'],mask_sha256=rec['mask_sha256']))
  counts[model]=dict(total_jobs=len(manifest['jobs']),cache_hits=hits,inference_jobs=len(manifest['jobs'])-hits)
 dump(CANDIDATE_ROOT/'exact_model_cache_audit.json',dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,counts=counts,proofs=proofs,rows=rows,seconds=time.perf_counter()-started))
 print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
