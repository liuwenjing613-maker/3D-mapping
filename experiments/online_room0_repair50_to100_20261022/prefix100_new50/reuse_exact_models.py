"""Reuse A model bytes only for exact RGB pixels and unchanged model definition."""
from online_common import *
import os,time,copy,ast
from PIL import Image
import numpy as np
A=C.parent/'prefix100';AR=R.parent/'prefix100'
def link(src,dst,h):
 src=Path(src);dst=Path(dst);assert sha(src)==h
 if not dst.exists():dst.symlink_to(src.resolve())
 assert sha(dst)==h
def main():
 start=time.perf_counter();a=read(AR/'frontend/input_manifest.json');b=read(R/'frontend/input_manifest.json')
 assert all(j['frame'] in range(250,496,5) for j in b['jobs'])
 proofs={};rows=[];matched={}
 for model,script in [('cropformer','infer_cropformer.py'),('sam2_1','infer_sam.py')]:
  p=read(AR/'frontend'/model/'model_provenance.json');assert sha(A/'frontend'/script)==p['code_sha256']
  sa=(A/'frontend'/script).read_text();sb=(C/'frontend'/script).read_text().replace(str(C),str(A)).replace(str(R),str(AR))
  assert ast.dump(ast.parse(sa),include_attributes=False)==ast.dump(ast.parse(sb),include_attributes=False)
  weight=Path(p.get('weights',p.get('checkpoint')));wh=p.get('weights_sha256',p.get('checkpoint_sha256'));assert sha(weight)==wh
  frozen_at=(AR/'frontend'/model/'model_provenance.json').stat().st_mtime
  if model=='cropformer':
   cfg=Path(p['config']);assert sha(cfg)==p['config_sha256']
   pred=Path('/home/chenkejun/CVPR/repos/CropFormer/demo_cropformer/predictor.py');assert sha(pred)==p['predictor_sha256']
   config_files=list((Path('/home/chenkejun/CVPR/repos/CropFormer')/'configs').rglob('*.yaml'))
   assert p['seed']==0 and p['threshold']==.5
  else:
   repo=Path('/home/chenkejun/CVPR/repos/sam2_1')
   cfg=repo/'sam2'/p['config'];assert cfg.is_file()
   pred=repo/'sam2/automatic_mask_generator.py';assert sha(pred)==p['generator_sha256']
   config_files=list((repo/'sam2/configs').rglob('*.yaml'))
   assert p['seed']==0 and p['apply_postprocessing']==False and p['precision']=='bfloat16 autocast; TF32 allowed'
   params=next(x.value for x in ast.parse(sa).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in x.targets))
   got=eval(compile(ast.Expression(params),'<static-params>','eval'),{'dict':dict});assert got==p['parameters']
  # Dependencies have not been modified since A initialized the predictor.
  assert all(q.stat().st_mtime<=frozen_at for q in config_files)
  proofs[model]=dict(source_provenance_sha256=sha(AR/'frontend'/model/'model_provenance.json'),model_code_equal_after_output_path_normalization=True,weights_sha256=wh,config_sha256=sha(cfg),config_tree_sha256={str(q):sha(q) for q in config_files},config_tree_predates_A_model_initialization=True,predictor_sha256=sha(pred),source_GPU=p['GPU'])
  index={}
  for job in a['jobs']:index.setdefault((job['RGB_sha256'],tuple(job['shape'])),job)
  hits=0
  for job in b['jobs']:
   key=(job['RGB_sha256'],tuple(job['shape']));old=index.get(key)
   if old is None:continue
   rp=AR/'frontend'/model/(old['key']+'.json');rec=read(rp)
   assert sha(job['input_file'])==sha(old['input_file'])==job['RGB_sha256']==rec['RGB_sha256']
   assert np.array_equal(np.asarray(Image.open(job['input_file'])),np.asarray(Image.open(old['input_file'])))
   out=R/'frontend'/model;out.mkdir(parents=True,exist_ok=True);target=out/(job['key']+'.png')
   link(rec['mask_file'],target,rec['mask_sha256']);nr=copy.deepcopy(rec)
   nr.update(key=job['key'],mask_file=str(target),RGB_sha256=job['RGB_sha256'],seconds=0.,reuse_from='exact-A:'+old['key'],cache_source_record=str(rp),cache_source_record_sha256=sha(rp),cache_proof='identical RGB PNG sha/pixels + same model weights/config/parameters and normalized source code')
   if model=='sam2_1':
    dst=target.with_suffix('.npz');link(rec['proposal_file'],dst,rec['proposals_sha256']);nr['proposal_file']=str(dst)
   else:
    nr['effective_zoom']=job['effective_zoom'];nr.pop('frozen_cache_comparison',None)
    if rec.get('network_input_preview'):
     prev=Path(rec['network_input_preview']);dest=out/(job['key']+'_network_input.jpg');link(prev,dest,sha(prev));nr['network_input_preview']=str(dest)
   write(target.with_suffix('.json'),nr);rows.append(dict(model=model,target_key=job['key'],source_key=old['key'],frame=job['frame'],RGB_sha256=job['RGB_sha256'],mask_sha256=rec['mask_sha256']));hits+=1
  matched[model]=dict(total_jobs=len(b['jobs']),cache_hits=hits,inference_jobs=len(b['jobs'])-hits)
 result=dict(status='PASS',arm='B',source=str(AR),proofs=proofs,counts=matched,rows=rows,seconds=time.perf_counter()-start,matching='Exact input pixel and model hashes only; no approximate matching')
 write(R/'exact_model_cache_audit.json',result);print(json.dumps(matched),flush=True)
if __name__=='__main__':main()
