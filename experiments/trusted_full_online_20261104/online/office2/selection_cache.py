import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/trusted_full_online_20261104')
import write_fence
"""Exact raw-model reuse by RGB bytes and normalized, frozen inference implementation."""
from common_continue import *
import ast,copy
from PIL import Image
import numpy as np
SC=CONT/'selection_code'/str(CHECKPOINT)/GROUP
MODEL_DIGEST_CACHE={}
def model_digest(path):
 # One complete verification per unchanged model/config file in this process.
 p=Path(path).resolve()
 before=p.stat();key=(str(p),before.st_size,before.st_mtime_ns,before.st_ctime_ns)
 if key in MODEL_DIGEST_CACHE:return MODEL_DIGEST_CACHE[key]
 result=digest(p)
 after=p.stat()
 assert (before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),('MODEL_FILE_CHANGED_DURING_HASH',str(p))
 MODEL_DIGEST_CACHE[key]=result
 return result

def source_pairs():
 result=[]
 for n in [50, 100, 200, 300, 400]:
  if n>CHECKPOINT:continue
  for g in ['A','B']:
   root=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104/online/office2')/str(n)/g/'candidates'
   if root!=CANDIDATE_ROOT:
    code=CONT/'selection_code'/str(n)/g
    resolved=root.resolve()
    if resolved!=root:
     # The identical first checkpoint is a read-only borrow, so its provenance names original code.
     expected=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/online')/SCENE/str(n)/g/'candidates'
     assert n==50 and resolved==expected
     code=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/online')/SCENE/'selection_code'/str(n)/g
     root=resolved
    result.append((code,root))
 # Exact frontend inference may also be read from original reached-prefix candidates.
 archive=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/online')/SCENE
 archive_code=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/online')/SCENE/'selection_code'
 seen={str(root.resolve()) for _,root in result}
 for n in [50,100,200,300,400]:
  if n>CHECKPOINT:continue
  for g in ['A','B']:
   root=archive/str(n)/g/'candidates'
   if str(root.resolve()) not in seen:
    result.append((archive_code/str(n)/g,root));seen.add(str(root.resolve()))
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
   # Only relocation roots are normalized; all inference logic/constants remain in the AST equality check.
   norm=norm.replace('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101','/home/chenkejun/CVPR/experiments/trusted_full_online_20261104').replace('/data/chenkejun/CVPR/results/unified_full_depth_20261101','/data/chenkejun/CVPR/results/trusted_full_online_20261104')
   assert ast.dump(ast.parse(norm),include_attributes=False)==ast.dump(ast.parse(target_source),include_attributes=False)
   weight=Path(p.get('weights',p.get('checkpoint')));wh=p.get('weights_sha256',p.get('checkpoint_sha256'));assert model_digest(weight)==wh
   if model=='cropformer':
    assert model_digest(p['config'])==p['config_sha256'] and digest('/home/chenkejun/CVPR/repos/CropFormer/demo_cropformer/predictor.py')==p['predictor_sha256']
    assert p['seed']==0 and p['threshold']==.5
    config_files=list(Path('/home/chenkejun/CVPR/repos/CropFormer/configs').rglob('*.yaml'))
   else:
    assert digest('/home/chenkejun/CVPR/repos/sam2_1/sam2/automatic_mask_generator.py')==p['generator_sha256']
    assert p['seed']==0 and p['apply_postprocessing']==False
    config_files=list(Path('/home/chenkejun/CVPR/repos/sam2_1/sam2/configs').rglob('*.yaml'))
   assert config_files and all(q.stat().st_mtime<=pp.stat().st_mtime for q in config_files)
   proofs.append(dict(source=str(root),source_code=str(source),source_code_sha256=digest(source),target_code_sha256=digest(SC/'frontend'/script),relocation_roots_only=True,model=model,source_provenance_sha256=digest(pp),weights_sha256=wh,normalized_inference_code_identical=True,config_tree_predates_original_model_initialization=True,config_tree_sha256={str(q):model_digest(q) for q in config_files}))
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
