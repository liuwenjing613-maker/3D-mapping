from common_continue import *
import sitecustomize
def main():
 start=time.perf_counter()
 assert sitecustomize.settings_path==str(ROOT/'input_guard.json') and sitecustomize.current_cutoff==CURRENT_RAW
 src=read(ROOT/'inputs_complete.json');assert len(src['raw_rgb_hashes'])==RAW_COUNT
 # Check first and last allowed raw frames; setup already hashed every dense source.
 paths=list(src['raw_rgb_hashes'])
 for p in [paths[0],paths[-1]]:assert digest(p)==src['raw_rgb_hashes'][p]
 model=Path('/home/chenkejun/CVPR/repos/sam2_1/sam2')
 for rel,h in src['sam_source_files'].items():assert digest(model/rel)==h
 raw=Path(paths[0]).resolve().parent
 probes=[]
 os.environ['CVPR_GUARD_TEST']='1'
 for path in [raw/f'frame{CURRENT_RAW+5:06d}.jpg',TOP/'full400_forbidden_probe/surface.ply',raw.parent/'info_semantic.json']:
  try:
   with path.open('rb') as f:f.read(1)
  except PermissionError as e:probes.append(dict(path=str(path),blocked=True,reason=str(e)))
  else:raise AssertionError(('guard_failed',str(path)))
 safe=ROOT/'guard_probe'/'f900001abcdef_cache.json'
 safe.parent.mkdir(exist_ok=True);safe.write_text('{"probe":true}')
 assert read(safe)=={'probe':True}
 safe.unlink()
 probes.append(dict(path=str(safe),blocked=False,reason='hex-like cache basename is not a frame'))
 os.environ.pop('CVPR_GUARD_TEST',None)
 dump(ROOT/'input_fence_audit.json',dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,raw_cutoff=CURRENT_RAW,
  dense_rgb_hash_count=RAW_COUNT,first_last_hashes_pass=True,SAM_source_files_checked=len(src['sam_source_files']),probes=probes,
  prefix_guard_enabled=True,seconds=time.perf_counter()-start,GT_used=False))
 print('PREFIX_FENCE_PASS',CHECKPOINT,GROUP,flush=True)
if __name__=='__main__':main()
