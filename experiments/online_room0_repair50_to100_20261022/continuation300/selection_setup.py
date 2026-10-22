"""Prepare current-prefix read-only sources and dense RGB for candidate/tracking stages."""
from common_continue import *
import shutil,copy
def link(src,dst):
 src=Path(src);dst=Path(dst);assert src.exists(),str(src)
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.resolve()==src.resolve(),str(dst)
 else:dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
def main():
 tick=time.perf_counter()
 complete=read(B/'mapping_complete.json')
 assert complete['status']=='PASS' and complete['frame_ids']==FIDS
 assert read(B/'config.json')['frame_selection']['stop_exclusive']==RAW_COUNT
 CANDIDATE_ROOT.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);C.mkdir(parents=True,exist_ok=True)
 sc=CONT/'selection_code'/str(CHECKPOINT)/GROUP
 for rel in ['baseline','geometry','association','inputs','mapping_complete.json']:
  link(B/rel,CANDIDATE_ROOT/rel)
 link(CONFIG,sc/'config.json')
 sourcehash={str(p):digest(p) for p in [B/'baseline/final/instance_surface.npz',B/'baseline/native/instance_surface.npz',B/'baseline/surface_evidence.npz',B/'baseline/surface_instance_frame_votes.npz',B/'geometry/surface.ply',B/'geometry/review_surface_graph_n02_p08.npz',CONFIG,B/'prior_commit.json',B/'association/observation_support_3cm.npz']}
 dump(CANDIDATE_ROOT/'mapping_input_freeze.json',dict(status='FROZEN',group=GROUP,checkpoint=CHECKPOINT,source_hashes=sourcehash,previous_checkpoint=OLD_COUNT,map_is_this_branch=True))
 cfg=read(CONFIG);old_scene=Path(cfg['source']['scene_root']);original=(old_scene/cfg['source']['rgb_pattern'].format(frame=0)).resolve().parent.parent
 scene=R/'inputs/scene';(scene/'results').mkdir(parents=True,exist_ok=True)
 hashes={};files=[]
 for fid in range(RAW_COUNT):
  src=original/cfg['source']['rgb_pattern'].format(frame=fid);dst=scene/cfg['source']['rgb_pattern'].format(frame=fid)
  link(src,dst);hashes[str(dst)]=digest(dst);files.append(str(src.resolve()))
 lines=(old_scene/cfg['source']['trajectory']).read_text().splitlines();assert len(lines)==RAW_COUNT
 (scene/cfg['source']['trajectory']).write_text('\n'.join(lines)+'\n')
 cfg['source']['scene_root']=str(scene);cfg['purpose']=f'checkpoint{CHECKPOINT}_branch{GROUP}_causal_bidirectional_tracking'
 dump(R/'track_config.json',cfg)
 old=read(B/'input_guard.json')
 caches=[str(ROOT),str(TOP/'repair50'),str(TOP/'repair100'/GROUP),str(TOP/'prefix100'),str(TOP/'prefix100_new50'),str(TOP/'tracking100')]
 # Reuse only complete past/current raw-prediction caches. No future checkpoint data.
 for n in [200,300]:
  if n>CHECKPOINT:continue
  for g in ['A','B']:
   peer=TOP/'continuation300'/str(n)/g
   if n<CHECKPOINT and g==GROUP:caches.extend(str(peer/k) for k in ['mapping','repair'])
   if peer!=ROOT:caches.extend(str(peer/k) for k in ['candidates','tracking'])
 allowed=set(old['allowed_source_files'])|set(files)
 for key in ['checkpoint','identity_counts','journal_manifest','source_pixel_support','final_map','geometry','audit']:
  val=read(B/'prior_commit.json').get(key)
  if val:allowed.add(str(Path(val).resolve()))
 readonly=[x for x in caches if x!=str(ROOT)]+[str(B)]
 caches.append(str(TOP/'continuation300/gpu_locks'))
 guard=dict(root=str(ROOT),allowed_source_files=sorted(allowed),restricted_input_roots=old['restricted_input_roots'],
  restricted_result_roots=old['restricted_result_roots'],allowed_prefix_cache_roots=sorted(set(caches)),
  read_only_result_roots=sorted(set(readonly)),frame_ids=FIDS,raw_cutoff=CURRENT_RAW)
 dump(ROOT/'input_guard.json',guard)
 oldsam=read(TOP/'tracking100/inputs_complete.json');samroot=Path('/home/chenkejun/CVPR/repos/sam2_1/sam2')
 for rel,h in oldsam['sam_source_files'].items():assert digest(samroot/rel)==h
 bundle=hashlib.sha256(json.dumps([(fid,hashes[str(scene/cfg['source']['rgb_pattern'].format(frame=fid))]) for fid in range(RAW_COUNT)],separators=(',',':')).encode()).hexdigest()
 initial=read(ROOT/'inputs_complete.json') if (ROOT/'inputs_complete.json').exists() else None
 if initial and not (ROOT/'mapping_inputs_complete.json').exists():dump(ROOT/'mapping_inputs_complete.json',initial)
 dump(ROOT/'inputs_complete.json',dict(status='PASS',group=GROUP,checkpoint=CHECKPOINT,raw_frames=RAW_COUNT,raw_cutoff=CURRENT_RAW,mapping_frames=CHECKPOINT,
  raw_rgb_hashes=hashes,raw_rgb_sequence_sha256=bundle,trajectory_sha256=digest(scene/cfg['source']['trajectory']),
  sam_source_files=oldsam['sam_source_files'],sam_source_sha256=oldsam['sam_source_sha256'],
  seconds=time.perf_counter()-tick,directions=['forward','reverse'],source_seeds_preserved=True,previous_commit_sha256=digest(B/'prior_commit.json'),
  candidate_raw_frame_range=[0 if GROUP=='A' else OLD_COUNT*5,CURRENT_RAW],GT_used=False))
 dump(CANDIDATE_ROOT/'setup_complete.json',dict(status='PASS',seconds=time.perf_counter()-tick,source_hashes=sourcehash))
 print(json.dumps(dict(status='PASS',stage='setup',checkpoint=CHECKPOINT,group=GROUP,raw_frames=RAW_COUNT,seconds=time.perf_counter()-tick)),flush=True)
if __name__=='__main__':main()
