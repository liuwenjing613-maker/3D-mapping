from common100 import *
from itertools import islice
import shutil
def main():
 tick=time.perf_counter();TRACKROOT.mkdir(parents=True,exist_ok=True)
 cfg=read(BASEC/'prefix100/config.json')
 old_scene=Path(cfg['source']['scene_root'])
 first=(old_scene/cfg['source']['rgb_pattern'].format(frame=0)).resolve()
 original_scene=first.parent.parent
 scene=TRACKROOT/'inputs/scene';(scene/'results').mkdir(parents=True,exist_ok=True)
 hashes={};files=[]
 for fid in range(496):
  original=original_scene/cfg['source']['rgb_pattern'].format(frame=fid)
  assert original.exists()
  link=scene/cfg['source']['rgb_pattern'].format(frame=fid)
  if link.exists():assert link.resolve()==original.resolve()
  else:link.symlink_to(original.resolve())
  hashes[str(link)]=digest(link);files.append(str(original.resolve()))
 lines=(old_scene/cfg['source']['trajectory']).read_text().splitlines()
 assert len(lines)==496
 (scene/cfg['source']['trajectory']).write_text('\n'.join(lines)+'\n')
 cfg['source']['scene_root']=str(scene)
 cfg['purpose']='100_frame_A_B_readonly_bidirectional_tracking'
 dump(C/'track_config100.json',cfg)
 old=read(B/'input_guard.json')
 allowed=set(old['allowed_source_files'])|set(files)|{str((TOP/'committed50.json').resolve())}
 guard=dict(root=str(TRACKROOT),allowed_source_files=sorted(allowed),restricted_input_roots=old['restricted_input_roots'],
 restricted_result_roots=old['restricted_result_roots'],allowed_prefix_cache_roots=[str(B),str(TOP/'prefix100_new50'),str(TOP/'repair50'),str(TOP/'review')],frame_ids=FIDS,raw_cutoff=495)
 dump(TRACKROOT/'input_guard.json',guard)
 bundle=hashlib.sha256(json.dumps([(fid,hashes[str(scene/cfg['source']['rgb_pattern'].format(frame=fid))]) for fid in range(496)],separators=(',',':')).encode()).hexdigest()
 dump(TRACKROOT/'inputs_complete.json',dict(status='PASS',raw_frames=496,raw_cutoff=495,mapping_frames=100,
  raw_rgb_hashes=hashes,raw_rgb_sequence_sha256=bundle,trajectory_sha256=digest(scene/cfg['source']['trajectory']),seconds=time.perf_counter()-tick,
  directions=['forward','reverse'],source_seeds_preserved=True,repair100_allowed=False,repair50_committed_sha256=digest(TOP/'committed50.json')))
 print(json.dumps(dict(status='PASS',raw_frames=496,seconds=time.perf_counter()-tick)))
if __name__=='__main__':main()
