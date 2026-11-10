"""Owned mapping helpers for causal continuation to 200 and 300 frames."""
from pathlib import Path
import os, json, hashlib, time
BASEC=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
C=BASEC/'continuation300'
TARGET=int(os.environ.get('CVPR_CHECKPOINT',os.environ.get('CVPR_TARGET_COUNT','0')))
GROUP=os.environ.get('CVPR_TRACK_GROUP')
assert TARGET in (200,300) and GROUP in ('A','B')
ROOT=TOP/'continuation300'/str(TARGET)/GROUP
R=ROOT/'mapping'
FIDS=tuple(range(0,TARGET*5,5))
PRIOR_COUNT=TARGET-100
PRIOR_RAW=(PRIOR_COUNT-1)*5
CUTOFF=FIDS[-1]
SNAPSHOT=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def atomic_text(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);tmp.replace(path)
def prior_paths():
 if TARGET==200:
  return TOP/'repair100'/GROUP/'commit_complete.json',TOP/'prefix100'
 return TOP/'continuation300'/'200'/GROUP/'repair/commit_complete.json',TOP/'continuation300'/'200'/GROUP/'mapping'
def setup():
 pointer,previous=prior_paths();prior=read(pointer)
 assert prior['status']=='PASS' and prior['group']==GROUP
 assert prior['processed_frames']==PRIOR_COUNT and prior['last_raw_frame']==PRIOR_RAW
 for key in ['checkpoint','source_pixel_support','journal_manifest','geometry','final_map']:
  assert sha(prior[key])==prior[key+'_sha256'],key
 lineage=Path(prior.get('observation_lineage',str(Path(prior['checkpoint']).parent/'observation_lineage.json')))
 assert read(lineage)['status']=='PASS'
 if 'observation_lineage_sha256' in prior:assert sha(lineage)==prior['observation_lineage_sha256']
 copied=dict(prior,prior_commit_path=str(pointer),prior_commit_sha256=sha(pointer),
  prior_mapping_root=str(previous),prior_processed_frames=PRIOR_COUNT,prior_raw_cutoff=PRIOR_RAW,
  observation_lineage=str(lineage),observation_lineage_sha256=sha(lineage))
 R.mkdir(parents=True,exist_ok=True)
 existing=R/'prior_commit.json'
 if existing.exists():assert read(existing)==copied
 else:write(existing,copied)
 config=read(BASEC/'prefix100/config.json')
 original=read(Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs/room0.json'))
 scene=Path(original['source']['scene_root']).resolve()
 masks=Path(original['source']['mask_root']).resolve()
 frozen=Path('/data/chenkejun/CVPR/datasets/cropformer_frozen/room0').resolve()
 config['frame_selection']['stop_exclusive']=PRIOR_RAW+1
 config['source'].update(scene_root=str(R/'inputs/scene'),mask_root=str(R/'inputs/masks'))
 config['purpose']=f'causal_room0_{GROUP}_continuation_{PRIOR_COUNT}_to_{TARGET}'
 if not (R/'mapping_complete.json').exists():write(R/'config.json',config)
 files=[]
 for fid in FIDS:
  for path in [scene/f'results/frame{fid:06d}.jpg',scene/f'results/depth{fid:06d}.png',masks/f'frame{fid:06d}.png',frozen/f'frame{fid:06d}.png']:
   files.append(str(path.resolve(strict=True)))
 trajectory=scene/original['source']['trajectory']
 files.extend([str(trajectory),str(frozen/'frames.jsonl')])
 cache=[str(previous),str(Path(prior['checkpoint']).parent.parent),str(TOP/'prefix100')]
 guard=dict(root=str(ROOT),allowed_source_files=sorted(set(files)),
  allowed_prefix_cache_roots=sorted(set(cache)),restricted_input_roots=sorted(set(map(str,[scene,masks,frozen]))),
  restricted_result_roots=['/data/chenkejun/CVPR/results','/data/chenkejun/CVPR/revisable_instance_map','/data/chenkejun/CVPR/evaluation_results'],
  frame_ids=list(FIDS),initial_cutoff=PRIOR_RAW,raw_cutoff=CUTOFF)
 write(R/'input_guard.json',guard)
 write(R/'mapping_plan.json',dict(status='READY',group=GROUP,target_processed_frames=TARGET,
  prior_processed_frames=PRIOR_COUNT,raw_cutoff=CUTOFF,prior_raw_cutoff=PRIOR_RAW,
  source_scene=str(scene),source_masks=str(masks),frozen_masks=str(frozen),source_trajectory=str(trajectory),
  original_config='/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs/room0.json',
  guard=str(R/'input_guard.json'),config=str(R/'config.json')))
 return copied
if __name__=='__main__':
 p=setup();print(json.dumps(dict(status='READY',group=GROUP,target=TARGET,prior_commit=p['prior_commit_path'],mapping=str(R))))
