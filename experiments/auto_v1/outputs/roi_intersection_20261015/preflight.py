from pathlib import Path
import json
def scan(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():
   if 'voxel' in k.lower():print(path+'.'+k,repr(v))
   elif isinstance(v,(dict,list)):scan(v,path+'.'+k)
 elif isinstance(x,list):
  for i,v in enumerate(x):scan(v,path+f'[{i}]')
for p in ['/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json','/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/config.json','/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/catalog/room2/candidate_manifest.json']:
 if Path(p).exists():print('SOURCE',p);scan(json.loads(Path(p).read_text()))
r=Path('/data/chenkejun/CVPR/results/e1_roi_top20_20261015/source/room2');m=json.loads((r/'manifest.json').read_text());print('MEMBERSHIP',m['fixed_membership_sha256'],len(m['items']));print('FIRST_ITEM_KEYS',list(m['items'][0]))
for folder in ['/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain','/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2','/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/instance_repair/frozen_e1_20261012/room2']:
 print('FOLDER',folder)
 for p in Path(folder).glob('*'):print(p.name, 'DIR' if p.is_dir() else p.stat().st_size)
 for p in Path(folder).glob('*.json'):
  if p.stat().st_size<1000000:
   print('JSON',p.name);scan(json.loads(p.read_text()))
base=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2')
for folder in ['configs','geometry','surface_p0','official_geometry_cache']:
 for p in (base/folder).rglob('*.json'):
  if p.stat().st_size<1000000:print('CONFIG',str(p));scan(json.loads(p.read_text()))
print('MAPPING',Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/mapping_summary.json').read_text()[:5700])
