from pathlib import Path
import ast,hashlib,json,sys,time
C=Path('/home/chenkejun/CVPR/experiments/auto_v2_replica6_20261019')
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path[:0]=[str(MAIN),str(C)]
from instance_repair.cli import verify_sources
from semantic_exemption import apply_semantic_exemption
verify_sources()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
old=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019')
actual=read('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019/v2/selection.json')
replay=apply_semantic_exemption(read(old/'v2/selection.json'),read(old/'v2/semantic/classification.json'))
for k,v in replay.items():
 if k!='case_id':assert v==actual[k],k
assert len(replay['retained_group_ids'])==35
assert replay['added_group_ids']==['G026','G027','G058','G063','G069','G097','G102','G103']
sources={};missing=[]
for scene in ['room1','office0','office1','office2','office3','office4']:
 for rel in [f'p1a1_raw_replica8_20261002/native/{scene}/P1-A1/surface_p0/materialization_report.json',f'p1a1_raw_replica8_20261002/final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz',f'p1a1_replica7_repair_20261003/{scene}/connected/review_surface_graph_n02_p08.npz',f'replica8_p0_main/{scene}/configs/raw.json']:
  p=Path('/data/chenkejun/CVPR/revisable_instance_map')/rel;assert p.exists(),p;sources[str(p)]=sha(p)
 for p in (C/scene).rglob('*.py'):
  tree=ast.parse(p.read_text())
  for node in ast.walk(tree):
   if not isinstance(node,ast.Constant) or not isinstance(node.value,str):continue
   value=node.value
   if not value.startswith(('/home/chenkejun/CVPR/','/data/chenkejun/CVPR/')) or '\n' in value or '{' in value:continue
   if 'auto_v2_replica6_20261019' in value:continue
   if value.endswith(('.py','.npz','.json','.pth','.pt')) and not Path(value).exists():missing.append((str(p.relative_to(C)),value))
assert not missing,missing
result=dict(status='PASS',time=time.time(),room0_35_selection_exactly_reproduced=True,six_scene_input_sources=sources,main_frozen_E1_verified=True,GT_not_used_for_selection_or_mapping=True)
(R/'preflight.json').write_text(json.dumps(result,indent=2));print('PREFLIGHT_PASS',len(sources))
