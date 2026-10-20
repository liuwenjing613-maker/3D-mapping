"""Data/path adapters only; archive every source and textual change."""
from pathlib import Path
import ast, hashlib, json, re, difflib
from patch_regroup import adapt as adapt_regroup
W=Path(__file__).resolve().parents[2]
O=Path(__file__).parent
C=O/'code'
NAME='scannet_e1_swap_partial_20261020'
REMOTE='/home/chenkejun/CVPR/experiments/'+NAME
R='/data/chenkejun/CVPR/results/'+NAME
SRC=W/'worktrees/auto-v2/experiments/auto_v2/replica6_20261019'
F=W/'worktrees/auto-v2/instance_repair/frozen_e1_20261012/room0'
SCENES={'scene0011_00':238,'scene0608_00':339}
changes=[]
def save(rel,s,p=None):
    path=C/rel;path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.py':ast.parse(s,filename=str(path))
    if path.suffix=='.json':json.loads(s)
    path.write_text(s,encoding='utf-8',newline='\n')
    if p:
        old=p.read_text(encoding='utf-8')
        changes.append(dict(file=rel,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            diff=''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True)))))

def generic(s,scene):
    s=s.replace('auto_v2_replica6_20261019',NAME).replace('office1',scene)
    s=s.replace('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main',R+'/inputs')
    s=s.replace('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002',R+'/compat')
    s=s.replace('/data/chenkejun/CVPR/datasets/cropformer_frozen/'+scene,R+'/inputs/'+scene+'/masks')
    s=s.replace('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs',REMOTE+'/configs')
    s=re.sub(r'\b1200\b','640',s);s=re.sub(r'\b680\b','480',s)
    s=s.replace('list(range(0,2000,5))','list(range(0,'+str((SCENES[scene]-1)*10+1)+',10))')
    s=s.replace('list(range(0, 2000, 5))','list(range(0, '+str((SCENES[scene]-1)*10+1)+', 10))')
    s=s.replace('list(range(2000))',f'list(range(0,{(SCENES[scene]-1)*10+1},10))')
    s=s.replace('range(2000)',f'range(0,{(SCENES[scene]-1)*10+1},10)')
    s=s.replace('==2000','=='+str(SCENES[scene]))
    for label in ["'verified_raw_frames':2000","'raw_RGB_files_checked':2000","'label_files_checked':2000"]:
        s=s.replace(label,label.replace('2000',str(SCENES[scene])))
    # Fixed counts are metadata/assertions in these adapters, not algorithmic thresholds.
    for pattern in ["'mapping_frames':400","'frame_count':400","'raw_frames_verified':400",
                    "'original_raw_frames_checked':400","'repair_conditions_replayed_frames':400",
                    'total_frames=400']:
        s=s.replace(pattern,pattern.replace('400',str(SCENES[scene])))
    return s

# The frozen Frame/Camera class and projection kernels stay intact.
p=F/'dependencies/revisable_instance_map/frame_io.py'
s=p.read_text(encoding='utf-8').replace('config["dataset"] != "Replica"','config["dataset"] != "ScanNet"')
s=s.replace('This source only supports the fixed Replica contract','Expected calibrated ScanNet input')
s=s.replace('        pose = self._poses[frame_id].copy()','        depth_m = np.where((depth_m > 0) & (depth_m < self.config["depth_policy"]["max_depth_m"]), depth_m, 0).astype(np.float32)\n        pose = self._poses[frame_id].copy()')
save('shared/frame_io.py',s,p)
p=F/'dependencies/run_repairs.py';s=p.read_text(encoding='utf-8')
s=s.replace('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002',R+'/compat')
s=s.replace('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main',R+'/inputs')
s=s.replace("assert sorted(self.records) == list(range(0, 2000, 5))","assert sorted(self.records) == json.loads((INPUT / scene / 'configs/raw.json').read_text())['frame_ids']")
save('shared/run_repairs.py',s,p)
save('shared/sitecustomize.py',f"""from pathlib import Path
import sys,json,os
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import LockedModules
F=MAIN/'instance_repair/frozen_e1_20261012/room0'
mapping=json.loads((F/'module_map.json').read_text())
mapping['revisable_instance_map.frame_io']={REMOTE+'/shared/frame_io.py'!r}
mapping['run_repairs']={REMOTE+'/shared/run_repairs.py'!r}
sys.meta_path.insert(0,LockedModules(F,mapping))
""")

# Existing ranking and candidate-view logic, using exactly the available original frame IDs.
for n in ['common.py','build_candidates.py']:
    p=SRC/'catalog_code'/n;s=p.read_text(encoding='utf-8')
    s=s.replace('auto_v2_replica6_20261019',NAME)
    s=s.replace("SCENES=['room0','room2','room1','office0','office1','office2','office3','office4']",f"SCENES={list(SCENES)!r}")
    s=s.replace('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main',R+'/inputs')
    if n=='common.py':
        start=s.index(" graph=Path(");end=s.index("\n config=",start)
        s=s[:start]+f" graph=Path({R!r})/'inputs'/scene/'review_surface_graph_n02_p08.npz'"+s[end:]
    else:
        s=s.replace('fids=np.arange(0,len(poses),5,dtype=int)',"fids=np.asarray(cfg['frame_ids'],dtype=int)")
        s=s.replace('for fid in range(len(poses)):',"for fid in cfg['frame_ids']:")
        s=s.replace("len(poses)","len(cfg['frame_ids'])")
        s=s.replace("src/f'results/frame{fid:06d}.jpg'","src/cfg['source']['rgb_pattern'].format(frame=fid)")
    save('catalog_code/'+n,s,p)

for scene,count in SCENES.items():
    for p in (SRC/'office1').rglob('*'):
        if p.suffix not in ['.py','.json'] or '__pycache__' in p.parts:continue
        rel=p.relative_to(SRC/'office1').as_posix()
        if p.name in ['adapter_provenance.json','confirmed_selection.json','evaluate_r2.py']:continue
        s=generic(p.read_text(encoding='utf-8'),scene)
        if rel=='frontend/prepare_inputs.py':
            s=s.replace(" assert len(entries)==60 and all(sum(e['group']==g for e in entries)==20 for g in ['CONFLICT','U','T'])"," assert entries and all(sum(e['group']==g for e in entries)<=20 for g in ['CONFLICT','U','T'])\n NENTRY=len(entries)\n NVIEW=sum(len(e['views']) for e in entries)")
            s=s.replace("assert NROI>0 and all(len(e['views'])==3 for e in entries)","assert NROI>0 and NVIEW>0 and all(0<=len(e['views'])<=3 for e in entries)")
            s=s.replace('==180','==NVIEW').replace('preferred_views=180','preferred_views=NVIEW').replace('views=180','views=NVIEW').replace('ranked_entries=60','ranked_entries=NENTRY')
            s=s.replace("first=cases[0]['views'][0]","first=next(v for c in cases for v in c['views'])")
        elif rel=='frontend/summarize_frontend.py':
            s=s.replace('total_views=180',"total_views=m['preferred_views']").replace('==180',"==m['preferred_views']").replace('views=180',"views=m['preferred_views']")
        elif rel=='overmerge/score.py':s=s.replace('view_entries=180',"view_entries=m['preferred_views']")
        elif rel=='frontend/normalize_manifest.py':s=s.replace("assert len(manifest['items'])==60","assert len(manifest['items'])==inputs['ranked_entries']")
        elif rel=='frontend/infer_cropformer.py':
            anchor="   rgb=read_image(job['input_file'],format='BGR');t=time.perf_counter();duplicate=job['same_RGB_as']"
            s=s.replace(anchor,anchor+f"""
   if job['kind']=='control':
    original_config=json.loads(Path('/home/chenkejun/CVPR/experiments/p1a1_scannet_engineering_fix_20261008/configs/{scene}_engineering.json').read_text())
    original_root=Path(original_config['source']['scene_root'])
    original_RGB=original_root/original_config['source']['rgb_pattern'].format(frame=job['frame'])
    rgb=read_image(str(original_RGB),format='BGR')
""".rstrip())
            anchor="    assert cv2.imwrite(str(png),labels)"
            s=s.replace(anchor,"""    if job['kind']=='control':
     intrinsic=original_root/'intrinsic'
     kd=np.loadtxt(intrinsic/'intrinsic_depth.txt')[:3,:3];kc=np.loadtxt(intrinsic/'intrinsic_color.txt')[:3,:3]
     vv,uu=np.indices(tuple(job['shape']),dtype=np.float64)
     rays=np.stack((uu,vv,np.ones_like(uu)),axis=-1)@(kc@np.linalg.inv(kd)).T
     labels=cv2.remap(labels,(rays[:,:,0]/rays[:,:,2]).astype(np.float32),(rays[:,:,1]/rays[:,:,2]).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
"""+anchor)
            s=s.replace("rec['frozen_cache_comparison']=agreement(cv2.imread(job['baseline_file'],cv2.IMREAD_UNCHANGED),labels)","rec['frozen_cache_comparison']=agreement(cv2.imread(job['baseline_file'],cv2.IMREAD_UNCHANGED),labels)\n    rec['original_RGB_sha256']=sha(original_RGB)\n    rec['control_path']='Original full color inference followed by calibrated nearest-label sampling, identical to historical ScanNet frontend'")
        elif rel=='e1_repair/bootstrap.py':
            s=s.replace("sys.meta_path.insert(0,LockedModules(F,mapping))",f"mapping['run_repairs']={REMOTE+'/shared/run_repairs.py'!r}\nmapping['revisable_instance_map.frame_io']={REMOTE+'/shared/frame_io.py'!r}\nsys.meta_path.insert(0,LockedModules(F,mapping))")
        elif rel=='e1_repair/run_case.py':
            s=s.replace('return range(len(poses))',"return raw['frame_ids']")
            s=s.replace("assert sorted(frames) == list(range(raw_count))","assert sorted(frames) == list(scene_frame_ids(cfg['scene']))")
        elif rel=='e1_repair/track_case.py':
            s=s.replace('total = len(poses)',"raw_ids = raw['frame_ids']\n    total = len(raw_ids)")
            s=s.replace('assert 0 <= fid < total',"assert fid in raw_ids\n    position = raw_ids.index(fid)")
            s=s.replace('list(range(fid, total))','raw_ids[position:]').replace('list(range(fid + 1))','raw_ids[:position + 1]')
            s=s.replace('assert sorted(records) == list(range(total))','assert sorted(records) == raw_ids')
        elif rel=='e1_repair/prepare_selection.py':
            s=s.replace(" inc=[];exc=[]",f""" # Approved execution eligibility: retain selection output, skip only zero forward support.
 import run_repairs as r
 import numpy as np
 from PIL import Image
 from dataclasses import replace
 scene=r.Scene({scene!r});skipped=[];support={{}}
 for tid,g in enumerate(source['groups'],1):
  if g['id'] not in kept:continue
  p=g['primary'];m=np.asarray(Image.open(R/'v1'/p['mask']))>0
  frame=scene.source.load_frame(p['frame'])
  pts,_,stats=r.project_frame_regions(replace(frame,mask_local=m.astype(np.uint16)),scene.tree,pixel_stride=2,max_distance_m=.015,workers=4)
  support[g['id']]=int(len(np.unique(pts)))
  if not len(pts):
   skipped.append(dict(group_id=g['id'],original_track_id=tid,mask_id=p['id'],frame=p['frame'],mask_sha256=p['mask_sha256'],reason='ZERO_VALID_FORWARD_PROJECTED_SURFACE_SUPPORT',valid_depth_pixels=int((m&(frame.depth_m>0)).sum()),projection=stats))
 skipped_ids={{x['group_id'] for x in skipped}}
 effective_kept=kept-skipped_ids
 scene.verify_unchanged()
 inc=[];exc=[]""")
            s=s.replace("(inc if g['id'] in kept else exc)", "(inc if g['id'] in effective_kept else exc)")
            s=s.replace(" assert inc", """ selection['original_automatic_retained_group_ids']=sorted(kept)
 selection['execution_skipped_seeds']=skipped
 selection['execution_seed_forward_support']=support
 selection['execution_skip_policy']={'approved_by_user':True,'rule':'Skip only zero valid forward projected support; same execution seeds for E1 and partial; unchanged 0<depth<4, pixel_stride=2 and maximum_distance_m=0.015','GT_used':False}
 assert inc""")
        elif rel=='e1_repair/prepare_auto_seeds.py':
            s=s.replace("assert {g['id'] for g in chosen['groups']}=={x['group_id'] for x in selection['included']}",
                "assert {g['id'] for g in chosen['groups']}=={x['group_id'] for x in selection['included']}|{x['group_id'] for x in selection.get('execution_skipped_seeds',[])}")
            s=s.replace("'source_policy':'Approved stage-5 semantic-class exemption; 35 original primary masks; M0466 source correction preserved'","'source_policy':'Approved ScanNet swap34 automatic masks; unchanged window/cabinet stage5 exemption; no manual overrides'")
        elif rel=='e1_repair/tracking_queue_v2.py':
            s=s.replace("or foreign or gpu['free']<16000", "or sum(x['memory'] for x in foreign)>=1024 or gpu['free']<24000")
        elif rel=='e1_repair/run_strict_repair.py':
            s=s.replace("priorfreeze['code_sha256']",f"read(PRIOR/'ablation/{scene}/mapping_complete.json')['code_sha256']")
        elif rel=='run_v1.py':
            s=s.replace("'PYTHONPATH':'/home/chenkejun/CVPR/repos/sam2_1'",f"'PYTHONPATH':{REMOTE+'/shared:/home/chenkejun/CVPR/repos/sam2_1'!r}")
            s=s.replace("uid not in busy and int(free)>12000","int(free)>24000 and int(util)<35")
        if scene=='scene0608_00':s=adapt_regroup(s,rel)
        save(scene+'/'+rel.replace('expand_office1','expand_'+scene),s,p)
    # Use today's validated containment-before-coverage order for both E1 and partial.
    p=W/'worktrees/auto-v2/experiments/auto_v2/replica8_swap34_20261020/rules/filter_rules.py'
    save(scene+'/v2/filter_rules.py',p.read_text(encoding='utf-8'),p)
    # Shared frozen inference, gates and numeric repair kernels are unchanged except data adapters.
    part=W/'worktrees/auto-v2/experiments/auto_v2/replica8_swap34_20261020/office1/partial'
    for n in ['partial_policy.py','test_partial_policy.py','controller.py','run_partial_replay.py','bootstrap.py','adapter_provenance.json']:
        p=part/n;s=p.read_text(encoding='utf-8')
        s=s.replace('replica8_swap34_20261020','auto_v2_replica6_20261019')
        s=generic(s,scene)
        if n=='bootstrap.py':
            s=s.replace("sys.meta_path.insert(0,LockedModules(F,mapping))",f"mapping['run_repairs']={REMOTE+'/shared/run_repairs.py'!r}\nmapping['revisable_instance_map.frame_io']={REMOTE+'/shared/frame_io.py'!r}\nsys.meta_path.insert(0,LockedModules(F,mapping))")
        if n=='run_partial_replay.py':
            s=s.replace("    groups = {g['id']: g for g in selected['groups']}", """    groups = {g['id']: g for g in selected['groups']}
    skipped = selection.get('execution_skipped_seeds', [])
    included_ids = {x['group_id'] for x in selection['included']}
    skipped_ids = {x['group_id'] for x in skipped}
    assert set(groups) == included_ids | skipped_ids and not included_ids & skipped_ids
    for item in skipped:
        assert item['reason'] == 'ZERO_VALID_FORWARD_PROJECTED_SURFACE_SUPPORT'
        assert selection['execution_seed_forward_support'][item['group_id']] == 0
    groups = {k: v for k, v in groups.items() if k in included_ids}""")
            s=s.replace("'count': COUNT, 'auto_v2_manifest': str(selection_path),",
                "'count': COUNT, 'execution_skipped_seeds': skipped, 'execution_seeds_identical_to_E1': True, 'auto_v2_manifest': str(selection_path),")
        save(scene+'/partial/'+n,s,p)
    # Both intermediate and final evaluation share the same main-source ScanNet adapter.
    for stage in ['e1_repair','partial']:
        save(scene+'/'+stage+'/evaluate_r2.py',f"""from pathlib import Path
import sys
sys.path.insert(0,{REMOTE!r})
from evaluate import evaluate_scene
evaluate_scene({scene!r}, include_partial={stage=='partial'!r})
""")
save('adapter_changes.json',json.dumps(dict(user_confirmed=True,scenes=SCENES,changes=changes),ensure_ascii=False,indent=2))
print('BUILT',len(changes))
