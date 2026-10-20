"""Extend the validated literal stage-order experiment without changing E1 kernels."""
from pathlib import Path
import ast, hashlib, json, runpy, shutil

W=Path(__file__).resolve().parents[2]
O=Path(__file__).resolve().parent
C=W/'worktrees/auto-v2/experiments/auto_v2/replica8_swap34_20261020'
SRC=W/'worktrees/auto-v2/experiments/auto_v2'
SITE=W/'results/固定案例_三模型对比_20261006/replica8_auto_E1_20261019'
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
CHANGED=['office1','office3','office4']
REL='replica8_swap34_20261020'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def write(rel,text):
    p=C/rel;p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix=='.py':ast.parse(text)
    p.write_text(text,encoding='utf-8',newline='\n')

for n in ['filter_rules.py','semantic_exemption.py']:
    write('rules/'+n,(SRC/'room1_swap34_20261020/v2'/n).read_text(encoding='utf-8'))
config={}
for scene in SCENES:
    oldrel='auto_v2_room0_pr35_20261019' if scene=='room0' else 'auto_v2_replica6_20261019/'+scene
    b='/data/chenkejun/CVPR/results/'
    d=dict(scene=scene,v1=b+oldrel+'/v1',selection=b+oldrel+'/v2/selection.json',
           measurements=b+oldrel+'/v2/measurements.json',semantic=b+oldrel+'/v2/semantic',
           old_E1=b+oldrel+'/e1_repair',old_partial=b+'auto_v2_replica8_partial_20261019/'+scene)
    if scene=='room2':
        d.update(v1=b+'instance_seed_grouping_20261015_r4',
                 selection=b+'auto_v2_replay_20261019_r2/selection.json',
                 measurements=b+'auto_v2_input_reextract_20261019/measurements.json',
                 semantic=b+'seed_semantic_filter_20261018',old_E1=b+'e1_auto35_original_20261019')
    reports=SITE/'filtering'/scene/'reports'
    d['frozen_input_sha256']={d['selection']:sha(reports/'selection.json'),
        d['measurements']:sha(reports/'measurements.json'),
        d['semantic']+'/classification.json':sha(reports/'semantic.json'),
        d['v1']+'/seed_manifest.json':sha(reports/'v1_manifest.json')}
    d['old_count']=len(read(reports/'selection.json')['retained_group_ids'])
    new=read(O/(scene+'_preview_selection.json'))
    d['new_count']=len(new['retained_group_ids'])+(2 if scene=='room2' else 0)
    d['action']='rerun' if scene in CHANGED else 'reuse_completed_swap' if scene=='room1' else 'reuse_identical_selection'
    d['reuse_E1']=b+'room1_swap34_20261020/e1_repair' if scene=='room1' else d['old_E1']
    d['reuse_partial']=b+'room1_swap34_20261020/partial' if scene=='room1' else d['old_partial']
    config[scene]=d
write('config.json',json.dumps(config,ensure_ascii=False,indent=2))
numeric=['track_case.py','observation_ops.py','adaptive_core.py','gate_v1.py',
         'strict_vote_ops.py','joint_batch_ops.py','objectwise_ops.py','one_vote.py']
locks={}
for scene in CHANGED:
    old=SRC/'replica6_20261019'/scene
    oldrel='auto_v2_replica6_20261019/'+scene;newrel=REL+'/'+scene
    for src in (old/'e1_repair').iterdir():
        if src.suffix not in ['.py','.json'] or src.name in ['confirmed_selection.json','adapter_provenance.json']:continue
        t=src.read_text(encoding='utf-8').replace(oldrel,newrel)
        t=t.replace(f'auto-v2-{scene}-semantic-pr-20261019',f'auto-v2-{scene}-swap34-20261020')
        if src.name=='run_selected.py':
            t=t.replace(" run('input_audit',[C/'audit_selected_inputs.py'])",
                        " run('input_audit',[C/'audit_selected_inputs.py'])\n run('reuse_tracking',[C/'reuse_exact_tracking.py'])")
            t=t.replace("str(C/'tracking_queue_v2.py')","str(C/'serialized_tracking.py')")
        write(scene+'/e1_repair/'+src.name,t)
    t=(SRC/'room1_swap34_20261020/e1_repair/reuse_exact_tracking.py').read_text(encoding='utf-8')
    t=t.replace('room1_swap34_20261020',newrel).replace('auto_v2_replica6_20261019/room1',oldrel)
    write(scene+'/e1_repair/reuse_exact_tracking.py',t)
    write(scene+'/e1_repair/serialized_tracking.py',"""from pathlib import Path
import fcntl, subprocess, sys
C=Path(__file__).resolve().parent
with open('/data/chenkejun/CVPR/results/replica8_swap34_20261020/tracking_queue.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    subprocess.run([sys.executable,'-B','-u',str(C/'tracking_queue_v2.py')],check=True,cwd=C)
""")
    for name in ['partial_policy.py','test_partial_policy.py','controller.py','run_partial_replay.py','evaluate_r2.py','bootstrap.py']:
        t=(SRC/'replica8_partial_20261019'/name).read_text(encoding='utf-8')
        if name=='bootstrap.py':
            start=t.index('SCENE=os.environ');end=t.index('MAIN=Path')
            t=t[:start]+f"""SCENE={scene!r}
SOURCE=Path('/data/chenkejun/CVPR/results/{newrel}/e1_repair')
SOURCE_CODE=Path('/home/chenkejun/CVPR/experiments/{newrel}/e1_repair')
COUNT=len(json.loads((SOURCE/'confirmed_selection.json').read_text())['included'])
R=Path('/data/chenkejun/CVPR/results/{newrel}/partial')
"""+t[end:]
        write(scene+'/partial/'+name,t)
    write(scene+'/partial/adapter_provenance.json',json.dumps(dict(
        schema='swap34-partial-adapter-v1',scene=scene,selected_masks=config[scene]['new_count'],
        source_E1=f'/data/chenkejun/CVPR/results/{newrel}/e1_repair',
        only_partial_algorithm_change='old_observation_plans whole -> partial',
        unchanged=['selected seeds','tracking','identities','gates','trusted regions','geometry','confirmation','postprocessing'],
        GT_used_for_mapping=False),indent=2))
    for n in numeric:
        assert (C/scene/'e1_repair'/n).read_bytes()==(old/'e1_repair'/n).read_bytes()
    locks[scene]={n:sha(C/scene/'e1_repair'/n) for n in numeric}
    t=(SRC/'room1_swap34_20261020/controller.py').read_text(encoding='utf-8')
    t=t.replace('room1_swap34_20261020',newrel)
    t=t.replace("[('selection', C / 'v2/select_reused.py'),\n              ('original_E1'", "[('original_E1'")
    t=t.replace("scope='room1'",f"scope={scene!r}")
    write(scene+'/controller.py',t)
write('adapter_provenance.json',json.dumps(dict(
    request='8场景都跑新顺序看效果',only_filter_change='containment before coverage skip',
    stage5_window_cabinet_exemption=True,preserved_room2_historical_exceptions=['G017','G025'],
    original_E1_numerical_source_sha256=locks,changed_scenes=CHANGED,
    reuse_rule='exact same selected source masks, IDs and downstream settings; room1 completed swap reused',
    tracking_reuse='entire identical cases only',tracking_schedulers='serialized across scenes',maximum_scene_workers=2,
    evaluation='main default object_observed_repair revision 2; fixed 350 GT; frozen=false'),ensure_ascii=False,indent=2))
print(json.dumps({s:{k:d[k] for k in ['old_count','new_count','action']} for s,d in config.items()},indent=2))
