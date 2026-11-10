"""Resume after frontend; approved boundary/ABC -> tracking -> gate -> repair -> commit."""
from pathlib import Path
import os,sys,time,json,fcntl,traceback
ROOTC=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
ROOTO=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
method,scene,n,group=sys.argv[1:5];n=int(n)
branch=ROOTC/method/scene;root=ROOTO/method/scene/str(n)/group
root.mkdir(parents=True,exist_ok=True)
lock=(root/'repair_handoff_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ.update(CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP=group,CVPR_SCENE=scene,CVPR_GPU_POOL='0,1,2,3,4,5,6,7',PYTHONDONTWRITEBYTECODE='1')
os.environ.pop('CVPR_ONLINE_GUARD',None);os.environ.pop('PYTHONPATH',None)
sys.path.insert(0,str(ROOTC));sys.path.insert(0,str(branch))
from common_continue import *
from selection_runtime import run
from source_guard import check
def status(state,**kw):dump(root/'repair_handoff_status.json',dict(status=state,method=method,scene=scene,checkpoint=n,group=group,pid=os.getpid(),unix=time.time(),**kw))
try:
    check()
    assert read(ROOTC/'adapter_audits/abc_camera/validation.json')['status']=='PASS'
    assert read(ROOTO/'online'/scene/'50/B/repair_runtime_adapter_validation.json')['status']=='PASS'
    while True:
        marker=root/'candidate_frontend_driver_status.json'
        if marker.exists():
            d=read(marker)
            if d['status']=='FAILED':raise RuntimeError('Frontend failed: '+str(marker))
            if d['status']=='PASS':break
        status('WAITING_FRONTEND');time.sleep(10)
    SC=branch/'selection_code'/str(n)/group;hook=ROOTC/'seed_hooks.py'
    steps=[('boundary',hook,['boundary'],PY,False,CANDIDATE_ROOT/'v1/audit.json'),
        ('semantic_background',SC/'v2/semantic/run.py',['--manifest',CANDIDATE_ROOT/'v1/seed_manifest.json','--raw-config',SC/'config.json','--out',CANDIDATE_ROOT/'v2/semantic','--config',SC/'v2/semantic/config.json'],PY,True,CANDIDATE_ROOT/'v2/semantic/classification.json'),
        ('v2',SC/'v2/measure_and_select.py',[],PY,False,CANDIDATE_ROOT/'v2/complete.json')]
    if method=='main':steps.append(('historical_exceptions',branch/'main_exceptions.py',[],PY,False,CANDIDATE_ROOT/'main_exceptions.json'))
    steps += [('qa',hook,['qa'],PY,False,CANDIDATE_ROOT/'online_qa.json'),
        ('projection_preflight',hook,['preflight'],PY,False,CANDIDATE_ROOT/'final_preflight/audit.json'),
        ('abc_semantics',hook,['semantics'],PY,True,CANDIDATE_ROOT/'abc_semantics/semantics.json'),
        ('final_abc',hook,['abc'],PY,False,CANDIDATE_ROOT/'final_abc/handoff.json')]
    for name,script,args,python,gpu,marker in steps:
        events=CANDIDATE_ROOT/'process_timing_events.jsonl'
        prior=[json.loads(s) for s in events.read_text().splitlines()] if events.exists() else []
        if marker.exists() and any(x['stage']==name and x['status']=='PASS' for x in prior):continue
        check();status('RUNNING',stage=name)
        run(name,script,args,python=python,gpu=gpu,out=CANDIDATE_ROOT)
    for name,script,marker in [('prepare','track_prepare.py',R/'prepare_complete.json'),('queue','track_queue.py',R/'tracking_queue_complete.json'),('gate','track_gate.py',R/'gate_complete.json')]:
        if marker.exists() and read(marker).get('status')=='PASS':continue
        check();status('RUNNING',stage='tracking_'+name);run(name,branch/script,out=R)
    dump(root/'selection_tracking_status.json',dict(status='PASS',checkpoint=n,group=group,stage='gate_ready_for_repair',unix=time.time()))
    for name,script,marker in [('repair','repair_continue.py',root/'repair/repair_complete.json'),('commit','commit_continue.py',root/'repair/commit_complete.json')]:
        if marker.exists() and read(marker).get('status')=='PASS':continue
        check();status('RUNNING',stage=name);run(name,branch/script,out=root/'repair')
    done=read(root/'repair/commit_complete.json');assert done['status']=='PASS' and done['processed_frames']==n
    dump(root.parent.parent/f'checkpoint{n}_{group}_complete.json',dict(status='PASS',scene=scene,group=group,checkpoint=n,previous_checkpoint=OLD_COUNT,
        commit=str(root/'repair/commit_complete.json'),initialized_from_zero=OLD_COUNT==0,prior_repaired_state_inherited=OLD_COUNT>0,GT_used=False))
    status('PASS',stage='CHECKPOINT_REPAIR_COMMITTED',final_B_complete=False,dbscan_complete=False,evaluation_complete=False)
except Exception:
    status('FAILED',traceback=traceback.format_exc());raise
