"""Run unchanged guarded completion on real completed native outputs, then evaluate."""
from pathlib import Path
import os,sys,time,json,fcntl,traceback
from dbscan_common import C,O,COUNTS,read,dump,load,sha
import dbscan_history as h
g=h.g
variant,scene=sys.argv[1:3];assert variant in ['no_repair','main','online_A','online_B'];assert scene in COUNTS
out=O/'dbscan'/variant/scene;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(status,**kw):dump(out/'status.json',dict(status=status,variant=variant,scene=scene,pid=os.getpid(),unix=time.time(),**kw))
try:
    from source_guard import check
    check()
    if (out/'complete.json').exists():
        old=read(out/'complete.json');assert old['status']=='PASS' and sha(out/'instance_surface.npz')==old['prediction_sha256'];sys.exit(0)
    while not (O/'evaluation'/variant/scene/'complete.json').exists():
        q=O/'published'/variant/scene/'finish_status.json'
        if q.exists() and read(q)['status']=='FAILED':raise RuntimeError('Final B or pre-DBSCAN evaluation failed')
        state('WAITING_FINAL_EVALUATION_BEFORE_DBSCAN');time.sleep(15)
    assert read(O/'evaluation'/variant/scene/'complete.json')['status']=='PASS'
    if variant=='no_repair':
        p=O/'dbscan_inputs/no_repair'/scene/'complete.json'
        while not p.exists():
            q=p.with_name('status.json')
            if q.exists() and read(q)['status']=='FAILED':raise RuntimeError('Original observation evidence builder failed')
            state('WAITING_ORIGINAL_EVIDENCE');time.sleep(15)
        d=read(p);assert d['status']=='PASS' and d['all_saved_observation_support_exact'] and sha(p.with_name('source_pixel_support.npz'))==d['output_sha256']
    state('RUNNING',stage='guarded_merge_and_fill');check();g.one(variant,scene)
    base=O/'dbscan/base'/variant/scene
    if variant.startswith('online_'):
        state('RUNNING',stage='legitimate_historical_core_completion');h.core_completion(variant,scene,base,out)
    else:
        original=load(O/'published'/variant/scene/'final_B/instance_surface.npz');labels=load(base/'merge_fill/instance_surface.npz')['instance_id']
        dump(out/'decision_freeze.json',dict(status='FROZEN_BEFORE_EVALUATION',GT_used=False,historical_core_used=False,reason='No previous repaired checkpoint exists for original mapping or final-only offline main repair',guarded_decisions_sha256=sha(base/'decision_freeze.json'),merge_fill_prediction_sha256=sha(base/'merge_fill/instance_surface.npz')))
        g.evaluate(variant,scene,'guarded_no_historical_core',original,labels,out,dict(GT_used_for_decision=False,history_core_used=False,final_B_once=True,no_online_feedback=True,terminal_only=True,decision_freeze_sha256=sha(out/'decision_freeze.json')))
    assert read(out/'complete.json')['status']=='PASS';state('PASS',stage='DBSCAN_AND_CURRENT_MAIN_R2_COMPLETE')
    print('DBSCAN_SCENE_PASS',variant,scene,flush=True)
except Exception:state('FAILED',traceback=traceback.format_exc());raise
