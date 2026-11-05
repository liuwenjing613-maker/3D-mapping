"""Official current-main pooled scoring of the same two scenes before/after DBSCAN."""
from pathlib import Path
import sys,os,time,fcntl,traceback,subprocess
from dbscan_common import C,O,COUNTS,read,dump,sha,run
variant=sys.argv[1];assert variant in ['no_repair','main','online_A','online_B']
out=O/'comparisons/pooled_native'/variant;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(status,**kw):dump(out/'status.json',dict(status=status,variant=variant,pid=os.getpid(),unix=time.time(),**kw))
try:
    if (out/'comparison.json').exists():assert read(out/'comparison.json')['status']=='PASS';sys.exit(0)
    while not all((O/'dbscan'/variant/s/'complete.json').exists() for s in COUNTS):
        for s in COUNTS:
            p=O/'dbscan'/variant/s/'status.json'
            if p.exists() and read(p)['status']=='FAILED':raise RuntimeError('DBSCAN failed: '+str(p))
        state('WAITING_TWO_SCENES_AFTER_DBSCAN');time.sleep(15)
    pools={};audit=[]
    for s in COUNTS:
        a=read(O/'evaluation'/variant/s/'r2/adapter_manifest.json');b=read(O/'dbscan'/variant/s/'r2/adapter_manifest.json')
        for k in ['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256']:assert a[k]==b[k],k
        assert sha(a['correspondence_cache'])==sha(b['correspondence_cache'])
        audit.append(dict(scene=s,geometry_GT_indices_equal=True,correspondence_sha256=sha(a['correspondence_cache'])))
    for arm in ['before_DBSCAN','after_DBSCAN']:
        dest=out/arm;dest.mkdir(parents=True,exist_ok=True);batch=[]
        for s in COUNTS:
            p=O/'evaluation'/variant/s/'r2' if arm=='before_DBSCAN' else O/'dbscan'/variant/s/'r2';a=read(p/'adapter_manifest.json')
            batch.append(dict(gt=a['gt_file'],prediction=str(p/'canonical_prediction.npz'),diagnostic_prediction=str(p/'diagnostic_support_prediction.npz')))
        dump(dest/'batch.json',dict(scenes=batch));state('RUNNING',stage=arm)
        with (dest/'run.log').open('a') as log:subprocess.run([sys.executable,'-B',str(C/'scannetpp_eval_batch.py'),'--scenes',str(dest/'batch.json'),'--out',str(dest)],cwd=C,stdout=log,stderr=subprocess.STDOUT,check=True)
        d=read(dest/'summary.json');assert d['CA_PRF1_0_5']['TP']+d['CA_PRF1_0_5']['FN']==70;pools[arm]=d
    dump(out/'comparison.json',dict(status='PASS',variant=variant,scenes=list(COUNTS),GT=70,pooled=pools,paired_audit=audit,per_scene=[read(O/'dbscan'/variant/s/'complete.json') for s in COUNTS],profile='object_observed_repair',revision=2,frozen=False,official_leaderboard=False))
    state('PASS',stage='TWO_SCENE_70GT_POOLED_COMPLETE');print('POOLED_PASS',variant,flush=True)
except Exception:state('FAILED',traceback=traceback.format_exc());raise
