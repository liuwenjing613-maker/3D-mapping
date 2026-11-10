"""Run only the shared original mapping stage; never claims repair/final-B/evaluation complete."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback,fcntl
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
scene,n=sys.argv[1],int(sys.argv[2]);assert scene in ['scene0011_00','scene0050_00']
branch=C/'online'/scene;out=O/'online'/scene/str(n)/'B'
out.mkdir(parents=True,exist_ok=True)
lock=(out/'mapping_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(C));from source_guard import check
def dump(p,d):
    p=Path(p);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def read(p):return json.loads(Path(p).read_text())
env={**os.environ,'CVPR_CHECKPOINT':str(n),'CVPR_TRACK_GROUP':'B','CVPR_MAPPING_RUN_ROOT':str(out),
    'CVPR_SURFACE_POLICY':'checkpoint','CVPR_SCENE':scene,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
def run(stage,script,guard=False):
    check()
    e=dict(env)
    if guard:e.update(CVPR_ONLINE_GUARD=str(out/'mapping/input_guard.json'),PYTHONPATH=str(branch/'map_guard')+':'+str(branch))
    with (out/(stage+'.log')).open('a') as log:
        p=subprocess.Popen([PY,'-B','-u',str(branch/script)],cwd=branch,env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        dump(out/'mapping_driver_status.json',dict(status='RUNNING',scene=scene,frames=n,stage=stage,pid=p.pid,driver_pid=os.getpid(),started=time.time()))
        rc=p.wait()
    assert rc==0,(stage,rc)
try:
    assert read(O/'attributes'/scene/str(n)/'complete.json')['frames']==n
    if not (O/'online'/scene/'initial/B/initial_complete.json').exists():run('initialize','initialize_empty.py')
    if not (out/'mapping/mapping_complete.json').exists():
        if not (out/'mapping/mapping_plan.json').exists():run('mapping_setup','map_support.py')
        run('mapping','map_continue.py',True)
    m=read(out/'mapping/mapping_complete.json');assert m['status']=='PASS' and m['frame_ids']==list(range(0,n*5,5))
    dump(out/'mapping_driver_status.json',dict(status='PASS',scene=scene,frames=n,stage='MAPPING_ONLY_COMPLETE',repair_complete=False,final_B_complete=False,evaluation_complete=False,ended=time.time()))
except Exception:
    dump(out/'mapping_driver_status.json',dict(status='FAILED',scene=scene,frames=n,traceback=traceback.format_exc(),ended=time.time()))
    raise
