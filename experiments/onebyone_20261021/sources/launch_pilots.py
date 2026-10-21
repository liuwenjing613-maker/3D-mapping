from pathlib import Path
import json,os,subprocess,sys,time
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
assert json.loads((R/'tests_complete.json').read_text())['status']=='PASS'
for group,scene in [('automatic','room0'),('manual','office0')]:
    out=R/group/scene
    assert not (out/'launch.json').exists()
    with (out/'mapping.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',str(C/'replay.py')],cwd=C,
            env={**os.environ,'POINT_GROUP':group,'POINT_SCENE':scene,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=p.pid,unix=time.time(),group=group,scene=scene);(out/'launch.json').write_text(json.dumps(info));print(info)
