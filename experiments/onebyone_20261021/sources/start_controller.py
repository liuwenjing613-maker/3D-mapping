from pathlib import Path
import json,os,subprocess,sys,time
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
assert not (R/'controller_launch.json').exists()
with (R/'controller.log').open('xb') as log:
    p=subprocess.Popen([sys.executable,'-B','-u',str(C/'controller.py')],cwd=C,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'})
v=dict(pid=p.pid,unix=time.time());(R/'controller_launch.json').write_text(json.dumps(v));print(v)
