from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
old=json.loads((R/'controller_launch.json').read_text());pid=old['pid'];proc=Path(f'/proc/{pid}/cmdline')
assert not (R/'controller_v2_launch.json').exists()
if proc.exists():
    cmd=proc.read_bytes().split(b'\0');assert str(C/'controller.py').encode() in cmd
    assert Path(f'/proc/{pid}').stat().st_uid==os.getuid()
    os.kill(pid,signal.SIGTERM)
with (R/'controller_v2.log').open('xb') as log:
    p=subprocess.Popen([sys.executable,'-B','-u',str(C/'controller_v2.py')],cwd=C,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'})
v=dict(pid=p.pid,previous_controller_pid=pid,unix=time.time(),reason='Read actual main CLI paired_repair_metrics.json; preserve all existing mapper processes, inputs, and frozen code',
       code_sha256={n:hashlib.sha256((C/n).read_bytes()).hexdigest() for n in ['evaluate_v2.py','controller_v2.py','resume_v2.py']})
(R/'controller_v2_launch.json').write_text(json.dumps(v,indent=2));print(v)
