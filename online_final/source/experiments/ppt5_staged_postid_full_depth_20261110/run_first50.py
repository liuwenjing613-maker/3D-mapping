from pathlib import Path
import sys,os,json,time,subprocess,traceback,fcntl
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110')
scene=sys.argv[1];root=R/'online'/scene/'50/B';branch=C/'branches'/scene
lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=root/'controller_status.json'
def put(d):p=status.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(status)
try:
 put(dict(status='RUNNING',stage='ACTUAL_FIRST50_CANDIDATES_TRACK_GATE_REPAIR_COMMIT',pid=os.getpid()))
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
 env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with (root/'controller.log').open('ab') as f:subprocess.run(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(C/'first50_chain.py'),str(root),str(branch),str(C)],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert json.loads((root/'repair/commit_complete.json').read_text())['processed_frames']==50
 put(dict(status='PASS',stage='ACTUAL_FIRST50_REPAIR_COMMIT',pid=os.getpid()))
except BaseException:put(dict(status='FAILED',traceback=traceback.format_exc(),pid=os.getpid()));raise
