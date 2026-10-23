"""One-shot recovery after the original supervisor and its active branches finish."""
from pathlib import Path
import json,os,time,subprocess,fcntl,traceback
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
read=lambda p:json.loads(Path(p).read_text())
def status(**row):
 p=O/'recovery_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(unix=time.time(),pid=os.getpid(),**row),indent=2)+'\n');q.replace(p)
def matches(pid,script):
 try:return str(script).encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
 except FileNotFoundError:return False
def main():
 lock=(O/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(O/'status_write_race_fix.json')['status']=='PASS'
 original=read(O/'launch.json')['pid']
 while matches(original,C/'run_batch.py'):
  state=read(O/'batch_status.json')
  status(status='WAITING_ORIGINAL_BRANCHES',original_supervisor=original,active=state['active'],failed=state['failed'])
  time.sleep(60)
 state=read(O/'batch_status.json')
 assert state['status']=='FAILED' and not state['active'],state
 assert state['failed']==[['office3','A']],('Additional failure needs diagnosis',state['failed'])
 failure=read(O/'office3/50/A/tracking/tracking_queue_failure.json')['traceback']
 assert 'FileNotFoundError' in failure and 'gpu_wait.json.tmp' in failure
 assert read(O/'office3/status_A.json')['status']=='FAILED'
 branch_scripts=[str(C/s/'run_branch.py').encode() for s in read(O/'run_plan.json')['scenes']]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=p.read_bytes().split(b'\0')
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
  assert not any(script in args for script in branch_scripts),('Branch still active',p.parent.name)
 status(status='RELAUNCHING_VALIDATED_RESUME',original_supervisor=original)
 subprocess.run(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B',str(C/'launch_batch.py')],cwd=C,check=True)
 status(status='RESUMED',launch=read(O/'launch.json'),failed_branch=['office3','A'],completed_stages_retained=True)
if __name__=='__main__':
 try:main()
 except Exception:status(status='FAILED',traceback=traceback.format_exc());raise
