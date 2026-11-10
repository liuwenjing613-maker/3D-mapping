"""Overlap independent scenes while retaining the original coordinator's in-memory resume hash checks."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,signal,time,traceback
import driver_bounded as d
O=d.O
def main():
 pid=d.read(O/'bounded_driver_pid.json')['pid']
 cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
 assert str(d.C/'driver_bounded.py') in cmd
 assert d.read(O/'driver_status.json')['stage']=='ROOM0_400_RUNNING'
 assert d.read(O/'native/room0_400_trace/progress.json')['processed_frames']<390
 assert d.read(O/'prefix_input_isolation_validation.json')['status']=='PASS'
 for name,h in d.read(O/'implementation_bounded_freeze.json')['files'].items():
  assert hashlib.sha256((d.C/name).read_bytes()).hexdigest()==h
 d.dump(O/'overlap_schedule.json',{'status':'RUNNING','coordinator_pid':pid,'scheduler_pid':os.getpid(),
  'started_unix':time.time(),'reason':'independent seven scenes overlap validated room0 run; frozen algorithms unchanged',
  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'completed':[]})
 os.kill(pid,signal.SIGSTOP)
 try:
  done=[]
  scenes=['room1','room2','office0','office1','office2','office3','office4']
  with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
   futures={pool.submit(d.scene_run,s):s for s in scenes}
   for future in concurrent.futures.as_completed(futures):
    scene=future.result();done.append(scene)
    status=d.read(O/'overlap_schedule.json');status['completed']=done
    d.dump(O/'overlap_schedule.json',status);print('SCENE_DONE',scene,flush=True)
  status=d.read(O/'overlap_schedule.json');status.update(status='COMPLETE',ended_unix=time.time())
  d.dump(O/'overlap_schedule.json',status)
 except BaseException as exc:
  status=d.read(O/'overlap_schedule.json');status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
  d.dump(O/'overlap_schedule.json',status);raise
 finally:
  os.kill(pid,signal.SIGCONT)
if __name__=='__main__':main()
