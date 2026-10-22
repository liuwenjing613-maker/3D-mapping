from common import *
import subprocess,traceback
def main():
 done=[]
 for name in ("preflight","wait_tracking","repair_pipeline","commit_state"):
  if name=="wait_tracking":
   while not (R/"tracking_queue_complete.json").exists():
    if (R/"tracking_queue_failure.json").exists():raise RuntimeError("tracking queue failed")
    dump(TOP/"workflow_status.json",dict(stage=name,completed=done,unix=time.time()));time.sleep(10)
   continue
  start=time.perf_counter()
  with (R/(name+"_run.log")).open("a") as log:
   q=subprocess.Popen([PY,"-u",str(C/(name+".py"))],cwd=C,env=os.environ,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   dump(TOP/"workflow_status.json",dict(stage=name,pid=q.pid,completed=done,unix=time.time()))
   if q.wait()!=0:raise RuntimeError(name+" failed")
  done.append(dict(stage=name,seconds=time.perf_counter()-start))
 dump(TOP/"workflow_status.json",dict(status="PASS",stage="50_committed_ready_to_continue",completed=done,unix=time.time()))
if __name__=="__main__":
 try:main()
 except Exception:dump(TOP/"workflow_failure.json",dict(traceback=traceback.format_exc()));raise
