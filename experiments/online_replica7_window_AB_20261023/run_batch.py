"""Up to four branch processes; stop dispatching new jobs on any failure."""
from pathlib import Path
import os,sys,subprocess,json,time,fcntl
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
lock=(O/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=read(O/'run_plan.json');pending=[];active=[];finished=[];failed=[]
for scene in plan['scenes']:
 for branch in plan['branches']:
  p=O/scene/f'status_{branch}.json'
  if p.exists() and read(p).get('status')=='COMPLETE':finished.append([scene,branch])
  else:pending.append((scene,branch))
while pending or active:
 while pending and len(active)<4 and not failed:
  scene,branch=pending.pop(0);out=O/scene;out.mkdir(exist_ok=True)
  log=(out/f'driver_{branch}.log').open('a')
  p=subprocess.Popen([PY,'-B','-u',str(C/scene/'run_branch.py'),branch],cwd=C/scene,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  active.append(dict(scene=scene,branch=branch,process=p,log=log,started=time.time()))
 for row in active[:]:
  rc=row['process'].poll()
  if rc is None:continue
  row['log'].close();active.remove(row)
  (failed if rc else finished).append([row['scene'],row['branch']])
 write(O/'batch_status.json',dict(status='FAILED' if failed else 'RUNNING' if active or pending else 'COMPLETE',
  supervisor_pid=os.getpid(),active=[dict(scene=r['scene'],branch=r['branch'],pid=r['process'].pid,started=r['started']) for r in active],
  pending=pending,completed=finished,failed=failed,updated_unix=time.time()))
 if failed and not active:break
 if active:time.sleep(30)
if not failed:write(O/'batch_mapping_repair_complete.json',dict(status='PASS',completed=finished,checkpoints=plan['checkpoints']))
sys.exit(1 if failed else 0)
