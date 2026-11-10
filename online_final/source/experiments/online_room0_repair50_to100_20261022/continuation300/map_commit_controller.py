"""Branch-isolated mapping/commit control; never advances past processed frame 300."""
from pathlib import Path
import os,json,time,subprocess,threading,traceback,hashlib
from execution_provenance import snapshot,verify
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300')
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
lock=threading.Lock()
state=dict(status='RUNNING',pid=os.getpid(),started_unix=time.time(),branches={})
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def ready(p):return Path(p).exists() and read(p).get('status')=='PASS'
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
def update(group,**kw):
 with lock:
  state['branches'][group]=dict(state['branches'].get(group,{}),**kw,updated_unix=time.time())
  write(T/'map_commit_controller_status.json',state)
def env_for(n,g,guard):
 return dict(os.environ,CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP=g,OPENBLAS_NUM_THREADS='4',
  OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(C/('map_guard' if guard=='mapping' else 'selection_guard')),
  CVPR_ONLINE_GUARD=str(T/str(n)/g/('mapping/input_guard.json' if guard=='mapping' else 'input_guard.json')))
def ensure_mapping(n,g):
 root=T/str(n)/g;folder=root/'mapping';launchpath=folder/'mapping_launch.json'
 if ready(folder/'mapping_complete.json'):return
 if launchpath.exists():
  launch=read(launchpath);assert alive(launch['pid']),str(folder/'mapping.log')
 else:
  setup_env=dict(os.environ,CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP=g)
  setup=subprocess.run([PY,'-B',str(C/'map_support.py')],env=setup_env,cwd=C,capture_output=True,text=True)
  assert setup.returncode==0,setup.stdout+setup.stderr
  log=folder/'mapping.log'
  provenance=snapshot(folder,'mapping',n,g)
  started=time.time()
  with log.open('wb') as stream:
   proc=subprocess.Popen([PY,'-B','-u',str(C/'map_continue.py')],env=env_for(n,g,'mapping'),cwd=C,stdout=stream,stderr=subprocess.STDOUT)
  launch=dict(status='STARTED',group=g,target=n,pid=proc.pid,started_unix=started,log=str(log),**provenance,
   code_sha256=hashlib.sha256((C/'map_continue.py').read_bytes()).hexdigest(),guard=str(folder/'input_guard.json'))
  write(launchpath,launch)
 update(g,stage='mapping',checkpoint=n,pid=launch['pid'])
 while not ready(folder/'mapping_complete.json'):
  if not alive(launch['pid']):raise RuntimeError('mapping exited without PASS: '+str(folder/'mapping.log'))
  time.sleep(5)
 verification=verify(folder)
 timing=read(folder/'mapping_timing.json')
 write(folder/'execution_timing.json',dict(status='PASS',pid=launch['pid'],
  started_unix=launch['started_unix'],ended_unix=timing['ended_unix'],
  seconds=timing['ended_unix']-launch['started_unix'],mapping_timing=str(folder/'mapping_timing.json'),
  scope='child process through final mapping report, no downstream wait',
  code_manifest=launch['code_manifest'],code_manifest_sha256=launch['code_manifest_sha256'],code_verification=verification))
def ensure_commit(n,g):
 root=T/str(n)/g;repair=root/'repair'
 if ready(repair/'commit_complete.json'):return
 update(g,stage='waiting_for_repair',checkpoint=n,pid=None)
 while not (ready(repair/'repair_complete.json') and ready(repair/'execution_timing.json')):
  timing=repair/'execution_timing.json'
  if timing.exists() and read(timing).get('status')=='FAILED':
   raise RuntimeError('repair failed; explicit review needed: '+str(timing))
  time.sleep(5)
 assert not (repair/'commit_failure.json').exists(),'Previous failed commit requires review'
 launchpath=repair/'commit_launch.json'
 if launchpath.exists():
  launch=read(launchpath);assert alive(launch['pid']),str(launch)
  update(g,stage='commit',checkpoint=n,pid=launch['pid'])
  while not ready(repair/'commit_complete.json'):
   if not alive(launch['pid']):raise RuntimeError('commit exited without PASS: '+str(launch))
   time.sleep(5)
  return
 provenance=snapshot(repair,'commit',n,g)
 log=repair/'commit_execution.log';start=time.perf_counter();started=time.time()
 with log.open('wb') as stream:
  proc=subprocess.Popen([PY,'-B','-u',str(C/'commit_continue.py')],env=env_for(n,g,'repair'),cwd=C,stdout=stream,stderr=subprocess.STDOUT)
 launch=dict(status='STARTED',group=g,target=n,pid=proc.pid,started_unix=started,log=str(log),**provenance,
  code_sha256=hashlib.sha256((C/'commit_continue.py').read_bytes()).hexdigest())
 write(launchpath,launch);update(g,stage='commit',checkpoint=n,pid=proc.pid)
 returncode=proc.wait();elapsed=time.perf_counter()-start
 passed=returncode==0 and ready(repair/'commit_complete.json')
 verification=verify(repair) if passed else None
 write(repair/'commit_execution_timing.json',dict(status='PASS' if passed else 'FAILED',returncode=returncode,
  seconds=elapsed,started_unix=started,ended_unix=time.time(),pid=proc.pid,scope='whole commit child process wall',
  code_manifest=launch['code_manifest'],code_manifest_sha256=launch['code_manifest_sha256'],code_verification=verification))
 assert passed,str(log)
 assert read(repair/'commit_complete.json')['processed_frames']==n
def branch(g):
 try:
  for n in (200,300):
   ensure_mapping(n,g);ensure_commit(n,g)
  update(g,status='PASS',stage='stopped_after_300_commit',checkpoint=300,pid=None)
 except Exception:
  update(g,status='FAILED',traceback=traceback.format_exc(),pid=None)
def main():
 threads=[threading.Thread(target=branch,args=(g,)) for g in ('A','B')]
 for thread in threads:thread.start()
 for thread in threads:thread.join()
 with lock:
  state['status']='PASS' if all(state['branches'][g].get('status')=='PASS' for g in ('A','B')) else 'FAILED'
  state['ended_unix']=time.time();write(T/'map_commit_controller_status.json',state)
if __name__=='__main__':main()
