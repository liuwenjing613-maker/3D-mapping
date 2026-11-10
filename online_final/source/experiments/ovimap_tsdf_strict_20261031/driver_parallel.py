"""Validated rollout: room0 first, then remaining seven scenes, then main R2 pooling."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,signal,subprocess,sys,time,traceback
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
P='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def run(cmd,log):
 with Path(log).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=C,env=ENV,check=True)
def stage(name,**kw):
 dump(O/'driver_status.json',{'stage':name,'time_unix':time.time(),**kw});print(name,kw,flush=True)
def scene_run(scene):
 logdir=O/'driver';logdir.mkdir(exist_ok=True)
 native=O/'native'/f'{scene}_400_trace'
 own=O/'ours'/f'{scene}_400_main'
 if (O/'evaluation'/scene/'comparison.json').exists():return scene
 if native.exists() or own.exists():raise RuntimeError('Incomplete existing scene needs explicit audited resume: '+scene)
 commands=[
  ['python3',str(C/'native_runner.py'),scene,'400','trace'],
  [P,'-B',str(C/'baseline_parallel.py'),'--scene',scene,'--count','400','--native-root',str(native),'--tag','main']]
 children=[]
 try:
  for index,cmd in enumerate(commands):
   f=(logdir/f'{scene}_{index}.log').open('w')
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=C,env=ENV,start_new_session=True)
   children.append(child);f.close()
  dump(logdir/f'{scene}_pids.json',{'native':children[0].pid,'ours':children[1].pid})
  while any(p.poll() is None for p in children):
   for p in children:
    if p.poll() not in (None,0):raise RuntimeError('Scene process failed '+scene+' pid '+str(p.pid)+' rc '+str(p.returncode))
   time.sleep(5)
  for p in children:
   if p.returncode!=0:raise RuntimeError('Scene process failed: '+scene)
  run([P,'-B',str(C/'evaluate_parallel.py'),scene],logdir/f'{scene}_evaluation.log')
  return scene
 except BaseException:
  for p in children:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  raise
def adopt_room0():
 old_driver=read(O/'driver_pid.json')['pid'];pids=read(O/'driver/room0_pids.json')
 # Check all targets by command line; only this experiment's owned processes are stopped.
 expected={old_driver:'driver.py',pids['ours']:'baseline.py',pids['native']:'native_runner.py'}
 for pid,name in expected.items():
  cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
  assert str(C/name) in cmd,(pid,cmd)
 os.kill(old_driver,signal.SIGTERM)
 os.killpg(pids['ours'],signal.SIGTERM)
 for _ in range(100):
  if not Path('/proc/%d'%pids['ours']).exists():break
  time.sleep(.1)
 root=O/'ours/room0_400_main';checkpoint=read(root/'checkpoint.json')
 records=[json.loads(x) for x in (root/'audit.jsonl').read_text().splitlines()]
 # Archive the interrupted audit; verified checkpoint state is reused without changing observations.
 (O/'pre_parallel_room0_audit.jsonl').write_text((root/'audit.jsonl').read_text())
 replay_check={r['frame_id']:hashlib.sha256((root/'maps'/('f%06d.npz'%r['frame_id'])).read_bytes()).hexdigest() for r in records}
 dump(O/'parallel_takeover.json',{'old_driver_pid':old_driver,'old_ours_pid':pids['ours'],
   'native_pid_continues':pids['native'],'checkpoint_frames':checkpoint['processed_frames'],
   'previously_published_frames':len(records),'reason':'exact-equivalent parallel and conservative spatial query optimization; all strict rules unchanged'})
 f=(O/'driver/room0_parallel.log').open('w')
 child=subprocess.Popen([P,'-B',str(C/'baseline_parallel.py'),'--scene','room0','--count','400',
   '--native-root',str(O/'native/room0_400_trace'),'--tag','main','--resume'],
   stdout=f,stderr=subprocess.STDOUT,cwd=C,env=ENV,start_new_session=True);f.close()
 dump(O/'driver/room0_parallel_pids.json',{'native':pids['native'],'ours':child.pid})
 while child.poll() is None or Path('/proc/%d'%pids['native']).exists():
  if child.poll() not in (None,0):raise RuntimeError('Parallel baseline failed')
  if not Path('/proc/%d'%pids['native']).exists():
   assert 'NATIVE_DONE' in (O/'driver/room0_0.log').read_text(),'Native producer failed'
  time.sleep(5)
 assert child.returncode==0
 for fid,h in replay_check.items():assert hashlib.sha256((root/'maps'/('f%06d.npz'%fid)).read_bytes()).hexdigest()==h,('resume_replay_mismatch',fid)
 run([P,'-B',str(C/'evaluate_parallel.py'),'room0'],O/'driver/room0_evaluation.log')

def main():
 stage('VALIDATION_GATE')
 for name in ['native_smoke_validation_20.json','causal_resume_validation.json','frozen_reference_validation.json','parallel_validation.json']:
  assert read(O/name)['status']=='PASS',name
 assert read(O/'ours/room0_20_validation_v2/complete.json')['status']=='PASS'
 hashes={str(p.relative_to(C)):hashlib.sha256(p.read_bytes()).hexdigest() for p in C.rglob('*') if p.is_file() and p.suffix in ['.py','.h','.cpp','.so','.txt']}
 dump(O/'implementation_parallel_freeze.json',{'time_unix':time.time(),'files':hashes,'approval':'user confirmed 2026-10-31','native_threads':1})
 stage('ROOM0_400_RUNNING')
 adopt_room0()
 stage('REMAINING_7_RUNNING',completed=['room0'])
 scenes=['room1','room2','office0','office1','office2','office3','office4'];done=['room0']
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for scene in pool.map(scene_run,scenes):
   done.append(scene);stage('REMAINING_7_RUNNING',completed=done)
 stage('POOLING_R2')
 run([P,'-B',str(C/'evaluate_parallel.py'),'pool'],O/'pooled_run.log')
 for name,h in hashes.items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==h,('code_changed',name)
 stage('COMPLETE',completed=done,comparison=read(O/'comparison.json'))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  stage('FAILED',error=repr(exc),traceback=traceback.format_exc());raise
