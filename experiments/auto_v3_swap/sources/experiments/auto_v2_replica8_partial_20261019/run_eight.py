from pathlib import Path
import json,os,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica8_partial_20261019')
WHOLE=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019')
SCENES=['room0','room2','room1','office0','office1','office2','office3','office4']
def read(p):return json.loads(p.read_text())
def save(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def run(scene,stage,script):
 save(scene+'/phase_status.json',dict(status='RUNNING',scene=scene,stage=stage,time=time.time()))
 env={**os.environ,'PARTIAL_SCENE':scene,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
 with (R/scene/(stage+'.controller.log')).open('xb') as f:
  p=subprocess.Popen([sys.executable,'-B','-u',str(C/script)],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  save(scene+'/active_process.json',dict(pid=p.pid,stage=stage,time=time.time()))
  assert p.wait()==0,(scene,stage)
def run_scene(scene):
  (R/scene).mkdir(exist_ok=True)
  if (R/scene/'run_complete.json').exists():assert read(R/scene/'run_complete.json')['status']=='PASS';return scene
  if scene=='room2':run(scene,'reuse_validation','reuse_room2.py')
  if (R/scene/'reuse_receipt.json').exists():
   run(scene,'evaluation','evaluate_r2.py')
   save(scene+'/run_complete.json',dict(status='PASS',reused_mapping=True,time=time.time()))
  else:run(scene,'partial','controller.py')
  assert read(R/scene/'evaluation_r2/comparison.json')['status']=='PASS'
  save(scene+'/phase_status.json',dict(status='PASS',scene=scene,time=time.time()))
  return scene
def main():
 while not (WHOLE/'whole_phase_complete.json').exists():
  assert not (WHOLE/'failure.json').exists(),'Original E1 phase failed; partial phase must not start.'
  save('status.json',dict(status='WAITING',stage='all_six_original_E1_completion',time=time.time()));time.sleep(30)
 assert read(WHOLE/'whole_phase_complete.json')['status']=='PASS'
 completed=[]
 save('status.json',dict(status='RUNNING',stage='partial_two_scene_workers',completed=completed,time=time.time()))
 with ThreadPoolExecutor(max_workers=2) as pool:
  pending=[pool.submit(run_scene,scene) for scene in SCENES]
  for task in as_completed(pending):
   completed.append(task.result())
   save('status.json',dict(status='RUNNING',stage='partial_two_scene_workers',completed=completed,time=time.time()))
 save('complete.json',dict(status='PASS',scenes=SCENES,time=time.time()))
 save('status.json',dict(status='PASS',stage='all_partial_complete',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:save('failure.json',dict(status='FAIL',traceback=traceback.format_exc(),time=time.time()));raise
