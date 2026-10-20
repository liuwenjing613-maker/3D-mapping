"""Finish all original E1 scenes before permitting the partial-only phase."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019')
ROOM0=Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019')
SCENES=['room1','office0','office1','office2','office3','office4']
env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
status_lock=Lock()
def read(p):return json.loads(p.read_text())
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def run(scene,stage,args):
 result=R/scene/'stage_receipts'/(stage+'.json')
 if result.exists():
  record=read(result);assert record['status']=='PASS'
  for p,h in record['code_sha256'].items():
   current=hashlib.sha256(Path(p).read_bytes()).hexdigest()
   if current==h:continue
   # Early receipts snapshot later frontend code too. An audited unused-file
   # adapter may change after CPU preparation, without changing that preparation.
   amendment=read(R/scene/'input_view_adapter_amendment.json')
   relative=Path(p).relative_to(C/scene).as_posix()
   assert stage in ['baseline_audit','baseline','catalog']
   assert relative in ['frontend/prepare_inputs.py','frontend/summarize_frontend.py','overmerge/score.py']
   assert amendment['status']=='PASS' and amendment['amended_files_not_executed_by_completed_stages']
   assert any(x['file']==relative and x['before_sha256']==h and x['after_sha256']==current for x in amendment['changes']),p
  return
 folders=[C/scene,C/'baseline_code',C/'catalog_code']
 codes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in folders for p in folder.rglob('*.py')}
 state=dict(status='RUNNING',scene=scene,stage=stage,time=time.time())
 with status_lock:dump(R/'status.json',state)
 dump(R/scene/'status.json',state)
 with (R/scene/(stage+'.controller.log')).open('xb') as log:
  q=subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=C/scene,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  dump(R/scene/'active_process.json',dict(pid=q.pid,stage=stage,time=time.time()))
  assert q.wait()==0,(scene,stage)
 dump(result,dict(status='PASS',code_sha256=codes,time=time.time()))
def prepare_baseline(scene):
 run(scene,'baseline_audit',[C/'baseline_code/audit_frozen_votes.py',scene])
 run(scene,'baseline',[C/'baseline_code/run_ablation.py',scene])
 run(scene,'catalog',[C/'catalog_code/build_candidates.py',scene])
def main():
 while not (ROOM0/'complete.json').exists():
  assert not (ROOM0/'failure.json').exists(),'room0 failed'
  dump(R/'status.json',dict(status='WAITING',stage='room0_35_completion',time=time.time()));time.sleep(20)
 assert read(ROOM0/'complete.json')['status']=='PASS'
 # Independent CPU preparation may overlap; GPU inference/repair scenes remain ordered.
 with ThreadPoolExecutor(max_workers=3) as pool:
  prepared={scene:pool.submit(prepare_baseline,scene) for scene in SCENES}
  for scene in SCENES:
   prepared[scene].result()
   run(scene,'auto_v1',[C/scene/'run_v1.py'])
   run(scene,'v2_E1_R2',[C/scene/'run_remaining.py'])
   assert read(R/scene/'e1_repair/run_complete.json')['status']=='PASS'
   dump(R/scene/'status.json',dict(status='PASS',stage='original_E1_complete',time=time.time()))
 dump(R/'whole_phase_complete.json',dict(status='PASS',scenes=SCENES,time=time.time()))
 dump(R/'status.json',dict(status='PASS',stage='original_E1_all_six_complete',time=time.time()))
 # The separately frozen partial replay will be launched only after this receipt exists.
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'failure.json',dict(status='FAIL',traceback=traceback.format_exc(),time=time.time()));raise
