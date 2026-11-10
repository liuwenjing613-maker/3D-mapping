from pathlib import Path
import os,sys,json,time,subprocess,traceback,concurrent.futures
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
PY='/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
scene=sys.argv[1];status=O/(scene+'_status.json')
def state(stage,**kw):status.write_text(json.dumps(dict(status='RUNNING',stage=stage,pid=os.getpid(),**kw)))
def wait(p):
 while not p.exists():
  fs=O/'frontend_status.json'
  if fs.exists() and json.loads(fs.read_text())['status']=='FAILED':raise RuntimeError('Shared frontend failed')
  fail=O/'native'/scene/'full/failure.json'
  if fail.exists():raise RuntimeError('Full-depth geometry failed')
  time.sleep(10)
def run(stage,script,*args):
 state(stage)
 with (O/(scene+'_'+stage+'.log')).open('a') as f:subprocess.run([PY,'-B','-u',str(C/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=C,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
try:
 state('WAITING_SHARED_FRONTEND');wait(O/'inputs'/scene/'complete.json')
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
  ovi=pool.submit(run,'OVIMAP','run_ovi.py',scene)
  state('WAITING_FULL_DEPTH_TSDF');wait(O/'native'/scene/'full/complete.json')
  run('attributes','run_attributes.py',scene)
  run('strict_mapping','run_mapping_stage_A.py',scene,200)
  run('final_B','finalize_baseline.py',scene)
  run('original_evidence','build_original_dbscan_evidence.py',scene)
  state('WAITING_GT_SCOPE');wait(O/'ground_truth/scope_lock.json')
  run('score_baseline','evaluate_scene.py',scene,'no_repair')
  run('DBSCAN','run_dbscan.py',scene)
  ovi.result()
  run('score_OVIMAP','evaluate_scene.py',scene,'OVIMAP')
 status.write_text(json.dumps(dict(status='PASS',scene=scene,methods=['OVIMAP','no_repair','no_repair_DBSCAN'])))
except Exception:
 status.write_text(json.dumps(dict(status='FAILED',scene=scene,traceback=traceback.format_exc())));raise
