from pathlib import Path
import os,sys,json,subprocess,time,traceback
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
PY='/data/chenkejun/ovimap_runtime_20260908/envs/cropformer/bin/python'
def run(args,log,env=None):
 with log.open('a') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,cwd=C)
try:
 run([sys.executable,'-B',str(C/'prepare_inputs.py')],O/'prepare_inputs.log')
 for scene in ['scene0011_00','scene0050_00']:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH='/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09',CUDA_CACHE_PATH=str(O/'cuda_cache'),XDG_CACHE_HOME=str(O/'runtime_cache'))
  (O/'frontend_status.json').write_text(json.dumps(dict(status='RUNNING',scene=scene,pid=os.getpid())))
  run([PY,'-B','-u',str(C/'infer_scannet.py'),'--config',str(O/'inputs'/scene/'inference_config.json')],O/(scene+'_frontend.log'),env)
 run([sys.executable,'-B',str(C/'finish_inputs.py')],O/'finish_inputs.log')
 (O/'frontend_status.json').write_text(json.dumps(dict(status='PASS',frames=400)))
except Exception:
 (O/'frontend_status.json').write_text(json.dumps(dict(status='FAILED',traceback=traceback.format_exc())));raise
