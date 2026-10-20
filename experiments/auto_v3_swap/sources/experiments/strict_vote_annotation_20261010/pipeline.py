import os,sys,time,json,subprocess
from pathlib import Path
C=Path('/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010')
R=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
s=sys.argv[1];r=R/s;c=C/s
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(r/'cuda_cache'),'XDG_CACHE_HOME':str(r/'runtime_cache'),'PYTHONPATH':'/home/chenkejun/CVPR/repos/sam2_1','LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
def run(script,python=PY,gpu=None):
 e=dict(env)
 if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
 f=(r/({'prepare_inputs.py':'prepare','infer_cropformer.py':'cropformer','infer_sam.py':'sam2_1','render_annotation.py':'render'}[script]+'.log')).open('ab')
 p=subprocess.Popen([python,'-u',str(c/script)],stdout=f,stderr=subprocess.STDOUT,env=e);f.close();return p
try:
 deadline=time.time()+1800
 while not (R/'catalog'/s/'candidate_manifest.json').exists():
  status=R/'catalog'/s/'progress.json'
  if status.exists() and json.loads(status.read_text()).get('stage')=='失败':raise RuntimeError(status.read_text())
  if time.time()>deadline:raise TimeoutError('catalog not ready in 30 minutes')
  time.sleep(10)
 p=run('prepare_inputs.py');assert p.wait()==0,'prepare failed'
 g=0 if s=='room0' else 2
 jobs=[run('infer_cropformer.py','/data/chenkejun/CVPR/runtime/cropformer_env/bin/python',g),run('infer_sam.py','/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python',g+1)]
 codes=[p.wait() for p in jobs];assert codes==[0,0],str(codes)
 p=run('render_annotation.py');assert p.wait()==0,'render failed'
 (r/'pipeline_complete.json').write_text(json.dumps({'status':'READY','scene':s}))
except Exception as e:
 (r/'pipeline_error.json').write_text(json.dumps({'error':repr(e)}));raise
