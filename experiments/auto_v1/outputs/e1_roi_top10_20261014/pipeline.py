from pathlib import Path
import subprocess,os,json,time
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/e1_roi_top10_20261014')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(R/'cuda_cache'),'XDG_CACHE_HOME':str(R/'runtime_cache'),'PYTHONPATH':'/home/chenkejun/CVPR/repos/sam2_1','LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
def run(name,python=PY,gpu=None):
 e=dict(env)
 if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
 with (R/(name+'.log')).open('ab') as f:return subprocess.Popen([python,'-u',str(C/(name+'.py'))],stdout=f,stderr=subprocess.STDOUT,env=e)
try:
 for name in ['audit_inputs','prepare_inputs']:
  assert run(name).wait()==0,name+' failed'
 jobs=[run('infer_cropformer','/data/chenkejun/CVPR/runtime/cropformer_env/bin/python',0),run('infer_sam','/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python',1)]
 codes=[j.wait() for j in jobs];assert codes==[0,0],str(codes)
 assert run('render_annotation').wait()==0,'render failed'
 (R/'pipeline_complete.json').write_text(json.dumps({'status':'READY','completed_at':time.time()}))
except Exception as e:
 (R/'pipeline_error.json').write_text(json.dumps({'error':repr(e)}));raise
