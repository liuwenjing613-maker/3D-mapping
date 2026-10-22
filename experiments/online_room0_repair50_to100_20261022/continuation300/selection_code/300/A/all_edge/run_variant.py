from pathlib import Path
import sys,subprocess,os,json,time,traceback
n=int(sys.argv[1]);gpu=sys.argv[2]
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300/selection_code/300/A')
R=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/300/A/candidates')
O=R/'all_edge_20261021'
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
HERE=Path(__file__).resolve().parent
env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
 'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09',
 'PYTHONPATH':str(C/'guard'),'CVPR_ONLINE_GUARD':str(R/'input_guard.json'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
 'PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(R/'runtime_cache'),'CUDA_CACHE_PATH':str(R/'cuda_cache')}
jobs=[('semantic',[str(C/'v2/semantic/run.py'),'--manifest',str(O/'v1/seed_manifest.json'),'--raw-config',str(C/'config.json'),'--out',str(O/'v2/semantic'),'--config',str(C/'v2/semantic/config.json')]),
 ('select',[str(HERE/'screen.py'),str(n),'select'])]
try:
 for name,args in jobs:
  with (O/(name+'.log')).open('wb') as log:
   p=subprocess.Popen([PY,'-B','-u',*args],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  (O/'progress.json').write_text(json.dumps(dict(stage=name,pid=p.pid,at=time.time())))
  code=p.wait()
  if code:raise RuntimeError(name+' exit '+str(code))
 (O/'progress.json').write_text(json.dumps(dict(stage='complete',at=time.time())))
except Exception:
 (O/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc())))
 raise
