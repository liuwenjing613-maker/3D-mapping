from pathlib import Path
import os,sys,json,time,traceback,subprocess
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
scene=sys.argv[1];status=O/(scene+'_online_continuation_status.json')
def state(s,**kw):
 q=status.with_suffix('.tmp');q.write_text(json.dumps(dict(status=s,scene=scene,unix=time.time(),**kw),indent=2));q.replace(status)
def read(p):return json.loads(p.read_text())
def run(stage,script,args,n):
 state('RUNNING',checkpoint=n,stage=stage)
 with (O/(scene+'_online'+str(n)+'_'+stage+'.log')).open('a') as f:
  subprocess.run([PY,'-B','-u',str(C/script),*args],cwd=C,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1'))
try:
 for n,old in [(120,50),(200,120)]:
  prev=O/'online'/scene/str(old)/'B';marker=prev/'repair_handoff_status.json'
  while True:
   if marker.exists():
    d=read(marker)
    if d['status']=='FAILED':raise RuntimeError('Previous real repair failed: '+str(marker))
    if d['status']=='PASS':break
   state('WAITING_PREVIOUS_REPAIR',checkpoint=n,previous=old);time.sleep(15)
  receipt=read(prev/'repair/commit_complete.json');assert receipt['status']=='PASS' and receipt['processed_frames']==old
  run('mapping','run_online_mapping_stage.py',[scene,str(n)],n)
  run('candidate_frontend','run_candidate_frontend.py',['online',scene,str(n),'B'],n)
  run('repair_handoff','run_repair_handoff.py',['online',scene,str(n),'B'],n)
  receipt=read(O/'online'/scene/str(n)/'B/repair/commit_complete.json');assert receipt['status']=='PASS' and receipt['processed_frames']==n
 state('PASS',stage='REPAIR_200_COMMITTED',final_B_complete=False,dbscan_complete=False,evaluation_complete=False)
except Exception:
 state('FAILED',traceback=traceback.format_exc());raise
