from pathlib import Path
import os,json,subprocess,time,traceback
UI=Path(__file__).resolve().parent;C=UI.parent;R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015');PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def save(n,v):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 completed=[]
 while not (R/'pipeline_complete.json').exists():
  if (R/'pipeline_failure.json').exists():raise RuntimeError('mapping pipeline failed')
  save('postprocess_status.json',{'stage':'waiting_for_frozen_predictions','status':'RUNNING'});time.sleep(15)
 for stage,script in [('evaluation_r2',C/'evaluate_r2.py'),('conflict_assets',UI/'prepare_conflict_assets.py'),('review',UI/'build_review.py')]:
  if stage=='review':
   while json.loads((R/'review_assets_status.json').read_text())['status']!='PASS':time.sleep(15)
  start=time.time();log=R/(stage+'.log')
  with log.open('xb') as f:
   p=subprocess.Popen([PY,'-u',str(script)],env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   save('postprocess_status.json',{'status':'RUNNING','stage':stage,'pid':p.pid,'completed':completed})
   assert p.wait()==0,stage+' failed: '+str(log)
  completed.append({'stage':stage,'seconds':time.time()-start})
 save('postprocess_complete.json',{'status':'PASS','stages':completed});save('postprocess_status.json',{'status':'PASS','stage':'complete','stages':completed})
if __name__=='__main__':
 try:main()
 except Exception:save('postprocess_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
