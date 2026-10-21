from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys,time,traceback
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
scenes=['room0','room1','room2','office0','office1','office2','office3','office4']
def run(group,scene):
 out=R/group/scene
 while not (out/'evaluation_complete.json').exists():
  if (R/'controller_v2_failure.json').exists():raise RuntimeError('Main controller failed')
  time.sleep(10)
 dest=out/'direct_controls_r2';dest.mkdir(exist_ok=True)
 with (dest/'controller.log').open('w') as log:
  p=subprocess.run([sys.executable,'-B','-u',str(C/'score_direct_controls.py')],cwd=C,env={**os.environ,'POINT_GROUP':group,'POINT_SCENE':scene,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,(group,scene)
 return dict(group=group,scene=scene,status='PASS')
def main():
 done=[];failed=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  fs={pool.submit(run,g,s):(g,s) for s in scenes for g in ['automatic','manual']}
  for f in as_completed(fs):
   try:done.append(f.result())
   except Exception:failed.append(dict(task=fs[f],traceback=traceback.format_exc()))
   (R/'control_scores_status.json').write_text(json.dumps(dict(completed=done,failures=failed,total=16)))
 assert not failed,failed
 (R/'control_scores_complete.json').write_text(json.dumps(dict(status='PASS',completed=done)))
if __name__=='__main__':main()
