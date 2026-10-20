from pathlib import Path
import os,json,subprocess,time,traceback
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office4/e1_repair')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
def save(name,v):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(name,args):
 save('run_status.json',{'stage':name,'status':'RUNNING','time':time.time()})
 with (R/(name+'.controller.log')).open('xb') as f:
  p=subprocess.Popen([PY,'-u',*map(str,args)],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  assert p.wait()==0,name+' failed'
def main():
 run('selection',[C/'prepare_selection.py'])
 run('prepare',[C/'prepare_auto_seeds.py'])
 run('tests',[C/'run_stage.py','tests'])
 run('input_audit',[C/'audit_selected_inputs.py'])


 save('queue_v2_adoption.json',{'jobs':[]})
 with (R/'tracking_queue.controller.log').open('xb') as f:
  q=subprocess.Popen([PY,'-u',str(C/'tracking_queue_v2.py')],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  save('tracking_queue_process.json',{'pid':q.pid})
  run('pipeline',[C/'pipeline.py'])
  assert q.wait()==0
 run('evaluation',[C/'evaluate_r2.py'])
 save('run_complete.json',{'status':'PASS','time':time.time(),'selected_seeds':len(json.loads((C/'confirmed_selection.json').read_text())['included']),'excluded_seeds':len(json.loads((C/'confirmed_selection.json').read_text())['excluded']),'variant':'original_E1'})
 save('run_status.json',{'status':'PASS','stage':'complete','time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:save('run_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
