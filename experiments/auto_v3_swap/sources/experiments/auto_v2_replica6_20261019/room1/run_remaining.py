"""Continue the authorized room1 experiment after the upstream run succeeds."""
from pathlib import Path
import json,subprocess,time,traceback,sys
from run_v1 import C,R,PY,ENV,gpu

def save(name,data):
 p=R/(name+'.json');t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def run(stage,args,cwd=None,device=None):
 env=dict(ENV)
 if device:env['CUDA_VISIBLE_DEVICES']=device
 with (R/(stage+'.log')).open('ab') as f:
  p=subprocess.Popen([PY,'-B','-u',*map(str,args)],cwd=cwd or C,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  save('remaining_status',dict(stage=stage,pid=p.pid,time=time.time()))
  assert p.wait()==0,stage+' failed'

def main():
 save('remaining_status',dict(stage='waiting_for_auto_v1',time=time.time()))
 while not (R/'v1_complete.json').exists():
  assert not (R/'v1_failure.json').exists(),'auto-v1 failed; see v1_failure.json'
  time.sleep(15)
 assert json.loads((R/'v1_complete.json').read_text())['status']=='PASS'
 start=sys.argv[1] if len(sys.argv)>1 else 'v2_semantic'
 assert start in ['v2_semantic','v2_measure_and_select','e1_controller']
 if start=='v2_semantic':
  run('v2_semantic',[C/'v2/semantic/run.py','--manifest',R/'v1/seed_manifest.json','--raw-config','/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room1/configs/raw.json','--out',R/'v2/semantic','--config',C/'v2/semantic/config.json'],device=gpu())
 if start in ['v2_semantic','v2_measure_and_select']:run('v2_measure_and_select',[C/'v2/measure_and_select.py'],cwd=C/'v2')
 (R/'e1_repair').mkdir(exist_ok=True)
 run('e1_controller',[C/'e1_repair/run_selected.py'],cwd=C/'e1_repair')
 save('remaining_complete',dict(status='PASS',time=time.time()))
 save('remaining_status',dict(stage='complete',time=time.time()))

if __name__=='__main__':
 try:main()
 except Exception:
  save('remaining_failure',dict(error=traceback.format_exc(),time=time.time()));raise
