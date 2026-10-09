"""Finish the existing repair workers, then use the corrected supplementary reporting wrapper."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
CODE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/continuous_tracking_ablation_20261008')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'

def read(p):return json.loads(Path(p).read_text())
def dump(name,value):
 p=ROOT/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def alive(pid):
 p=Path('/proc')/str(pid)/'status'
 return p.exists() and '\nState:\tZ' not in p.read_text()
def main():
 transition=read(ROOT/'supervisor_transition.json');workers=transition['repair_workers']
 start=(ROOT/'launch.json').stat().st_mtime
 while True:
  complete=[];waiting=[]
  for row in workers:
   scene=row['scene'];p=ROOT/scene/'predictions_frozen.json'
   if p.exists():assert read(p)['status']=='PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT';complete.append(scene)
   else:
    assert alive(row['pid']) and not (ROOT/scene/'repair_failure.json').exists(),f'{scene} repair worker failed'
    waiting.append(row)
  dump('status.json',{'status':'RUNNING','stage':'repair','jobs':waiting,'completed_repair_scenes':complete,
       'reporting_revision':'per-identity supplementary statistics only','elapsed_seconds':round(time.time()-start,2)})
  if len(complete)==2:break
  time.sleep(10)
 jobs=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
 for scene in ['room0','room2']:
  with (ROOT/scene/'evaluation_v2.log').open('xb') as log:
   p=subprocess.Popen([PY,'-u',str(CODE/'evaluate_ablation_v2.py'),'--scene',scene,'--stage','evaluate'],stdout=log,stderr=subprocess.STDOUT,env=env)
  jobs.append((scene,p))
 while jobs:
  remaining=[]
  for scene,p in jobs:
   code=p.poll()
   if code is None:remaining.append((scene,p))
   elif code:raise RuntimeError(f'{scene} evaluation exit {code}; inspect evaluation_v2.log')
  dump('status.json',{'status':'RUNNING','stage':'evaluate','jobs':[{'scene':s,'pid':p.pid} for s,p in remaining],
       'completed_repair_scenes':['room0','room2'],'elapsed_seconds':round(time.time()-start,2)})
  jobs=remaining
  if jobs:time.sleep(10)
 summaries={s:read(ROOT/s/'evaluation_summary.json') for s in ['room0','room2']}
 assert all(x['status']=='PASS' for x in summaries.values())
 result={'status':'PASS','conditions':['unrestricted_reference','gap0','gap1'],'scenes':summaries,
    'elapsed_seconds':round(time.time()-start,2),'old_maps_preserved':True,'SAM_rerun':False,
    'GT_used_for_filter_or_repair':False,'unified_v3_code_protocol_and_flags_unchanged':True,
    'supplementary_reporting_revision':transition,
    'supplementary_reporting_wrapper_sha256':hashlib.sha256((CODE/'evaluate_ablation_v2.py').read_bytes()).hexdigest()}
 dump('complete.json',result);dump('status.json',{'status':'PASS','stage':'complete','elapsed_seconds':result['elapsed_seconds']})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'status':'FAIL','traceback':traceback.format_exc()});dump('status.json',{'status':'FAIL','stage':'inspect failure.json'});raise
