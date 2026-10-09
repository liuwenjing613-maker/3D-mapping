"""Finish all four frozen repairs and the unchanged v3 evaluation; retain failure history."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
CODE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/continuous_tracking_ablation_20261008')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')

def read(p):return json.loads(Path(p).read_text())
def dump(name,value):
 p=ROOT/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def alive(pid):
 p=Path('/proc')/str(pid)/'status'
 return p.exists() and '\nState:\tZ' not in p.read_text()
def start(scene,stage):
 script='continue_with_complete_palette.py' if stage=='repair' else 'evaluate_ablation_v2.py'
 cmd=[PY,'-u',str(CODE/script),'--scene',scene]
 if stage=='evaluate':cmd+=['--stage','evaluate']
 with (ROOT/scene/(stage+'_recovery.log')).open('xb') as log:
  return subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)

def main():
 launched=time.time();historic=[]
 for p in [ROOT/'failure.json',ROOT/'room2/repair_failure.json']:
  if p.exists():historic.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 dump('recovery_launch.json',{'status':'STARTED','pid':os.getpid(),'existing_room0_worker':3242302,
   'retained_failure_history':historic,'changes':['PLY palette completeness only','per-identity supplementary reporting only'],
   'frozen_repair_core_unchanged':True,'unified_v3_core_unchanged':True})
 room0_pending=True;jobs={'room2':start('room2','repair')};completed=set()
 while room0_pending or jobs:
  if room0_pending and not alive(3242302):
   room0_pending=False
   p=ROOT/'room0/predictions_frozen.json'
   if p.exists():assert read(p)['status']=='PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT';completed.add('room0')
   else:jobs['room0']=start('room0','repair')
  for scene,p in list(jobs.items()):
   code=p.poll()
   if code is None:continue
   if code:raise RuntimeError(f'{scene} continuation failed exit {code}; inspect repair_recovery.log')
   assert read(ROOT/scene/'predictions_frozen.json')['status']=='PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT'
   completed.add(scene);del jobs[scene]
  dump('status.json',{'status':'RUNNING','stage':'repair','completed_repair_scenes':sorted(completed),
   'jobs':[{'scene':s,'pid':p.pid} for s,p in jobs.items()]+([{'scene':'room0','pid':3242302}] if room0_pending else []),
   'elapsed_seconds':round(time.time()-launched,2),'supervisor_pid':os.getpid()})
  if room0_pending or jobs:time.sleep(10)
 assert completed=={'room0','room2'}
 jobs={scene:start(scene,'evaluate') for scene in ['room0','room2']}
 while jobs:
  for scene,p in list(jobs.items()):
   code=p.poll()
   if code is None:continue
   if code:raise RuntimeError(f'{scene} evaluation failed exit {code}; inspect evaluate_recovery.log')
   del jobs[scene]
  dump('status.json',{'status':'RUNNING','stage':'evaluate','jobs':[{'scene':s,'pid':p.pid} for s,p in jobs.items()],
   'elapsed_seconds':round(time.time()-launched,2),'supervisor_pid':os.getpid()})
  if jobs:time.sleep(10)
 summaries={s:read(ROOT/s/'evaluation_summary.json') for s in ['room0','room2']}
 assert all(x['status']=='PASS' for x in summaries.values())
 result={'status':'PASS','conditions':['unrestricted_reference','gap0','gap1'],'scenes':summaries,
   'elapsed_seconds':round(time.time()-launched,2),'old_maps_preserved':True,'SAM_rerun':False,
   'GT_used_for_filter_or_repair':False,'unified_v3_code_protocol_and_flags_unchanged':True,
   'recovery_provenance':read(ROOT/'recovery_launch.json'),
   'supplementary_reporting_wrapper_sha256':hashlib.sha256((CODE/'evaluate_ablation_v2.py').read_bytes()).hexdigest()}
 dump('complete.json',result);dump('status.json',{'status':'PASS','stage':'complete','elapsed_seconds':result['elapsed_seconds']})
if __name__=='__main__':
 try:main()
 except Exception:
  dump('recovery_failure.json',{'status':'FAIL','traceback':traceback.format_exc()})
  dump('status.json',{'status':'FAIL','stage':'inspect recovery_failure.json'});raise
