from common100 import *
import subprocess,traceback
WRAPPER="import sitecustomize; assert sitecustomize.settings_path and sitecustomize.current_cutoff==495; import sys,runpy; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
STATE=dict(status='RUNNING',stage='waiting_candidates',message='准备100帧两组原种子双向追踪，只评估门控，不执行第二次修复。',groups={})
def publish():
 STATE['unix']=time.time();dump(TOP/'review/tracking100_status.json',STATE)
def run(group,name):
 root=TRACKROOT/group;root.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();env={**os.environ,'CVPR_TRACK_GROUP':group,'PYTHONPATH':str(C/'guard')+':'+str(C),
 'CVPR_ONLINE_GUARD':str(TRACKROOT/'input_guard.json'),'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
 assert Path(env['CVPR_ONLINE_GUARD']).exists()
 with (root/(name+'.log')).open('a') as log:
  p=subprocess.Popen([PY,'-u','-c',WRAPPER,str(C/(name+'.py'))],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  STATE['stage']=name;STATE['groups'][group]={'stage':name,'pid':p.pid,'status':'RUNNING'}
  while p.poll() is None:
   if (root/'tracking_queue_status.json').exists() and name=='queue_group':
    q=read(root/'tracking_queue_status.json');STATE['groups'][group].update(completed_cases=q['completed_cases'],total_cases=q['total_cases'])
    STATE['message']=f"{group}组追踪 {q['completed_cases']}/{q['total_cases']} 案例完成；范围 raw0..495，第二次修复未启动。"
   elif name=='gate100' and (root/'progress.json').exists():
    q=read(root/'progress.json');STATE['groups'][group].update({k:v for k,v in q.items() if k in ['processed_frames','total_frames','gate_seconds']})
    STATE['message']=f"{group}组正在原门控只读检查，第二次修复未启动。"
   else:STATE['message']=f"{group}组正在{name}，第二次修复未启动。"
   STATE['groups'][group]['stage_elapsed_seconds']=time.perf_counter()-start;publish();time.sleep(3)
  if p.returncode:raise RuntimeError(group+' '+name+' failed')
 return time.perf_counter()-start
def main():
 started=time.perf_counter();publish();all_results={}
 for group in ['A','B']:
  ready=TOP/'review'/('statusA.json' if group=='A' else 'statusB.json')
  while not ready.exists() or read(ready).get('status')!='PASS':
   STATE['stage']='waiting_candidates';STATE['message']=f"{group}组等待候选独立QA通过；已经完成的组保留结果，第二次修复未启动。";publish();time.sleep(10)
  while not (TRACKROOT/'gate50_equivalence.json').exists():time.sleep(3)
  assert read(TRACKROOT/'gate50_equivalence.json')['status']=='PASS'
  timings={}
  for name in ['prepare_group','queue_group','gate100','render_group']:
   timings[name]=run(group,name)
  root=TRACKROOT/group
  q=read(root/'tracking_queue_complete.json');g=read(root/'gate_complete.json')
  result=dict(status='PASS',stage='complete',group=group,cases=g['cases'],objects=g['objects'],timings=timings,tracking_seconds=q['seconds'],
   inferred_cases=q['inferred_cases'],reused_cases=q['reused_cases'],gate_seconds=g['gate_seconds'],
   page_url=f'http://127.0.0.1:8798/online-repair/tracking100/{group}/',summary_path=str(root/'gate_complete.json'),
   raw_prefix=[0,495],repair100_applied=False)
  dump(root/'complete.json',result);all_results[group]=result;STATE['groups'][group]=result;publish()
 STATE.update(status='PASS',stage='complete',message='A/B两组追踪与原门控只读检查完成，第二次修复未执行。',seconds=time.perf_counter()-started)
 dump(TRACKROOT/'complete.json',dict(status='PASS',groups=all_results,seconds=STATE['seconds'],repair100_applied=False))
 publish()
if __name__=='__main__':
 try:main()
 except Exception:
  STATE.update(status='FAILED',stage='failure',message='追踪对照遇到错误，已停止；未执行第二次修复。',traceback=traceback.format_exc());publish()
  dump(TRACKROOT/'failure.json',STATE);raise
