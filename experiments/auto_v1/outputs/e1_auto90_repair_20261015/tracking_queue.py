"""Resumable E1 tracking queue; no source data changes and no GPU process eviction."""
from pathlib import Path
import os,json,time,subprocess,shutil,hashlib,traceback
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def ram():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
def resources():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [{'index':int(a),'uuid':b,'memory':int(c),'util':int(d)} for a,b,c,d in [x.split(', ') for x in text.strip().splitlines()] if int(a) in (0,1,2)]
def main():
 freeze=json.loads((R/'input_freeze.json').read_text());assert sha(C/'track_case.py')==freeze['tracking_code_sha256']
 assert json.loads((R/'prepare_complete.json').read_text())['status']=='PASS'
 cases=json.loads((R/'seed_diagnostics.json').read_text())['cases'];pending=[];done=[];active={};launches=[];start=time.time()
 for row in cases:
  p=R/'cases'/row['case_uid']/'tracking/complete.json'
  if p.exists():
   d=json.loads(p.read_text());assert d['status']=='PASS' and d['case_config_sha256']==sha(row['config_path']) and d['code_sha256']==sha(C/'track_case.py')
   done.append({'case_uid':row['case_uid'],'report_sha256':sha(p),'seconds':d['elapsed_seconds']})
  else:pending.append(row)
 # Largest-object batches first, deterministic; every approved batch will run.
 pending.sort(key=lambda x:(-len(x['objects']),x['case_uid']))
 while pending or active:
  for gpu,job in list(active.items()):
   code=job['process'].poll()
   if code is None:continue
   uid=job['row']['case_uid'];p=R/'cases'/uid/'tracking/complete.json'
   assert code==0 and p.exists(),('tracking failed',uid,job['log'])
   d=json.loads(p.read_text());assert d['status']=='PASS' and d['completed_frames']==d['total_frames']==2000
   assert d['case_config_sha256']==sha(job['row']['config_path']) and d['directions_use_independent_seed_only_states'] and not d['GT_used']
   done.append({'case_uid':uid,'gpu':gpu,'report_sha256':sha(p),'seconds':d['elapsed_seconds']});del active[gpu]
   print(json.dumps({'event':'COMPLETE',**done[-1]}),flush=True)
  for gpu in resources():
   uidgpu=gpu['uuid']
   if not pending or uidgpu in active or gpu['memory']>=1024 or gpu['util']>=10 or ram()<65:continue
   assert shutil.disk_usage(R).free/1024**3>80
   row=pending.pop(0);cfg=json.loads(Path(row['config_path']).read_text());assert sha(row['config_path'])==freeze['seed_and_config_hashes'][row['config_path']]
   env={**os.environ,'CUDA_VISIBLE_DEVICES':uidgpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(R/'runtime_cache'),'CUDA_CACHE_PATH':str(R/'cuda_cache'),'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
   log=R/'cases'/row['case_uid']/'tracking.log'
   with log.open('ab') as f:p=subprocess.Popen([cfg['tracking']['python'],'-u',str(C/'track_case.py'),'--config',row['config_path']],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   active[uidgpu]={'process':p,'row':row,'log':str(log)};launches.append({'case_uid':row['case_uid'],'pid':p.pid,'gpu':uidgpu,'time':time.time()})
   print(json.dumps({'event':'LAUNCH',**launches[-1]}),flush=True)
  status={'status':'RUNNING' if pending or active else 'PASS','completed_cases':len(done),'total_cases':len(cases),'pending':[r['case_uid'] for r in pending],
   'active':[{'case_uid':j['row']['case_uid'],'gpu':g,'pid':j['process'].pid} for g,j in active.items()],'completed':done,'elapsed_seconds':time.time()-start,'updated_unix':time.time()}
  save('tracking_queue_status.json',status);save('tracking_launch.json',{'tasks':launches,'queue_sha256':sha(__file__),'excluded_GPU3':'CUDA initialization failed in preflight','GT_used':False})
  if pending or active:time.sleep(10)
 save('tracking_queue_complete.json',{'status':'PASS','cases':done,'elapsed_seconds':time.time()-start,'GT_used':False})
if __name__=='__main__':
 try:main()
 except Exception:save('tracking_queue_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
