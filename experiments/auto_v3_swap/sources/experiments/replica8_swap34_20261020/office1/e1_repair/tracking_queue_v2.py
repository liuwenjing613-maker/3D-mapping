"""Two independent E1 workers per available GPU; safely adopt existing jobs."""
from pathlib import Path
import os,json,time,subprocess,hashlib,shutil,traceback
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020/office1/e1_repair')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def alive(job):
 p=Path('/proc')/str(job['pid'])
 if not p.exists():return False
 cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 return str(C/'track_case.py') in cmd and job['row']['config_path'] in cmd
def ram():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024**2
def gpu_state():
 txt=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[dict(index=int(a),uuid=b,used=int(c),free=int(d),util=int(e)) for a,b,c,d,e in [x.split(', ') for x in txt.strip().splitlines()] if int(a) in range(8)]
 txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 apps=[dict(pid=int(a),gpu=b,memory=int(c)) for a,b,c in [x.split(', ') for x in txt.strip().splitlines()] if a.isdigit() and c.isdigit()]
 return rows,apps
def main():
 freeze=json.loads((R/'input_freeze.json').read_text());assert sha(C/'track_case.py')==freeze['tracking_code_sha256']
 cases=json.loads((R/'seed_diagnostics.json').read_text())['cases'];byuid={x['case_uid']:x for x in cases};active={};done=[];pending=[];started=time.time();launches=[]
 for adopted in json.loads((R/'queue_v2_adoption.json').read_text())['jobs']:
  row=byuid[adopted['case_uid']];active[row['case_uid']]={'row':row,'pid':adopted['pid'],'gpu':adopted['gpu'],'adopted':True}
 for row in cases:
  uid=row['case_uid'];p=R/'cases'/uid/'tracking/complete.json'
  if p.exists():
   d=json.loads(p.read_text());assert d['status']=='PASS' and d['case_config_sha256']==sha(row['config_path']) and d['code_sha256']==sha(C/'track_case.py')
   done.append({'case_uid':uid,'seconds':d['elapsed_seconds'],'report_sha256':sha(p)});active.pop(uid,None)
  elif uid not in active:pending.append(row)
 pending.sort(key=lambda x:(-len(x['objects']),x['case_uid']))
 while pending or active:
  for uid,job in list(active.items()):
   if 'process' in job:
    if job['process'].poll() is None:continue
    assert job['process'].returncode==0,('tracking failed',uid)
   elif alive(job):continue
   p=R/'cases'/uid/'tracking/complete.json';assert p.exists(),('tracking did not complete',uid)
   d=json.loads(p.read_text());assert d['status']=='PASS' and d['completed_frames']==d['total_frames']==2000
   assert d['case_config_sha256']==sha(job['row']['config_path']) and d['directions_use_independent_seed_only_states'] and not d['GT_used']
   done.append({'case_uid':uid,'seconds':d['elapsed_seconds'],'report_sha256':sha(p),'gpu':job['gpu']});del active[uid];print('COMPLETE',uid,flush=True)
  gpus,apps=gpu_state();ourpids={j['pid'] for j in active.values()}
  for gpu in gpus:
   uuid=gpu['uuid'];ours=sum(j['gpu']==uuid for j in active.values());foreign=[x for x in apps if x['gpu']==uuid and x['pid'] not in ourpids and x['memory']>=128]
   if not pending or ours>=2 or foreign or gpu['free']<16000 or ram()<70:continue
   if not ours and (gpu['used']>=1024 or gpu['util']>=10):continue
   if not ours:
    env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
    probe=subprocess.run(['/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python','-B','-c','import torch; torch.zeros(1,device="cuda")'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)
    if probe.returncode:continue
   assert shutil.disk_usage(R).free/1024**3>=80
   row=pending.pop(0);uid=row['case_uid'];cfg=json.loads(Path(row['config_path']).read_text());assert sha(row['config_path'])==freeze['seed_and_config_hashes'][row['config_path']]
   env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(R/'runtime_cache'),'CUDA_CACHE_PATH':str(R/'cuda_cache'),'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}
   with (R/'cases'/uid/'tracking.log').open('ab') as f:p=subprocess.Popen([cfg['tracking']['python'],'-u',str(C/'track_case.py'),'--config',row['config_path']],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   active[uid]={'row':row,'pid':p.pid,'gpu':uuid,'process':p,'adopted':False};launches.append({'case_uid':uid,'pid':p.pid,'gpu':uuid,'time':time.time()});print('LAUNCH',uid,uuid,flush=True)
  save('tracking_queue_status.json',{'status':'RUNNING' if pending or active else 'PASS','queue_version':2,'maximum_independent_jobs_per_GPU':2,'completed_cases':len(done),'total_cases':len(cases),'completed':done,'pending':[x['case_uid'] for x in pending],'active':[{'case_uid':uid,'gpu':j['gpu'],'pid':j['pid']} for uid,j in active.items()],'elapsed_seconds':time.time()-started,'updated_unix':time.time()})
  save('tracking_launch_v2.json',{'queue_sha256':sha(__file__),'tasks':launches,'adoption':str(R/'queue_v2_adoption.json'),'tracking_code_unchanged':True})
  if pending or active:time.sleep(10)
 save('tracking_queue_complete.json',{'status':'PASS','cases':done,'elapsed_seconds':time.time()-started,'queue_version':2,'GT_used':False})
if __name__=='__main__':
 try:main()
 except Exception:save('tracking_queue_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
