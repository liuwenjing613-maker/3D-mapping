from common100 import *
from concurrent.futures import ThreadPoolExecutor
import subprocess,traceback,threading,copy
import numpy as np
from PIL import Image
GPUS=['GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129','GPU-9a11b2b0-1816-1e33-9dd1-395cf0aaf9dd']
LOCK=threading.Lock();ROWS=[]
WRAPPER="import sitecustomize; assert sitecustomize.settings_path and sitecustomize.current_cutoff==495; import sys,runpy; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
def prior_case(case):
 for group in ['A','B']:
  if group==GROUP:continue
  root=TRACKROOT/group;p=root/'seed_diagnostics.json'
  if not p.exists():continue
  for other in read(p)['cases']:
   if other['fingerprint']!=case['fingerprint']:continue
   assert other['fingerprint_data']==case['fingerprint_data']
   complete=root/'cases'/other['case_uid']/'tracking/complete.json'
   if complete.exists() and read(complete)['status']=='PASS':return group,root,other,complete
 return None
def reuse(case,prior):
 started=time.perf_counter();group,root,other,complete=prior
 old=read(complete);cfg=read(case['config_path']);uid=case['case_uid'];out=R/'cases'/uid/'tracking';out.mkdir(exist_ok=True)
 oldids=[x['track_id'] for x in other['objects']];newids=[x['track_id'] for x in case['objects']]
 assert len(oldids)==len(newids);mapping=dict(zip(oldids,newids));rows=[]
 for row in old['frames']:
  src=Path(row['label_file']);assert digest(src)==row['label_sha256']
  dest=out/src.name
  if oldids==newids:
   if not dest.exists():os.link(src,dest)
  else:
   labels=np.asarray(Image.open(src),np.uint16);result=np.zeros(labels.shape,np.uint16)
   for a,b in mapping.items():result[labels==a]=b
   Image.fromarray(result).save(dest)
  rows.append({**row,'label_file':str(dest),'label_sha256':digest(dest),'areas':{str(b):row['areas'][str(a)] for a,b in mapping.items()}})
 seed=R/'cases'/uid/'prepared_seed.png'
 np.testing.assert_array_equal(np.asarray(Image.open(seed)),np.asarray(Image.open(out/f"f{cfg['seed_frame']:06d}.png")))
 report={**old,'case_uid':uid,'track_ids':newids,'seed_sha256':digest(seed),'case_config_sha256':digest(case['config_path']),
  'frames':rows,'elapsed_seconds':time.perf_counter()-started,'initialization_seconds':0.,'direction_seconds':{},
  'GPU_visible':'reused_no_inference','inference_reused':True,'reused_from_group':group,'reused_from_case':other['case_uid'],
  'fingerprint':case['fingerprint'],'original_inference_seconds':old['elapsed_seconds'],
  'original_direction_seconds':old.get('direction_seconds',{}),'original_report_sha256':digest(complete),
  'reuse_rule':'exact ordered binary seeds, joint grouping, RGB prefix, seed frame, model and tracking config; ID-remap preserves tie order'}
 dump(out/'complete.json',report);dump(out/'status.json',{k:v for k,v in report.items() if k!='frames'})
 return dict(case_uid=uid,seconds=time.perf_counter()-started,inference_reused=True,reused_from_group=group,reused_from_case=other['case_uid'],fingerprint=case['fingerprint'],raw_frames=496)
def worker(gpu,cases):
 for case in cases:
  uid=case['case_uid'];start=time.perf_counter();done=R/'cases'/uid/'tracking/complete.json'
  if done.exists():
   report=read(done);assert report['code_sha256']==digest(C/'track_case.py') and report['case_config_sha256']==digest(case['config_path'])
   row=dict(case_uid=uid,seconds=0.,previously_complete=True,inference_reused=report.get('inference_reused',False),GPU=gpu,raw_frames=496)
  else:
   prior=prior_case(case)
   if prior:row=reuse(case,prior)
   else:
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09',
      'CUDA_CACHE_PATH':str(TRACKROOT/'runtime_cache/cuda'),'XDG_CACHE_HOME':str(TRACKROOT/'runtime_cache'),'MPLCONFIGDIR':str(TRACKROOT/'runtime_cache/mpl')}
    with (R/'cases'/uid/'tracking.log').open('w') as log:
     q=subprocess.run([SAM,'-u','-c',WRAPPER,str(C/'track_case.py'),'--config',case['config_path']],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if q.returncode:raise RuntimeError(uid+' tracking failed; see tracking.log')
    report=read(done);report.update(inference_reused=False,fingerprint=case['fingerprint'])
    dump(done,report)
    row=dict(case_uid=uid,seconds=time.perf_counter()-start,inference_reused=False,GPU=gpu,raw_frames=496,
     inference_seconds=report['elapsed_seconds'],initialization_seconds=report['initialization_seconds'],direction_seconds=report['direction_seconds'])
  with LOCK:
   ROWS.append(row);dump(R/'tracking_queue_status.json',dict(status='RUNNING',group=GROUP,completed_cases=len(ROWS),total_cases=TOTAL,cases=ROWS))
   print(json.dumps(row),flush=True)
def main():
 global TOTAL
 started=time.perf_counter()
 frozen=read(TRACKROOT/'inputs_complete.json')
 sam_root=Path('/home/chenkejun/CVPR/repos/sam2_1/sam2')
 for rel,h in frozen['sam_source_files'].items():assert digest(sam_root/rel)==h
 for path,h in read(R/'input_freeze.json')['seed_and_config_hashes'].items():assert digest(path)==h
 cases=read(R/'seed_diagnostics.json')['cases'];TOTAL=len(cases)
 dump(R/'tracking_queue_status.json',dict(status='RUNNING',group=GROUP,completed_cases=0,total_cases=TOTAL,raw_prefix=[0,495]))
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,gpu,cases[i::2]) for i,gpu in enumerate(GPUS)]
  for fut in futures:fut.result()
 dump(R/'tracking_queue_complete.json',dict(status='PASS',group=GROUP,seconds=time.perf_counter()-started,cases=ROWS,total_cases=TOTAL,
  inferred_cases=sum(not x.get('inference_reused',False) and not x.get('previously_complete',False) for x in ROWS),
  reused_cases=sum(x.get('inference_reused',False) for x in ROWS),dense_RGB_tracking_preserved=True,model_reinitialization_preserved=True,
  raw_frames=496,mapping_frames=100,raw_cutoff=495,repair100_applied=False))
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'tracking_queue_failure.json',dict(traceback=traceback.format_exc()));raise
