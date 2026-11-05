import sys
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/trusted_full_online_20261104')
import write_fence
"""Track original seeds over all currently arrived RGB, preserving dense video memory."""
from common_continue import *
from concurrent.futures import ThreadPoolExecutor
from selection_runtime import environment,WRAPPER,gpu_slot
import subprocess,threading,traceback,copy,shutil
import numpy as np
from PIL import Image
LOCK=threading.Lock();ROWS=[]
def prior_case(case):
 # Only exact SAM inference inputs may be reused; association, repair and geometry are never reused here.
 archive=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/online')/SCENE/str(CHECKPOINT)
 peers=[ROOT.parent/('B' if GROUP=='A' else 'A')/'tracking']+[archive/g/'tracking' for g in ['B','A']]
 for peer in peers:
  p=peer/'seed_diagnostics.json'
  if not p.exists():continue
  for other in read(p)['cases']:
   if other['fingerprint']!=case['fingerprint']:continue
   assert other['fingerprint_data']==case['fingerprint_data']
   fp=case['fingerprint_data']
   assert fp['raw_frames']==RAW_COUNT and fp['tracking_code_sha256']==digest(C/'track_case.py')
   complete=peer/'cases'/other['case_uid']/'tracking/complete.json'
   if not complete.exists():continue
   report=read(complete)
   if report['status']!='PASS':continue
   assert report['code_sha256']==fp['tracking_code_sha256']
   assert report['checkpoint_sha256']==fp['checkpoint_sha256']
   assert report['total_frames']==report['completed_frames']==RAW_COUNT
   assert [row['frame'] for row in report['frames']]==list(range(RAW_COUNT))
   assert report['seed_frame']==fp['seed_frame'] and report['seed_mode']==fp['seed_mode']
   assert report['directions_use_independent_seed_only_states'] is True
   oldids=[x['track_id'] for x in other['objects']];newids=[x['track_id'] for x in case['objects']]
   assert oldids==sorted(oldids) and newids==sorted(newids)
   assert report['track_ids']==oldids and len(oldids)==len(newids)
   seed=np.asarray(Image.open(peer/'cases'/other['case_uid']/'prepared_seed.png'),np.uint16)
   assert set(np.unique(seed)).issubset({0,*oldids})
   assert [hashlib.sha256(np.packbits(seed==i).tobytes()).hexdigest() for i in oldids]==fp['ordered_seed_binary_sha256']
   assert all(Path(row['label_file']).resolve().parent==complete.parent.resolve() for row in report['frames'])
   return peer,other,complete
 return None
def reuse(case,prior):
 started=time.perf_counter();peer,other,complete=prior
 old=read(complete);cfg=read(case['config_path']);uid=case['case_uid'];out=R/'cases'/uid/'tracking';out.mkdir(exist_ok=True)
 oldids=[x['track_id'] for x in other['objects']];newids=[x['track_id'] for x in case['objects']]
 assert len(oldids)==len(newids);mapping=dict(zip(oldids,newids));rows=[]
 for row in old['frames']:
  src=Path(row['label_file']);assert digest(src)==row['label_sha256'];dest=out/src.name
  if oldids==newids:
   if not dest.exists():shutil.copyfile(src,dest)
   assert digest(dest)==row['label_sha256']
  else:
   labels=np.asarray(Image.open(src),np.uint16);result=np.zeros(labels.shape,np.uint16)
   for a,b in mapping.items():result[labels==a]=b
   Image.fromarray(result).save(dest)
  rows.append({**row,'label_file':str(dest),'label_sha256':digest(dest),'areas':{str(b):row['areas'][str(a)] for a,b in mapping.items()}})
 seed=R/'cases'/uid/'prepared_seed.png'
 np.testing.assert_array_equal(np.asarray(Image.open(seed)),np.asarray(Image.open(out/f"f{cfg['seed_frame']:06d}.png")))
 report={**old,'case_uid':uid,'track_ids':newids,'seed_sha256':digest(seed),'case_config_sha256':digest(case['config_path']),
  'frames':rows,'elapsed_seconds':time.perf_counter()-started,'initialization_seconds':0.,'direction_seconds':{},
  'GPU_visible':'reused_no_inference','inference_reused':True,'reused_from_group':peer.parent.name,'reused_from_case':other['case_uid'],'reused_from_root':str(peer),
  'fingerprint':case['fingerprint'],'original_inference_seconds':old['elapsed_seconds'],
  'original_direction_seconds':old.get('direction_seconds',{}),'original_report_sha256':digest(complete),
  'reuse_rule':'exact ordered seeds, joint grouping, dense RGB prefix, seed frame, model and config; ID remap preserves tie order'}
 dump(out/'complete.json',report);dump(out/'status.json',{k:v for k,v in report.items() if k!='frames'})
 return dict(case_uid=uid,seconds=time.perf_counter()-started,inference_reused=True,reused_from_group=peer.parent.name,reused_from_case=other['case_uid'],raw_frames=RAW_COUNT)
def worker(case):
 uid=case['case_uid'];start=time.perf_counter();done=R/'cases'/uid/'tracking/complete.json'
 if done.exists():
  report=read(done);assert report['code_sha256']==digest(C/'track_case.py') and report['case_config_sha256']==digest(case['config_path'])
  row=dict(case_uid=uid,seconds=0.,previously_complete=True,inference_reused=report.get('inference_reused',False),raw_frames=RAW_COUNT)
 else:
  prior=prior_case(case)
  if prior:row=reuse(case,prior)
  else:
   with gpu_slot(R) as (gpu,waiting):
    # A peer may have finished the exact same inference while waiting for a GPU.
    prior=prior_case(case)
    if prior:row=reuse(case,prior);row['gpu_wait_seconds']=waiting
    else:
     inference_start=time.perf_counter()
     with (R/'cases'/uid/'tracking.log').open('a') as log:
      q=subprocess.run([SAM,'-B','-u','-c',WRAPPER,str(C/'track_case.py'),'--config',case['config_path']],cwd=C,
        env=environment({'CUDA_VISIBLE_DEVICES':gpu}),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
     if q.returncode:raise RuntimeError(uid+' tracking failed; see tracking.log')
     report=read(done);report.update(inference_reused=False,fingerprint=case['fingerprint']);dump(done,report)
     row=dict(case_uid=uid,seconds=time.perf_counter()-inference_start,gpu_wait_seconds=waiting,inference_reused=False,GPU=gpu,raw_frames=RAW_COUNT,
       inference_seconds=report['elapsed_seconds'],initialization_seconds=report['initialization_seconds'],direction_seconds=report['direction_seconds'])
 row['wall_seconds_including_wait']=time.perf_counter()-start
 with LOCK:
  ROWS.append(row);dump(R/'tracking_queue_status.json',dict(status='RUNNING',checkpoint=CHECKPOINT,group=GROUP,completed_cases=len(ROWS),total_cases=TOTAL,cases=ROWS))
  with (R/'tracking_case_timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(row),flush=True)
def main():
 global TOTAL
 started=time.perf_counter();frozen=read(ROOT/'inputs_complete.json');samroot=Path('/home/chenkejun/CVPR/repos/sam2_1/sam2')
 for rel,h in frozen['sam_source_files'].items():assert digest(samroot/rel)==h
 for path,h in read(R/'input_freeze.json')['seed_and_config_hashes'].items():assert digest(path)==h
 cases=read(R/'seed_diagnostics.json')['cases'];TOTAL=len(cases)
 dump(R/'tracking_queue_status.json',dict(status='RUNNING',checkpoint=CHECKPOINT,group=GROUP,completed_cases=0,total_cases=TOTAL,raw_prefix=[0,CURRENT_RAW]))
 with ThreadPoolExecutor(max_workers=2) as pool:
  for future in [pool.submit(worker,c) for c in cases]:future.result()
 dump(R/'tracking_queue_complete.json',dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,seconds=time.perf_counter()-started,cases=ROWS,total_cases=TOTAL,
  inferred_cases=sum(not x.get('inference_reused',False) and not x.get('previously_complete',False) for x in ROWS),
  reused_cases=sum(x.get('inference_reused',False) for x in ROWS),dense_RGB_tracking_preserved=True,model_reinitialization_preserved=True,
  raw_frames=RAW_COUNT,mapping_frames=CHECKPOINT,raw_cutoff=CURRENT_RAW,current_checkpoint_repaired=False))
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'tracking_queue_failure.json',dict(traceback=traceback.format_exc()));raise
