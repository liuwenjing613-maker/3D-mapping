from pathlib import Path
import os,sys,json,time,traceback,subprocess,hashlib
C=Path(__file__).parent
O=Path('/data/chenkejun/CVPR/results/scannet0518_threshold_transfer_20261110')
ORIG=O
R=O/'scannet400/main/scene0518_00'
py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
start=time.time();status=O/'main_run_status.json'
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def state(s,**kw):dump(status,dict(status=s,mode='main offline single repair400',filter_version='online-v2-B-threshold-transfer-v1/split_new3500_inherit100',elapsed_seconds=time.time()-start,**kw))
def passed(p):return p.exists() and read(p).get('status')=='PASS'
def run(n,name,args,marker=None,cwd=None,extra=None,launch=None):
 root=R/str(n)/'B'
 if marker and passed(marker):return
 state('RUNNING',checkpoint=n,stage=name)
 t=time.time()
 with (root/('transfer_'+name+'.log')).open('ab',buffering=0) as f:
  p=subprocess.Popen([py,'-B','-u',*map(str,args)],cwd=cwd or C,env={**env,**(extra or {})},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if launch:dump(launch,{'pid':p.pid,'started':t,'stage':name})
  rc=p.wait()
 with (O/'timings.jsonl').open('a') as f:f.write(json.dumps({'checkpoint':n,'stage':name,'pid':p.pid,'start':t,'end':time.time(),'exit':rc})+'\n')
 if rc:raise RuntimeError(f'{n}/{name} failed exit {rc}; see {root}/transfer_{name}.log')
 if marker:assert passed(marker),(name,str(marker),read(marker))
def prepare(n):
 root=R/str(n)/'B';code=C/'scannet400_frontend/main/scene0518_00'/str(n)/'B';b=root/'mapping';cand=root/'candidates';cand.mkdir(exist_ok=True)
 m=read(b/'mapping_complete.json');assert m['frame_count']==n and not m['GT_used'] and not m['future_inputs_used']
 for name in ['baseline','geometry','association','inputs','mapping_complete.json']:
  dst=cand/name
  if not dst.exists():dst.symlink_to((b/name).resolve(),target_is_directory=(b/name).is_dir())
 (code/'config.json').write_text((b/'config.json').read_text())
 rows=[x for x in read(ORIG/'scannet400_aligned_input/selection.json')['rows'] if x['frame']<n*5];assert len(rows)==n
 dump(cand/'selection_prefix.json',{'rows':rows})
 return code
def reuse_tracking(n,root):
 out=root/'tracking_sam3';out.mkdir(exist_ok=True)
 if passed(out/'driver_status.json'):return
 sel=read(root/'candidates/fixedH/result.json');source=ORIG/'scannet400/online/scene0518_00'/str(n)/'B/tracking_sam3'
 cache={}
 for p in source.glob('*/status.json'):
  s=read(p)
  if s.get('status')=='PASS' and s.get('frame_count')==n:cache[(s['seed_sha256'],s['seed_frame'])]=p.parent
 reused=[]
 for r in sel['selected_records']:
  p=out/r['mask_id'];src=cache.get((r['mask_sha256'],r['frame']))
  if src and not p.exists():
   p.symlink_to(src.resolve(),target_is_directory=True);reused.append({'mask_id':r['mask_id'],'source':str(src),'seed_sha256':r['mask_sha256'],'seed_frame':r['frame'],'frames':n})
 dump(root/'tracking_reuse.json',{'exact_seed_and_prefix_only':True,'same_original_RGB_sequence':True,'reused':reused})
try:
 for n in [400]:
  root=R/str(n)/'B';code=C/'scannet400_frontend/main/scene0518_00'/str(n)/'B'
  root.mkdir(parents=True,exist_ok=True)
  while not passed(root/'mapping_driver_status.json'):
   p=root/'mapping_driver_status.json'
   if p.exists() and read(p).get('status')=='FAILED':raise RuntimeError(str(read(p)))
   state('WAITING_MAPPING400');time.sleep(20)
  code=prepare(n)
  if not (root/'candidates/catalog/scene0518_00/candidate_manifest.json').exists():
   run(n,'catalog',[code/'catalog/build_candidates.py','scene0518_00'],cwd=code/'catalog',launch=root/'catalog_build_launch.json')
  front=code/'front_driver.py'
  if not front.exists():front=C/'scannet400_frontend/front_driver.py'
  run(n,'frontend',[front,'main',n,'B'],root/'candidate_frontend_status.json')
  run(n,'upstream',[C/'scannet400_frontend/upstream_worker.py','main',n,'B'],root/'approved_upstream_status.json')
  run(n,'medoids',[C/'scannet_medoids_worker.py',root],root/'candidates/medoids/status.json')
  run(n,'semantic',[C/'scannet400_frontend/semantic_launcher.py',root],root/'semantic_driver_status.json')
  run(n,'visibility',[C/'scannet400_frontend/visibility_worker.py',root],root/'candidates/visibility/status.json')
  run(n,'background_thresholds',[C/'reclassify_background.py',root],root/'candidates/background_tuned/status.json')
  run(n,'screening',[code/'fixedH_geometry.py'],root/'candidates/fixedH/status.json')
  sel=read(root/'candidates/fixedH/result.json')
  assert all(0<=r['frame']<n*5 for r in sel['selected_records'])
  assert sel['threshold_tuning']==read(ORIG/'threshold_tuning_20261109/best_config.json')['parameters']
  dump(root/'fixedH_driver_status.json',{'status':'PASS','stage':'SCREENING_ONLY_NOT_TRACKING_OR_REPAIR','filter_version':'online-v2-B-threshold-transfer-v1/split_new3500_inherit100'})
  # No cross-scene tracking reuse.
  run(n,'tracking',[C/'scannet400_frontend/sam3_scheduler.py',root],root/'tracking_sam3/driver_status.json')
  run(n,'gate',[C/'scannet_gate_wait.py',root],root/'gate_driver_status.json')
  run(n,'repair_commit',[C/'scannet_repair_supervisor.py',root,C/'scannet_repair_runtime/scannet400/main/scene0518_00'],root/'actual_repair_driver_status.json')
  cm=read(root/'repair/commit_complete.json');assert cm['processed_frames']==n
  dump(root/'threshold_transfer_status.json',{'status':'PASS','stage':'CAUSAL_ACTUAL_REPAIR_AND_COMMIT','checkpoint':n,'selected':len(sel['selected_records']),'new_ID_candidates':sum(r['id_action']=='new' for r in sel['selected_records']),'GT_used':False,'final_evaluation_complete':False})
 root=R/'400/B'
 run(400,'postprocess',[C/'scannet_offline_postprocess.py',root,C/'scannet_repair_runtime/scannet400/main/scene0518_00'],root/'postprocess_final/status.json',extra={'CVPR_CHECKPOINT':'400','CVPR_TRACK_GROUP':'B','CVPR_SCENE':'scene0518_00'})
 state('PASS',stage='MAIN400_REAL_REPAIR_POSTPROCESS_COMPLETE_EVALUATION_PENDING',GT_used_for_repair=False,evaluation_complete=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
