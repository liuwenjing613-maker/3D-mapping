from pathlib import Path
import os,sys,time,json,traceback,subprocess,hashlib
import numpy as np
scene=sys.argv[1];R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online')/scene/'400';out=R/'B/evaluation_current_r2';out.mkdir(parents=True,exist_ok=True);start=time.time();main=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(p,d):p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');q.replace(p)
def state(s,**kw):put(out/'status.json',dict(status=s,scene=scene,elapsed_seconds=time.time()-start,**kw))
def wait(p):
 while True:
  d=read(p) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError(str(p)+' failed '+str(d))
  if d['status']=='PASS':return d
  state('WAITING_INPUTS',waiting_for=str(p));time.sleep(20)
def cli(args,name):
 env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with (out/name).open('ab') as f:subprocess.run([py,'-B','-m','unified_eval.cli',*map(str,args)],cwd=main,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 before_info=wait(R/'no_repair_replay/status.json');after_info=wait(R/'B/postprocess_final/status.json');assert before_info['stage']=='NO_REPAIR_FINAL_B_DBSCAN_ON_ONLINE_GEOMETRY' and after_info['stage']=='FINAL_B_AND_GUARDED_DBSCAN_COMPLETE';before=Path(before_info['surface']);after=Path(after_info['surface'])
 with np.load(before) as a,np.load(after) as b:np.testing.assert_array_equal(a['xyz_m'],b['xyz_m']);np.testing.assert_array_equal(a['rgb'],b['rgb'])
 lock=read(main/'unified_eval/configs/current_protocol.json');assert lock['profile_revision']==2 and lock['evaluable_GT_count']==350 and not lock['frozen'];assert sha(main/'unified_eval/configs'/lock['config_file'])==lock['config_sha256']
 for name,h in lock['evaluator_code_sha256'].items():assert sha(main/'unified_eval'/name)==h
 ref=read(Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/full_depth_rescreen_postprocessed/postprocess')/scene/'before/r2/adapter_manifest.json');gt=Path(ref['gt_file']);assert sha(gt)==ref['gt_file_sha256']==lock['canonical_GT'][scene]['file_sha256'];corr=out/'shared_correspondence.npz'
 for arm,path in [('before',before),('after',after)]:
  state('RUNNING',stage='R2_ADAPT_'+arm);prov=out/(arm+'_provenance.json');put(prov,{'scene':scene,'arm':arm,'baseline':'raw no-repair observations replayed on same online TSDF geometry','method':'online-v2 B; 100,200,400 causal commits','postprocessing':'final B + guarded DBSCAN','source_sha256':sha(path),'GT_used_for_repair':False})
  cli(['adapt-surface','--gt',gt,'--surface',path,'--fixed-correspondence',corr,'--source-provenance',prov,'--method-name','replica_onlineB_'+arm+'_fixed_geometry','--method-commit','fixedH_sam3_originalgate_onlineB_20261109','--out',out/arm],arm+'.log')
 a=read(out/'before/adapter_manifest.json');b=read(out/'after/adapter_manifest.json')
 for k in ['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256','correspondence_sha256']:assert a[k]==b[k],k
 state('RUNNING',stage='R2_PAIRED');cli(['eval-repair-pair','--gt',gt,'--before',out/'before/canonical_prediction.npz','--after',out/'after/canonical_prediction.npz','--out',out/'paired'],'paired.log');state('PASS',stage='FINAL_B_DBSCAN_R2_PAIRED',geometry_and_correspondence_identical=True,baseline='no-repair raw observations replayed on online TSDF geometry',GT_used_for_repair=False)
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
