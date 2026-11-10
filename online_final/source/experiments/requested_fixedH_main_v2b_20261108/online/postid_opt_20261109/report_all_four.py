from pathlib import Path
import sys,os,time,json,traceback,subprocess
C=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
O=R.parent
main=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
scenes=['office0','office1','room2','office3'];variants=['postid_d025_n3500','postid_d050_n500','postid_early025_n3500_late050_n350','postid_early050_n500_late050_n350'];start=time.time();out=R/'pilot4_all_results';out.mkdir(exist_ok=True)
def rd(p):return json.loads(p.read_text())
def wr(p,d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def state(s,**kw):wr(out/'status.json',{'status':s,'elapsed_seconds':time.time()-start,**kw})
try:
 while True:
  ss=[(v,s,rd(R/v/s/'optimization_status.json') if (R/v/s/'optimization_status.json').exists() else {'status':'WAITING'}) for v in variants for s in scenes]
  if any(d['status']=='FAILED' for _,_,d in ss):raise RuntimeError(str([(v,s,d) for v,s,d in ss if d['status']=='FAILED']))
  done=sum(d['status']=='PASS' for _,_,d in ss)
  if done==16:break
  state('WAITING_REAL_ONLINE_FINAL',complete=done,total=16);time.sleep(30)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 results={}
 for arm in ['no_repair','original_fixedH_100_200_400']+variants:
  entries=[]
  for s in scenes:
   root=(O/s/'400/B/evaluation_current_r2'/('before' if arm=='no_repair' else 'after')) if arm in ['no_repair','original_fixedH_100_200_400'] else R/arm/s/'400/B/evaluation_current_r2/after'
   a=rd(root/'adapter_manifest.json');ref=rd(O/s/'400/B/evaluation_current_r2/before/adapter_manifest.json')
   for k in ['gt_file_sha256','gt_scope_sha256','geometry_xyz_sha256','correspondence_sha256']:assert a[k]==ref[k],(arm,s,k)
   entries.append({'gt':a['gt_file'],'prediction':str(root/'canonical_prediction.npz'),'diagnostic_prediction':str(root/'diagnostic_support_prediction.npz')})
  dest=out/arm;dest.mkdir(exist_ok=True);wr(dest/'batch.json',{'scenes':entries});state('EVALUATING_POOLED',arm=arm)
  with (dest/'evaluation.log').open('ab') as log:subprocess.run([py,'-B','-m','unified_eval.cli','eval-batch','--scenes',str(dest/'batch.json'),'--out',str(dest)],cwd=main,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
  d=rd(dest/'summary.json');assert d['CA_PRF1_0_5']['TP']+d['CA_PRF1_0_5']['FN']==164
  results[arm]={'F1':d['CA_PRF1_0_5']['F1'],'AP50':d['CA_AP50_uniform'],'AP':d['CA_AP_uniform'],'PQ':d['CA_PQ']['PQ'],'mCov':d['CA_mCov'],**{k:d['CA_PRF1_0_5'][k] for k in ['TP','FP','FN']}}
 wr(out/'comparison.json',{'status':'PASS','scenes':scenes,'GT':164,'results':results,'new_schedule':[100,300,400],'original_schedule':[100,200,400],'changes_include_schedule_and_filter':True,'GT_used_in_repair':False,'filter':'post-sameview identity assignment; dice025/new3500; dice050/new500; early025/new3500 then late050/new350; early050/new500 then late050/new350','mode':'actual online repair/commit, finalB, guardedDBSCAN, locked main R2','geometry':'same certified native prefix geometry; not PPT full-depth','default_method_updated':False})
 state('PASS',stage='FOUR_SCENE_FOUR_VARIANT_ACTUAL_ONLINE_R2')
except BaseException:state('FAILED',traceback=traceback.format_exc());raise
