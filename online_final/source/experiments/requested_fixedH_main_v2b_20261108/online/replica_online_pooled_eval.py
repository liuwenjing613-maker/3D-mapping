from pathlib import Path
import os,time,json,subprocess,traceback
R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online');out=R/'pooled_final_r2';out.mkdir(exist_ok=True);main=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python';scenes=['room0','room1','room2','office0','office1','office2','office3','office4'];start=time.time()
def read(p):return json.loads(Path(p).read_text())
def put(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');q.replace(p)
try:
 while True:
  statuses=[]
  for s in scenes:
   p=R/s/'400/B/evaluation_current_r2/status.json';d=read(p) if p.exists() else {'status':'WAITING'};statuses.append((s,d))
  if any(x['status']=='FAILED' for _,x in statuses):raise RuntimeError('scene eval failed '+str([(s,x) for s,x in statuses if x['status']=='FAILED']))
  if all(x['status']=='PASS' for _,x in statuses):break
  put(out/'status.json',{'status':'WAITING_SCENES','done':sum(x['status']=='PASS' for _,x in statuses),'total':8,'elapsed_seconds':time.time()-start});time.sleep(30)
 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None);results={}
 for arm in ['before','after']:
  rows=[]
  for s in scenes:
   p=R/s/'400/B/evaluation_current_r2'/arm;a=read(p/'adapter_manifest.json');rows.append({'gt':a['gt_file'],'prediction':str(p/'canonical_prediction.npz'),'diagnostic_prediction':str(p/'diagnostic_support_prediction.npz')})
  dest=out/arm;dest.mkdir(exist_ok=True);put(dest/'batch.json',{'scenes':rows})
  with (dest/'evaluation.log').open('ab') as f:subprocess.run([py,'-B','-m','unified_eval.cli','eval-batch','--scenes',str(dest/'batch.json'),'--out',str(dest)],cwd=main,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
  x=read(dest/'summary.json');assert x['CA_PRF1_0_5']['TP']+x['CA_PRF1_0_5']['FN']==350;results[arm]={'F1':x['CA_PRF1_0_5']['F1'],'PQ':x['CA_PQ']['PQ'],'AP':x['CA_AP_uniform'],'AP50':x['CA_AP50_uniform'],'mCov':x['CA_mCov'],'TP':x['CA_PRF1_0_5']['TP'],'FP':x['CA_PRF1_0_5']['FP'],'FN':x['CA_PRF1_0_5']['FN']}
 put(out/'comparison.json',{'status':'PASS','scenes':8,'GT':350,'scope':'online-v2 B 100/200/400 vs raw no-repair replay on identical online TSDF geometry; final B + guarded DBSCAN; current main R2','results':results});put(out/'status.json',{'status':'PASS','stage':'ALL_8_PAIRED_R2','elapsed_seconds':time.time()-start})
except BaseException:put(out/'status.json',{'status':'FAILED','traceback':traceback.format_exc()});raise
