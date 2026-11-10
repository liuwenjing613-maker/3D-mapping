from pathlib import Path
import os,sys,json,time,subprocess,traceback
root=Path(sys.argv[1]);branch=Path(sys.argv[2]);status=root/'actual_repair_driver_status.json';started=time.time();n=int(root.parts[-2]);g=root.parts[-1]
def write(x):q=status.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(status)
def run(stage,cmd):
 t=time.time();write({'status':'RUNNING','stage':stage,'started_unix':t,'elapsed_seconds':t-started,'repair_complete':False});env={**os.environ,'CVPR_CHECKPOINT':str(n),'CVPR_TRACK_GROUP':g,'CVPR_SCENE':'office2','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with (root/(stage+'_supervisor.log')).open('ab') as f:rc=subprocess.call(cmd,cwd=branch,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 with (root/'actual_repair_timings.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'start':t,'end':time.time(),'seconds':time.time()-t,'returncode':rc})+'\n')
 if rc:raise RuntimeError(stage+' failed: '+str(root/(stage+'_supervisor.log')))
try:
 while True:
  p=root/'gate_driver_status.json';x=json.loads(p.read_text()) if p.exists() else {'status':'WAITING'}
  if x['status']=='FAILED':raise RuntimeError('original gate failed: '+str(x))
  if x['status']=='PASS':break
  write({'status':'WAITING_GATE','elapsed_seconds':time.time()-started});time.sleep(15)
 py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python';code=branch/'package.py'
 run('package',[py,'-B','-u',str(code),str(root)]);assert json.loads((root/'package_complete.json').read_text())['status']=='PASS_GATE_INPUT_PACKAGE_ONLY'
 run('repair',[py,'-B','-u',str(branch/'repair_continue.py')]);r=json.loads((root/'repair/repair_complete.json').read_text());assert r['status']=='PASS' and r['mapping_frames']==n and not r['GT_used']
 run('commit',[py,'-B','-u',str(branch/'commit_continue.py')]);c=json.loads((root/'repair/commit_complete.json').read_text());assert c['status']=='PASS' and c['processed_frames']==n
 write({'status':'PASS','stage':'REAL_REPAIR_AND_COMMIT','elapsed_seconds':time.time()-started,'GT_used':False,'model':'SAM3','original_mask_gate':True,'final_B_complete':False,'DBSCAN_complete':False,'evaluation_complete':False,'commit':str(root/'repair/commit_complete.json')})
except BaseException:write({'status':'FAILED','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started});raise
