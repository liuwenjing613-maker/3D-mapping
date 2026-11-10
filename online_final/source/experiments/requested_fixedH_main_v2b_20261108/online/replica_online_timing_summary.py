from pathlib import Path
import os,json,time,traceback
R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online');out=R/'timing_final';out.mkdir(exist_ok=True);start=time.time();scenes=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def put(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
try:
 while True:
  p=R/'pooled_final_r2/status.json';d=read(p) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError('pooled eval failed: '+str(d))
  if d['status']=='PASS':break
  put(out/'status.json',{'status':'WAITING_FINAL_R2','elapsed_seconds':time.time()-start});time.sleep(30)
 all_intervals=[];records={}
 for scene in scenes:
  cps={};beg=None
  for cp in [100,200,400]:
   root=R/scene/str(cp)/'B';m=read(root/'mapping/mapping_timing.json');beg=m['started_unix'] if beg is None else min(beg,m['started_unix']);track=read(root/'tracking_sam3/driver_status.json');gate=read(root/'gate_sam3/status.json');post=root/'postprocess_final/status.json';event=[]
   for f in (root/'candidates').rglob('process_timing_events.jsonl'):
    for line in f.read_text().splitlines():
     x=json.loads(line)
     if x.get('status')=='PASS':
      row={'stage':x.get('stage'),'seconds':x.get('seconds'),'started_unix':x.get('started_unix'),'ended_unix':x.get('ended_unix'),'gpu':x.get('gpu'),'source':str(f)};event.append(row)
      if x.get('gpu') and x.get('started_unix') and x.get('ended_unix'):all_intervals.append({'scene':scene,'checkpoint':cp,'stage':'candidate_'+str(x.get('stage')),'gpu':x['gpu'],'start':x['started_unix'],'end':x['ended_unix']})
   seed=[];gp=root/'tracking_sam3/gpu_events.jsonl'
   for line in gp.read_text().splitlines() if gp.exists() else []:
    x=json.loads(line);p=root/'tracking_sam3'/x['mask']/'status.json'
    if p.exists():
     a=read(p)
     if a.get('status')=='PASS':
      row={'mask':x['mask'],'gpu_index':x['gpu'],'gpu_uuid':a.get('gpu'),'started_unix':x['unix'],'elapsed_seconds':a['elapsed_seconds']};seed.append(row);all_intervals.append({'scene':scene,'checkpoint':cp,'stage':'sam3_tracking','gpu':a.get('gpu',str(x['gpu'])),'start':x['unix'],'end':x['unix']+a['elapsed_seconds']})
   timings=[json.loads(x) for x in (root/'actual_repair_timings.jsonl').read_text().splitlines()];assert read(root/'actual_repair_driver_status.json')['status']=='PASS'
   rec={'mapping':m,'candidate_process_events':event,'tracking_scheduler_wall_seconds_including_queue':track.get('elapsed_seconds'),'tracking_seed_gpu_seconds_sum':sum(x['elapsed_seconds'] for x in seed),'tracking_seeds':seed,'gate_wall_seconds':gate.get('elapsed_seconds'),'repair_and_commit_attempts':timings}
   for f,k in [('candidate_prepare_status.json','candidate_prepare'),('candidate_chain_status.json','candidate_chain'),('candidates/background070/status.json','background'),('fixedH_driver_status.json','fixedH')]:
    p=root/f
    if p.exists():rec[k]=read(p)
   if cp==400:
    assert read(post)['status']=='PASS';commit=root/'repair/commit_complete.json';evalstatus=root/'evaluation_current_r2/status.json';assert read(evalstatus)['status']=='PASS';rec['postprocess_wall_approx_seconds_from_commit_to_completion']=max(0,post.stat().st_mtime-commit.stat().st_mtime);rec['evaluation_end_unix']=evalstatus.stat().st_mtime;rec['end_to_end_wall_seconds_including_waits']=max(0,evalstatus.stat().st_mtime-beg);rec['baseline_replay']=read(R/scene/'400/no_repair_replay/status.json')
   cps[str(cp)]=rec
  records[scene]={'checkpoints':cps}
 points=[]
 for r in all_intervals:points.extend([(r['start'],1),(r['end'],-1)])
 current=maximum=0
 for _,delta in sorted(points,key=lambda x:(x[0],x[1])):current+=delta;maximum=max(maximum,current)
 put(out/'summary.json',{'status':'PASS','scope':'Replica online-v2 B 100/200/400; timing evidence is from actual processes; excludes computation reused before this run','notes':['端到端为首次建图开始到评价完成的墙钟时间，包含排队、人工检查及等待；阶段存在并行，不可相加','GPU 并发只覆盖有明确起止记录的候选模型与 SAM3 追踪，可能低估其他 GPU 任务','后处理时间按提交与完成文件时间估算；共享服务器及缓存影响耗时'],'per_scene':records,'measured_gpu_intervals':all_intervals,'measured_peak_overlapping_gpu_tasks':maximum});put(out/'status.json',{'status':'PASS','stage':'TIMING_SUMMARY_COMPLETE','elapsed_seconds':time.time()-start})
except BaseException:put(out/'status.json',{'status':'FAILED','traceback':traceback.format_exc()});raise
