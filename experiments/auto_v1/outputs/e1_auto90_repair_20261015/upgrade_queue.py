"""Replace only our queue coordinator; leave every running tracker alive."""
from pathlib import Path
import os,signal,json,subprocess,shutil,time
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015');R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
assert not (R/'queue_v2_adoption.json').exists()
old=json.loads((R/'queue_process.json').read_text());pid=old['pid'];proc=Path('/proc')/str(pid)
cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert cmd.strip()=='python3 -u '+str(C/'tracking_queue.py'),cmd
os.kill(pid,signal.SIGSTOP)
try:
 cases=json.loads((R/'seed_diagnostics.json').read_text())['cases'];bycfg={x['config_path']:x for x in cases}
 jobs=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   status=(p/'status').read_text();ppid=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
   if ppid!=pid:continue
   parts=(p/'cmdline').read_bytes().split(b'\0');args=[x.decode() for x in parts if x]
   if str(C/'track_case.py') not in args:continue
   cfg=args[args.index('--config')+1];assert cfg in bycfg
   env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
   jobs.append({'pid':int(p.name),'case_uid':bycfg[cfg]['case_uid'],'gpu':env[b'CUDA_VISIBLE_DEVICES'].decode(),'config_path':cfg})
  except FileNotFoundError:continue
 shutil.copyfile(R/'queue_process.json',R/'queue_process_v1.json')
 shutil.copyfile(R/'tracking_queue_status.json',R/'tracking_queue_status_v1_final.json')
 (R/'queue_v2_adoption.json').write_text(json.dumps({'old_queue_pid':pid,'old_queue_stopped_at':time.time(),'jobs':jobs,'inference_processes_untouched':True},indent=2))
 with (R/'tracking_queue_v2.log').open('xb') as f:q=subprocess.Popen(['python3','-u',str(C/'tracking_queue_v2.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (R/'queue_process.json').write_text(json.dumps({'pid':q.pid,'queue_version':2,'adopted_jobs':jobs}))
 # Terminate the verified coordinator only. Its independently launched children keep running.
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 print(json.dumps({'queue_pid':q.pid,'adopted_jobs':jobs,'max_per_GPU':2}))
except BaseException:
 os.kill(pid,signal.SIGCONT)
 raise
