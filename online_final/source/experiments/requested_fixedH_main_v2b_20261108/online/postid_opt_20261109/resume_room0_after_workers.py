from pathlib import Path
import time,subprocess,json,os
C=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109');R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109');py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
for pid in []:
 while (Path('/proc')/str(pid)/'stat').exists():
  try:
   if (Path('/proc')/str(pid)/'stat').read_text().split()[2]=='Z':break
  except FileNotFoundError:break
  time.sleep(5)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
with (R/'postid_d025_n3500/room0/controller.log').open('ab') as f:
 p=subprocess.Popen([py,'-B','-u',str(C/'controller.py'),str(C/'postid_d025_n3500'),'room0'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(5)
with (R/'eight_results/report.log').open('ab') as f:
 q=subprocess.Popen([py,'-B','-u',str(C/'report_eight.py')],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(R/'room0_oom_resume.json').write_text(json.dumps(dict(controller_pid=p.pid,report_pid=q.pid,prior_workers_waited=[])))
