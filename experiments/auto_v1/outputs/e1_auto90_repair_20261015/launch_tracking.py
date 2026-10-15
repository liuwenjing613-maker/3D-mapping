from pathlib import Path
import subprocess,os,json
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015');R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
assert (R/'prepare_complete.json').exists()
p=R/'queue_process.json';assert not p.exists(),'Do not launch duplicate queue'
with (R/'tracking_queue.log').open('xb') as f:
 q=subprocess.Popen(['python3','-u',str(C/'tracking_queue.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
p.write_text(json.dumps({'pid':q.pid,'status':'LAUNCHED'}));print(p.read_text())
