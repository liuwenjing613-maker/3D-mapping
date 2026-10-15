from pathlib import Path
import subprocess,json
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015');R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
p=R/'postprocess_process.json';assert not p.exists()
with (R/'postprocess.log').open('xb') as f:q=subprocess.Popen(['python3','-u',str(C/'postprocess.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
p.write_text(json.dumps({'pid':q.pid}));print(p.read_text())
