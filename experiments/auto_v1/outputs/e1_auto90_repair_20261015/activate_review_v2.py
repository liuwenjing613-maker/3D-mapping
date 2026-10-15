from pathlib import Path
import os,signal,time,json,subprocess
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015')
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
read=lambda p:json.loads(p.read_text())
pid=read(R/'postprocess_process.json')['pid'];proc=Path('/proc')/str(pid)
assert not (R/'postprocess_process_v2.json').exists()
assert str(C/'postprocess.py').encode() in (proc/'cmdline').read_bytes()
os.kill(pid,signal.SIGSTOP)
try:
 state=read(R/'postprocess_status.json')
 assert state['stage']=='waiting_for_frozen_predictions',state
 assert not (proc/'task'/str(pid)/'children').read_text().strip()
 assert not (R/'evaluation_r2.log').exists()
 os.kill(pid,signal.SIGTERM)
finally:os.kill(pid,signal.SIGCONT)
for _ in range(40):
 if not proc.exists() or 'State:\tZ' in (proc/'status').read_text():break
 time.sleep(.1)
else:raise RuntimeError('old review coordinator did not exit')
with (R/'postprocess_v2.log').open('xb') as log:
 child=subprocess.Popen(['python3','-u',str(C/'review_tools/postprocess.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':child.pid,'previous_pid':pid,'previous_coordinator_stopped_while_waiting':True,'mapping_pipeline_untouched':True,'reason':'add exact before/after overlap conflict visualization','unix':time.time()}
(R/'postprocess_process_v2.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
