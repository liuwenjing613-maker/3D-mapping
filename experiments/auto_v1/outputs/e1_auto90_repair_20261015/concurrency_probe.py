from pathlib import Path
import json,subprocess
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
s=json.loads((R/'tracking_queue_status.json').read_text())
print('RAM',[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))])
for x in s['active']:
 p=Path('/proc')/str(x['pid'])
 print('ACTIVE',x)
 print('RSS',[x for x in (p/'status').read_text().splitlines() if x.startswith(('VmRSS:','VmPeak:','Threads:'))])
 print('CMD',(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True))
