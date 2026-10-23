"""Unique atomic JSON temp names for concurrent status writers; no numeric edits."""
from pathlib import Path
import ast,hashlib,json,time,runpy,threading
from concurrent.futures import ThreadPoolExecutor
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
SCENES=['room1','room2','office0','office1','office2','office3','office4']
old="p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')"
new="p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')"
rows=[]
for scene in SCENES:
 p=C/scene/'common_continue.py';s=p.read_text()
 if new in s:continue
 assert old in s and 'import json,hashlib,os,time\n' in s
 t=s.replace('import json,hashlib,os,time\n','import json,hashlib,os,time,threading\n').replace(old,new)
 ast.parse(t);p.write_text(t)
 rows.append(dict(path=str(p),before_sha256=hashlib.sha256(s.encode()).hexdigest(),after_sha256=hashlib.sha256(t.encode()).hexdigest()))
module=runpy.run_path(str(C/'office3/common_continue.py'));dump=module['dump']
dest=O/'verification/status_write_stress';dest.mkdir(parents=True,exist_ok=True);output=dest/'shared.json'
barrier=threading.Barrier(8)
def writer(worker):
 barrier.wait()
 for i in range(100):dump(output,dict(worker=worker,iteration=i,payload='x'*1000))
with ThreadPoolExecutor(max_workers=8) as pool:
 list(pool.map(writer,range(8)))
last=json.loads(output.read_text());assert last['worker'] in range(8) and last['iteration']==99
assert not list(dest.glob('*.tmp'))
sample={'status':'PASS','unicode':'状态','fraction':.125,'values':[0,1,2]};dump(output,sample)
assert output.read_text()==json.dumps(sample,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
result=dict(status='PASS',unix=time.time(),scope='JSON temporary names only; values and numeric algorithm unchanged',files=rows,
 concurrent_threads=8,concurrent_writes=800,valid_final_json=True,no_temporary_files_left=True,serialization_bytes_unchanged=True)
(O/'status_write_race_fix.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
