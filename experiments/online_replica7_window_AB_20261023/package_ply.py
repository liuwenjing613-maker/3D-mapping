from pathlib import Path
import json,hashlib,tarfile,time
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
assert json.loads((O/'post_complete.json').read_text())['status']=='PASS'
dest=O/'deliverables.tar.gz';marker=O/'deliverables_package.json'
if marker.exists():print(marker.read_text());raise SystemExit
start=time.time();tmp=O/'deliverables.tar.gz.partial'
with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:tar.add(O/'deliverables',arcname='deliverables')
tmp.replace(dest);h=hashlib.sha256()
with dest.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
d=dict(status='PASS',path=str(dest),bytes=dest.stat().st_size,sha256=h.hexdigest(),PLY_files=140,seconds=time.time()-start)
marker.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
