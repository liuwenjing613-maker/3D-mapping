"""Install only downloaded, hash-verified completed outputs into the local review page."""
from pathlib import Path
import hashlib,json,tarfile,shutil
L=Path(__file__).resolve().parent;D=L/'window_repair400';W=L/'comparison_page/repair_window_only'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
complete=read(D/'post_complete.json');assert complete['status']=='PASS'
assert sha(D/'review.tar.gz')==complete['review_sha256']
W.mkdir(exist_ok=True)
with tarfile.open(D/'review.tar.gz') as tar:
 for m in tar.getmembers():
  p=(W/m.name).resolve();assert p.is_relative_to(W.resolve()) and m.isfile(),m.name
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
for row in complete['PLY_files']:
 p=D/Path(row['path']).name;assert sha(p)==row['sha256'];shutil.copyfile(p,W/p.name);assert sha(W/p.name)==row['sha256']
for src,dest in [('window_repair_viewer.html','index.html'),('window_repair_viewer.js','app.js')]:shutil.copyfile(L/src,W/dest)
for row in read(W/'manifest.json')['pairs']:
 for f in row['files']:
  p=(W/f['path']).resolve();assert p.is_relative_to((L/'comparison_page').resolve()) and sha(p)==f['sha256']
for rel,href in [('comparison_page/index.html','repair_window_only/index.html'),('comparison_page/window_only/index.html','../repair_window_only/index.html')]:
 p=L/rel;s=p.read_text(encoding='utf-8')
 if href not in s:s=s.replace('</h1>','</h1><p><a href="'+href+'">查看完整修复的原方案 / 新方案 R2 与 PLY 对照 →</a></p>',1);p.write_text(s,encoding='utf-8')
print('PASS http://127.0.0.1:8801/repair_window_only/index.html')
