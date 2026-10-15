from pathlib import Path
import hashlib,json,shutil,zipfile
H=Path(__file__).resolve().parent
ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
OUT=ROOT/'room2_E1_overmerge_audit_20261014';OUT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(H/'web_assets.zip') as z:
 for n in z.namelist():
  target=(OUT/n).resolve();assert target.is_relative_to(OUT.resolve())
 z.extractall(OUT)
hashes=json.loads((OUT/'files_sha256.json').read_text(encoding='utf-8'))
for name,h in hashes.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h,name
for name in ['index.html','style.css','app.js']:shutil.copyfile(H/name,OUT/name)
d=json.loads((OUT/'data.json').read_text(encoding='utf-8'));missing=[]
for s in d['sources']:
 if not (OUT/s['ids']).resolve().is_file():missing.append(s['ids'])
for f in d['frames'].values():
 for k in ['rgb','points']:
  if not (OUT/f[k]).resolve().is_file():missing.append(f[k])
assert not missing,missing
print(json.dumps(dict(status='PASS',assets=len(hashes),stats=d['stats'],url='http://127.0.0.1:8785/'+OUT.name+'/index.html'),ensure_ascii=False))
