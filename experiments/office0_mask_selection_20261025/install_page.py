from pathlib import Path
import json,hashlib,tarfile,shutil,gzip
L=Path(__file__).resolve().parent
W=L.parent/'online_full_scene_room0_AB_20261023/comparison_page/office0_masks'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads((L/'export_complete.json').read_text(encoding='utf-8'));assert receipt['status']=='PASS' and sha(L/'page_data.tar.gz')==receipt['archive_sha256']
W.mkdir(parents=True,exist_ok=True)
with tarfile.open(L/'page_data.tar.gz') as tar:
 for m in tar.getmembers():
  p=(W/m.name).resolve();assert p.is_relative_to(W.resolve()) and m.isfile(),m.name
  p.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(m) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
manifest=json.loads((W/'file_manifest.json').read_text(encoding='utf-8'))
for name,v in manifest['files'].items():
 p=W/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
index=json.loads((W/'index.json').read_text(encoding='utf-8'));assert len(index['runs'])==10
rows=[]
for r in index['runs']:
 with gzip.open(W/r['path'],'rt',encoding='utf-8') as f:d=json.load(f)
 assert len(d['candidates'])==r['counts']['raw_unique']
 assert {x['id'] for x in d['rows'] if x['status']=='kept'}=={x['mask_id'] for x in d['gate']['tracks'].values()}
 for x in d['rows']:
  for k in ['mask','full','zoom','rgb']:assert x[k] in manifest['files']
 for x in d['sources']:
  assert x['labels'] in manifest['files'] and x['original'] in manifest['files']
 rows.append(dict(key=r['key'],candidates=len(d['candidates']),masks=len(d['rows']),final=d['counts']['final']))
for name in ['index.html','app.js','style.css']:shutil.copyfile(L/name,W/name)
p=W.parent/'replica7/index.html';s=p.read_text(encoding='utf-8')
if '../office0_masks/index.html' not in s:s=s.replace('<h1>70次修复，逐次对比</h1>','<h1>70次修复，逐次对比</h1><p><a href="../office0_masks/index.html">office0：Mask 完整筛选过程 →</a></p>');p.write_text(s,encoding='utf-8')
result=dict(status='PASS',files=len(manifest['files']),runs=rows,archive_sha256=receipt['archive_sha256'],all_assets_referenced_exist=True,url='http://127.0.0.1:8801/office0_masks/index.html')
(L/'local_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=True))
