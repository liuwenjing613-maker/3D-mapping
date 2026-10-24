"""Verify every downloaded PLY and install 70 lossless pairs using same-drive hardlinks."""
from pathlib import Path
import os,json,tarfile,hashlib,shutil,numpy as np
L=Path(__file__).resolve().parent
W=L.parent/'online_full_scene_room0_AB_20261023/comparison_page/replica7'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
package=read(L/'deliverables_package.json');assert sha(L/'deliverables.tar.gz')==package['sha256']
with tarfile.open(L/'deliverables.tar.gz') as tar:
 for m in tar.getmembers():
  p=(L/m.name).resolve();assert p.is_relative_to(L.resolve()) and (m.isdir() or m.isfile()),m.name
  if m.isdir():p.mkdir(parents=True,exist_ok=True);continue
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists() and p.stat().st_size==m.size:continue
  with tar.extractfile(m) as src,p.open('wb') as dest:shutil.copyfileobj(src,dest,8*1024*1024)
W.mkdir(parents=True,exist_ok=True);s=read(L/'review/summary.json');exports={(e['scene'],e['branch'],e['checkpoint'],e['stage']):e for e in s['exports']}
dtype=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')])
def data(path,n):
 with path.open('rb') as f:
  while f.readline()!=b'end_header\n':assert f.tell()<2048
  offset=f.tell()
 assert path.stat().st_size==offset+31*n
 return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=(n,))
pairs=[]
for scene in s['scenes']:
 for branch in ['A','B']:
  for n in [50,100,200,300,400]:
   files=[];arrays=[]
   for stage in ['before','after']:
    e=exports[scene,branch,n,stage];p=L/'deliverables'/scene/Path(e['path']).name
    assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],str(p)
    target=W/'assets'/scene/p.name;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
     try:os.link(p,target)
     except PermissionError:shutil.copyfile(p,target)
    assert os.path.samefile(p,target) or sha(target)==e['sha256']
    arrays.append(data(p,e['points']));files.append(dict(path=target.relative_to(W).as_posix(),sha256=e['sha256'],instances=e['positive_instances'],source_sha256=e['source_sha256']))
   a,b=arrays;assert len(a)==len(b) and all(np.array_equal(a[k],b[k]) for k in ['x','y','z'])
   changed=int(np.count_nonzero(a['instance_id']!=b['instance_id']))
   bounds=[[float(a[k].min()) for k in ['x','y','z']],[float(a[k].max()) for k in ['x','y','z']]]
   pairs.append(dict(key=f'{scene}_{branch}_{n}',scene=scene,branch=branch,checkpoint=n,points=len(a),changed=changed,bounds=bounds,files=files))
   print('PAIR_PASS',scene,branch,n,len(a),changed,flush=True)
manifest=dict(status='PASS',pairs=pairs,PLY_files=140,all_geometry_pairs_exact=True,source=str(L/'review/evidence_manifest.json'))
(W/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
for src,name in [(L/'viewer.html','index.html'),(L/'viewer.js','app.js'),(L/'review/summary.json','summary.json'),(L/'diagnostics.json','diagnostics.json'),(L/'review/REPORT.txt','REPORT.txt')]:shutil.copyfile(src,W/name)
parent=W.parent/'index.html';html=parent.read_text(encoding='utf-8')
if 'replica7/index.html' not in html:
 html=html.replace('</h1>','</h1><p><a href="replica7/index.html">七场景完整在线 A/B：70次修复前后 PLY 与 R2 →</a></p>',1);parent.write_text(html,encoding='utf-8')
(L/'delivery_verification.json').write_text(json.dumps(dict(status='PASS',pairs=70,PLY_files=140,package_sha256=package['sha256'],all_PLY_hashes_exact=True,all_paired_xyz_exact=True,page=str(W)),indent=2)+'\n')
print('PAGE_READY http://127.0.0.1:8801/replica7/index.html')
