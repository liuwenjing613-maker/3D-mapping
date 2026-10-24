from pathlib import Path
import tarfile,hashlib,json
L=Path(__file__).resolve().parent
with tarfile.open(L/'review.tar.gz') as tar:
 for m in tar.getmembers():
  p=(L/m.name).resolve();assert p.is_relative_to(L.resolve()) and (m.isfile() or m.isdir()),m.name
  if m.isdir():p.mkdir(parents=True,exist_ok=True);continue
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
manifest=json.loads((L/'review/evidence_manifest.json').read_text(encoding='utf-8'))
for row in manifest['files']:
 p=L/'review'/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
print('REVIEW_HASHES_PASS',len(manifest['files']))
s=json.loads((L/'review/summary.json').read_text(encoding='utf-8'))
for r in s['rows']:
 p=r['repair_pair'];print(r['scene'],r['branch'],round(r['before']['CA_AP50_uniform']*100,2),round(r['after']['CA_AP50_uniform']*100,2),p.keys())
print('PAIR_SAMPLE',json.dumps(s['rows'][0]['repair_pair'],ensure_ascii=True)[:3500])
