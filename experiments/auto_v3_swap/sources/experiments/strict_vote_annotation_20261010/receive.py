from pathlib import Path
import hashlib,json,zipfile,sys
H=Path(__file__).resolve().parent; W=H.parents[1]
for s in sys.argv[1:] or ['room0','room2']:
 h=H/s;receipt=json.loads((h/'delivery_receipt.json').read_text(encoding='utf-8'))
 p=h/(s+'_top30_candidates.zip')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['archive_sha256']
 assert p.stat().st_size==receipt['bytes']
 out=W/'results/固定案例_三模型对比_20261006'/(s+'_strict_vote_annotation_20261010');out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(p) as z:
  assert len(z.namelist())==receipt['files']
  for n in z.namelist():assert (out/n).resolve().is_relative_to(out.resolve())
  z.extractall(out)
 checks=json.loads((out/'files_sha256.json').read_text())
 for rel,digest in checks.items():assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==digest,rel
 (h/'download_verification.json').write_text(json.dumps({'status':'PASS','archive_sha256':receipt['archive_sha256'],'verified_files':len(checks)}))
 print(s,'RECEIVED',len(checks),'files')
