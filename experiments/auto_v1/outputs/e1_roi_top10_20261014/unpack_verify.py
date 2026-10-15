from pathlib import Path
import zipfile,json,hashlib
H=Path(__file__).resolve().parent;W=H.parents[1]
D=W/'results/固定案例_三模型对比_20261006/room2_E1_prerepair_CUT_top10_20261014'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
receipt=json.loads((H/'delivery_receipt.json').read_text(encoding='utf-8'))
p=H/'room2_CUT_top10_candidates.zip';assert sha(p)==receipt['archive_sha256']
D.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(p) as z:
 for name in z.namelist():
  dest=(D/name).resolve();assert dest.is_relative_to(D.resolve())
 z.extractall(D)
files=json.loads((D/'files_sha256.json').read_text(encoding='utf-8'))
for rel,digest in files.items():assert sha(D/rel)==digest,rel
result={'status':'PASS','archive_sha256':receipt['archive_sha256'],'files_verified':len(files),'target':str(D)}
(H/'download_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('DOWNLOAD_VERIFIED',len(files))
