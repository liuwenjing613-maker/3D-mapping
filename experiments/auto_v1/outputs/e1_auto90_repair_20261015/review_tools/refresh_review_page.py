"""Refresh presentation-only files after report completion; preserve predictions."""
from pathlib import Path
import json,hashlib,shutil,zipfile
UI=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015');O=R/'review'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((R/'postprocess_complete.json').read_text())['status']=='PASS'
for name in ['review.html','review.css','review.js']:shutil.copyfile(UI/name,O/('index.html' if name=='review.html' else name))
(O/'review_provenance.json').write_text(json.dumps({'code':{str(p):sha(p) for p in UI.glob('*') if p.is_file()},'conflict_assets_receipt_sha256':sha(R/'conflict_assets_complete.json'),'repair_algorithms_unchanged':True},indent=2))
archive=R/'review.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in sorted(O.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(O).as_posix())
receipt={'file':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len([p for p in O.rglob('*') if p.is_file()])}
(R/'review_archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
