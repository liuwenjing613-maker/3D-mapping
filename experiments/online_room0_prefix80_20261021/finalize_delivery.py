"""Package only verified outputs after selection and comparison images complete."""
from online_common import *
import subprocess
import time
import zipfile
import sys

deadline=time.time()+3600
while not ((R/'selection_complete.json').exists() and (R/'visuals_complete.json').exists()):
    if (R/'selection_failure.json').exists():raise RuntimeError((R/'selection_failure.json').read_text())
    assert time.time()<deadline
    time.sleep(5)
for script in (['build_review.py'] if '--package-only' not in sys.argv else [])+['verify_online.py']:
    subprocess.run([PY,'-B','-u',str(C/script)],cwd=C,check=True)
assert read(R/'causality_audit.json')['status']=='PASS'
review=R/'review'
files=[dict(file=p.relative_to(review).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
       for p in sorted(review.rglob('*')) if p.is_file()]
write(R/'delivery_manifest.json',dict(status='PASS',files=files,bytes=sum(f['bytes'] for f in files)))
archive=R/'review_delivery.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for row in files:z.write(review/row['file'],row['file'])
write(R/'delivery_complete.json',dict(status='PASS',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
    review=read(R/'review_complete.json'),causality_audit_sha256=sha(R/'causality_audit.json')))
print('DELIVERY_READY',archive.stat().st_size,flush=True)
