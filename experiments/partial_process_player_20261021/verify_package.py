"""Check exact archived bytes without needing the large replay assets."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
for name,expected in manifest['files'].items():
 actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
 assert actual==expected,name
print('PASS:',len(manifest['files']),'files match archived SHA-256')
