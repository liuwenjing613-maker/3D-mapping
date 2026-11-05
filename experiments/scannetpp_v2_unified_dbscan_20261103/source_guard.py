from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def check(branch=None):
 d=json.loads((ROOT/'source_lock.json').read_text())
 for name,h in d['sha256'].items():
  p=Path(name)
  if branch is not None and p.is_relative_to(ROOT) and p.parent!=ROOT and not p.is_relative_to(Path(branch)):continue
  assert hashlib.sha256(p.read_bytes()).hexdigest()==h,('SOURCE_CHANGED',name)
 return d['run_id']
