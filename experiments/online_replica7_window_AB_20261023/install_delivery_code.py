from pathlib import Path
import tarfile,hashlib,json,shutil
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023').resolve()
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
manifest=json.loads((O/'delivery_code_manifest.json').read_text())
with tarfile.open(O/'delivery_code.tar.gz') as tar:
 assert {m.name for m in tar.getmembers()}==set(manifest)
 for m in tar.getmembers():
  p=(C/m.name).resolve();assert p.is_relative_to(C) and m.isfile()
  data=tar.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==manifest[m.name]
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
shutil.copyfile(O/'delivery_code_manifest.json',C/'delivery_code_manifest.json')
print('SOURCE_AND_DELIVERY_EVIDENCE_PASS',len(manifest))
