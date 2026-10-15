from pathlib import Path
import sys,json,hashlib,time
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
receipt=json.loads((H/'delivery_receipt.json').read_text(encoding='utf-8'));remote=receipt['archive'];local=H/Path(remote).name
for attempt in range(4):
 c=None
 try:
  c=connect();c.get_transport().set_keepalive(10)
  with c.open_sftp() as s:
   size=s.stat(remote).st_size;offset=local.stat().st_size if local.exists() else 0
   assert 0<=offset<=size==receipt['bytes'];print('RESUME',offset,'/',size,flush=True)
   with s.open(remote,'rb') as src,local.open('ab') as dest:
    src.seek(offset);src.prefetch(size,max_concurrent_requests=16)
    while offset<size:
     b=src.read(min(1024*1024,size-offset))
     if not b:raise EOFError('short transfer')
     dest.write(b);offset+=len(b)
  break
 except Exception as e:
  print('TRANSFER_RETRY',attempt,type(e).__name__,flush=True)
  if attempt==3:raise
  time.sleep(2)
 finally:
  if c:c.close()
digest=hashlib.sha256(local.read_bytes()).hexdigest();assert digest==receipt['archive_sha256']
print('ARCHIVE_SHA256_VERIFIED',digest,flush=True)
