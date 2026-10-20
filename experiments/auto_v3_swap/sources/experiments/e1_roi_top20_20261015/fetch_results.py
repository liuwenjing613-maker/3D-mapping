"""Download only complete verified run outputs from the authorized server."""
from pathlib import Path
import sys,time,json,hashlib
H=Path(__file__).resolve().parent;F=H.parent/'overmerge_id_ratio_top20_20261015'
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
R='/data/chenkejun/CVPR/results/e1_roi_top20_20261015';A='/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015'
c=connect();c.get_transport().set_keepalive(10)
try:
 with c.open_sftp() as s:
  def exists(path):
   try:s.stat(path);return True
   except FileNotFoundError:return False
  def read(path):
   with s.open(path) as f:return json.load(f)
  def wait(path):
   started=time.monotonic()
   while not exists(path):
    if exists(R+'/pipeline_error.json'):raise RuntimeError(read(R+'/pipeline_error.json'))
    if time.monotonic()-started>7200:raise TimeoutError(path)
    time.sleep(10)
  def get(remote,local):
   local.parent.mkdir(parents=True,exist_ok=True);size=s.stat(remote).st_size
   with s.open(remote,'rb') as src,local.open('wb') as dst:
    src.prefetch(size,max_concurrent_requests=16)
    while data:=src.read(1024*1024):dst.write(data)
   assert local.stat().st_size==size;print('DOWNLOADED',local.name,size,flush=True)
  print('WAITING_FOR_SEGMENTATIONS',flush=True);wait(R+'/pipeline_complete.json')
  for name in ['delivery_receipt.json','input_manifest.json','ranking_audit.json']:get(R+'/'+name,H/name)
  receipt=json.loads((H/'delivery_receipt.json').read_text(encoding='utf-8'));z=H/'room2_CUT_top20_candidates.zip';get(receipt['archive'],z)
  assert hashlib.sha256(z.read_bytes()).hexdigest()==receipt['archive_sha256'];print('ARCHIVE_VERIFIED',flush=True)
  print('WAITING_FOR_OVERMERGE_FILTER',flush=True);wait(A+'/audit.json')
  for name in ['audit.json','proposed_reject.json','retained.json']:get(A+'/'+name,F/name)
  get(A+'/web/data.json',F/'data.json')
  assert json.loads((F/'audit.json').read_text(encoding='utf-8'))['status']=='PASS'
  print('ALL_RESULTS_DOWNLOADED',flush=True)
finally:c.close()
