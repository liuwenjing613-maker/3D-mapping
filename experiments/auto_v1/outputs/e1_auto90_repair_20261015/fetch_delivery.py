from pathlib import Path
import sys,json,hashlib,zipfile
H=Path(__file__).resolve().parent;W=H.parents[1]
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
R='/data/chenkejun/CVPR/results/e1_auto90_repair_20261015';C='/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015'
c=connect();c.get_transport().set_keepalive(10)
try:
 with c.open_sftp() as s:
  for name in ['postprocess_complete.json','review_archive.json','report_complete.json','evaluation_complete.json','prepare_complete.json','tests_complete.json','pipeline_complete.json','tracking_queue_complete.json','queue_v2_adoption.json','conflict_assets_complete.json','postprocess_process_v2.json','pipeline_code_freeze.json','tracking_validation.json']:
   s.get(R+'/'+name,str(H/name))
  receipt=json.loads((H/'review_archive.json').read_text())
  with s.open(R+'/review.zip','rb') as src,(H/'review.zip').open('wb') as dst:
   src.prefetch(max_concurrent_requests=16)
   while b:=src.read(1048576):dst.write(b)
  for name in ['adapter_provenance.json','run_batch.py','run_strict_repair.py','run_history_core.py','expand_room2.py','run_observation_repair.py']:
   s.get(C+'/'+name,str(H/name))
 assert hashlib.sha256((H/'review.zip').read_bytes()).hexdigest()==receipt['sha256']
 target=W/'results/固定案例_三模型对比_20261006/room2_E1_auto90_repair_20261015';target.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(H/'review.zip') as z:
  assert all((target/n).resolve().is_relative_to(target.resolve()) for n in z.namelist());z.extractall(target)
 for name in ['summary.json','data.json','export_receipt.json']:
  v=json.loads((target/name).read_text(encoding='utf-8'))
  if name=='summary.json':print(json.dumps(v,ensure_ascii=False))
 print('DELIVERED',target,'FILES',receipt['files'],'BYTES',receipt['bytes'])
finally:c.close()
