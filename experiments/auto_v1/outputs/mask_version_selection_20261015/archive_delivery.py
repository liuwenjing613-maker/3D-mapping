from pathlib import Path
import sys,zipfile,json,shlex,hashlib
H=Path(__file__).resolve().parent;W=H.parents[1]/'results/固定案例_三模型对比_20261006/room2_E1_selected_ROI_20261015'
C='/home/chenkejun/CVPR/experiments/mask_version_selection_20261015';R='/data/chenkejun/CVPR/results/mask_version_selection_20261015'
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
manifest={p.relative_to(W).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in W.rglob('*') if p.is_file() and p.name!='files_sha256.json'}
(W/'files_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
zfile=H/'review_page.zip'
with zipfile.ZipFile(zfile,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
 for p in W.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(W).as_posix())
c=connect();c.get_transport().set_keepalive(10)
try:
 with c.open_sftp() as s:
  for p in H.iterdir():
   if p.suffix in ['.py','.js','.txt'] or p.name in ['page_verification.json','browser_qa.json']:s.put(str(p),C+'/'+p.name)
  s.put(str(zfile),R+'/review_page.zip')
 code='''from pathlib import Path
import json,zipfile,hashlib
r=Path('/data/chenkejun/CVPR/results/mask_version_selection_20261015');out=r/'review';out.mkdir(exist_ok=True)
with zipfile.ZipFile(r/'review_page.zip') as z:
 assert all((out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist());z.extractall(out)
for name,target in [('room2_E1_retained_ROI_top20_20261015','/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015/retained_ROI_review'),('room2_E1_prerepair_CUT_top20_20261015','/data/chenkejun/CVPR/results/e1_roi_top20_20261015/review'),('room2_E1_intersection_ROI_20261015','/data/chenkejun/CVPR/results/roi_intersection_20261015/review')]:
 link=r/name;target=Path(target)
 if not link.exists():link.symlink_to(target)
 assert link.resolve()==target
for name,sha in json.loads((out/'files_sha256.json').read_text()).items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
print('ARCHIVED_AND_VERIFIED',out)
'''
 _,o,e=c.exec_command('python3 -c '+shlex.quote(code));print(o.read().decode(),e.read().decode());assert o.channel.recv_exit_status()==0
finally:c.close()
