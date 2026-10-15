from pathlib import Path
import json,shlex,sys,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006';WEB=ROOT/'room2_E1_retained_ROI_20261014'
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
archive=H/'retained_ROI_page.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
 for p in WEB.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(WEB).as_posix())
print('ARCHIVE_BYTES',archive.stat().st_size,flush=True)
c=connect();c.get_transport().set_keepalive(10)
C='/home/chenkejun/CVPR/experiments/overmerge_id_ratio_20261014';R='/data/chenkejun/CVPR/results/overmerge_id_ratio_20261014'
try:
 with c.open_sftp() as s:
  for name in ['build_retained.py','route_retained.py','verify_retained.py','retained_browser_qa.json','retained_build_audit.json','retained_verification.json','index.html','app.js','add_rejected_toggle.py','rejected_comparison.js','rejected_toggle_qa.json','README.txt']:
   s.put(str(H/name),C+'/'+name)
  s.put(str(archive),R+'/retained_ROI_page.zip')
 code="from pathlib import Path;import zipfile;r=Path("+repr(R)+");out=r/'retained_ROI_review';out.mkdir(exist_ok=True);z=zipfile.ZipFile(r/'retained_ROI_page.zip');assert all((out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist());z.extractall(out);print('ARCHIVED',out)"
 _,o,e=c.exec_command('python3 -c '+shlex.quote(code));print(o.read().decode(),e.read().decode());assert o.channel.recv_exit_status()==0
finally:c.close()
