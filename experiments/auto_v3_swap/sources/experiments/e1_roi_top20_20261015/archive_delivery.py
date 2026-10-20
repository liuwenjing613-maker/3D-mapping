from pathlib import Path
import sys,zipfile,json,shlex
H=Path(__file__).resolve().parent;F=H.parent/'overmerge_id_ratio_top20_20261015';ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
W=ROOT/'room2_E1_retained_ROI_top20_20261015';O=ROOT/'room2_E1_prerepair_CUT_top20_20261015'
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
zfile=F/'retained_ROI_page.zip'
with zipfile.ZipFile(zfile,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
 for p in W.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(W).as_posix())
c=connect();c.get_transport().set_keepalive(10)
try:
 with c.open_sftp() as s:
  for folder,code in [(H,'e1_roi_top20_20261015'),(F,'overmerge_id_ratio_top20_20261015')]:
   for p in folder.iterdir():
    if p.suffix in ['.py','.js','.txt'] or p.name.endswith('_verification.json') or p.name in ['browser_qa.json','retained_build_audit.json']:
     s.put(str(p),'/home/chenkejun/CVPR/experiments/'+code+'/'+p.name)
  for name in ['index.html','selection_setup.json','ranking_audit.json','plan.txt','final_files_sha256.json']:s.put(str(O/name),'/data/chenkejun/CVPR/results/e1_roi_top20_20261015/review/'+name)
  r='/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015'
  s.put(str(zfile),r+'/retained_ROI_page.zip')
 code="from pathlib import Path;import zipfile;r=Path("+repr(r)+");out=r/'retained_ROI_review';out.mkdir(exist_ok=True);z=zipfile.ZipFile(r/'retained_ROI_page.zip');assert all((out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist());z.extractall(out);link=r/'room2_E1_prerepair_CUT_top20_20261015';target=Path('/data/chenkejun/CVPR/results/e1_roi_top20_20261015/review');link.symlink_to(target) if not link.exists() else None;assert link.resolve()==target;print('ARCHIVED',out)"
 _,o,e=c.exec_command('python3 -c '+shlex.quote(code));print(o.read().decode(),e.read().decode());assert o.channel.recv_exit_status()==0
finally:c.close()
