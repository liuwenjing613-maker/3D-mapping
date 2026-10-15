from pathlib import Path
import sys,shlex,json
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
C='/home/chenkejun/CVPR/experiments/e1_roi_top20_20261015'
R='/data/chenkejun/CVPR/results/e1_roi_top20_20261015'
c=connect()
try:
 def cmd(s):
  _,o,e=c.exec_command(s);text=o.read().decode();err=e.read().decode();print(text,err);assert o.channel.recv_exit_status()==0
 action=sys.argv[1]
 if action=='upload':
  cmd('mkdir -p '+C+' '+R)
  with c.open_sftp() as s:
   for name in ['prepare_inputs.py','infer_cropformer.py','infer_sam.py','render_helpers.py','render_annotation.py','audit_inputs.py','pipeline.py','normalize_manifest.py','finalize_projections.py']:
    s.put(str(H/name),C+'/'+name)
  print('UPLOADED')
 elif action=='launch':
  script="from pathlib import Path;import subprocess; r=Path("+repr(R)+"); assert not (r/'pipeline.pid').exists(); f=(r/'pipeline.log').open('ab'); p=subprocess.Popen(['python3','-u',"+repr(C+'/pipeline.py')+"],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); (r/'pipeline.pid').write_text(str(p.pid));print(p.pid)"
  cmd('python3 -c '+shlex.quote(script))
 elif action=='status':
  script="from pathlib import Path; r=Path("+repr(R)+"); print('\\n'.join(p.name+': '+p.read_text()[-900:] for p in r.glob('*.log')));print('\\n'.join(p.name+': '+p.read_text() for p in r.glob('pipeline_*.json')))"
  cmd('python3 -c '+shlex.quote(script))
 elif action=='finalize':
  with c.open_sftp() as s:
   for name in ['finalize_projections.py','render_annotation.py']:
    s.put(str(H/name),C+'/'+name)
  cmd('/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python '+C+'/finalize_projections.py')
 elif action=='get':
  with c.open_sftp() as s:
   for name in sys.argv[2:]:s.get(R+'/'+name,str(H/name));print('DOWNLOADED',name)
finally:c.close()
