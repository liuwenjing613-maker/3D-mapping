from pathlib import Path
import sys,shlex,json,hashlib,zipfile
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
C='/home/chenkejun/CVPR/experiments/mask_version_selection_20261015';R='/data/chenkejun/CVPR/results/mask_version_selection_20261015';PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
c=connect();c.get_transport().set_keepalive(10)
try:
 def run(cmd):
  _,o,e=c.exec_command(cmd);print(o.read().decode(),e.read().decode());assert o.channel.recv_exit_status()==0
 mode=sys.argv[1]
 if mode in ['launch','recompute']:
  run('mkdir -p '+C+' '+R)
  with c.open_sftp() as s:
   for name in ['compute.py','selection_rules.py']:s.put(str(H/name),C+'/'+name)
  if mode=='recompute':run(PY+' '+C+'/compute.py')
  else:
   code="from pathlib import Path;import subprocess;r=Path("+repr(R)+");assert not (r/'compute.pid').exists();f=(r/'compute.log').open('ab');p=subprocess.Popen(["+repr(PY)+",'-u',"+repr(C+'/compute.py')+"],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(r/'compute.pid').write_text(str(p.pid));print(p.pid)"
   run('python3 -c '+shlex.quote(code))
 elif mode=='status':run('tail -c 2800 '+R+'/compute.log')
 elif mode=='get':
  with c.open_sftp() as s:
   for name in ['receipt.json','selection_data.zip']:
    with s.open(R+'/'+name,'rb') as src,(H/name).open('wb') as dst:
     src.prefetch(max_concurrent_requests=16)
     while b:=src.read(1048576):dst.write(b)
  assert hashlib.sha256((H/'selection_data.zip').read_bytes()).hexdigest()==json.loads((H/'receipt.json').read_text())['sha256']
  with zipfile.ZipFile(H/'selection_data.zip') as z:
   assert all((H/n).resolve().is_relative_to(H) for n in z.namelist());z.extractall(H)
  print((H/'audit.json').read_text())
finally:c.close()
