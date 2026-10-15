from pathlib import Path
import sys,shlex
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
C='/home/chenkejun/CVPR/experiments/overmerge_id_ratio_top20_20261015'
R='/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015'
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
c=connect();c.get_transport().set_keepalive(10)
try:
 def run(s):
  _,o,e=c.exec_command(s)
  for line in o:print(line,end='',flush=True)
  err=e.read().decode();print(err,end='');assert o.channel.recv_exit_status()==0
 action=sys.argv[1]
 if action=='upload':
  run('mkdir -p '+C+' '+R)
  with c.open_sftp() as s:
   for name in sys.argv[2:]:s.put(str(H/name),C+'/'+name)
 elif action=='launch':
  code="from pathlib import Path;import subprocess; r=Path("+repr(R)+"); f=(r/'score.log').open('ab'); p=subprocess.Popen(["+repr(PY)+",'-u',"+repr(C+'/score.py')+"],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); (r/'score.pid').write_text(str(p.pid));print(p.pid)"
  run('python3 -c '+shlex.quote(code))
 elif action=='status':
  code="from pathlib import Path;r=Path("+repr(R)+");print((r/'score.log').read_text()[-3500:]);print((r/'audit.json').read_text() if (r/'audit.json').exists() else 'RUNNING')"
  run('python3 -c '+shlex.quote(code))
 elif action=='exec':run(PY+' '+C+'/'+sys.argv[2])
 elif action=='get':
  with c.open_sftp() as s:
   for name in sys.argv[2:]:
    with s.open(R+'/'+name,'rb') as remote, (H/name).open('wb') as local:
     remote.prefetch(max_concurrent_requests=16)
     while data:=remote.read(1024*1024):local.write(data)
    print('DOWNLOADED',name,flush=True)
finally:c.close()
