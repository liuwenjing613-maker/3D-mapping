from pathlib import Path
import sys,shlex,json,hashlib,zipfile
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
C='/home/chenkejun/CVPR/experiments/roi_intersection_20261015';R='/data/chenkejun/CVPR/results/roi_intersection_20261015';PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
c=connect();c.get_transport().set_keepalive(10)
try:
 def run(cmd):
  _,o,e=c.exec_command(cmd);print(o.read().decode(),e.read().decode());assert o.channel.recv_exit_status()==0
 mode=sys.argv[1]
 if mode=='launch':
  run('mkdir -p '+C+' '+R)
  with c.open_sftp() as s:s.put(str(H/'compute.py'),C+'/compute.py')
  code="from pathlib import Path;import subprocess;r=Path("+repr(R)+");assert not (r/'compute.pid').exists();f=(r/'compute.log').open('ab');p=subprocess.Popen(["+repr(PY)+",'-u',"+repr(C+'/compute.py')+"],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(r/'compute.pid').write_text(str(p.pid));print(p.pid)"
  run('python3 -c '+shlex.quote(code))
 elif mode=='status':
  code="from pathlib import Path;r=Path("+repr(R)+");print((r/'compute.log').read_text()[-2200:]);print('READY', (r/'receipt.json').exists())"
  run('python3 -c '+shlex.quote(code))
 elif mode=='get':
  with c.open_sftp() as s:
   for name in ['receipt.json','audit.json','intersection_web.zip']:
    with s.open(R+'/'+name,'rb') as src,(H/name).open('wb') as dst:
     src.prefetch(max_concurrent_requests=16)
     while b:=src.read(1048576):dst.write(b)
    print('DOWNLOADED',name)
  receipt=json.loads((H/'receipt.json').read_text(encoding='utf-8'));assert hashlib.sha256((H/'intersection_web.zip').read_bytes()).hexdigest()==receipt['sha256']
  out=H.parents[1]/'results/固定案例_三模型对比_20261006/room2_E1_intersection_ROI_20261015';out.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(H/'intersection_web.zip') as z:
   assert all((out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist());z.extractall(out)
  checks=json.loads((out/'files_sha256.json').read_text(encoding='utf-8'))
  for p,h in checks.items():assert hashlib.sha256((out/p).read_bytes()).hexdigest()==h
  print('VERIFIED',len(checks),'FILES')
finally:c.close()
