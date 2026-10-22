from online_common import *
from run_selection import ENV,devices
import subprocess,os,time,traceback
A=C/'all_edge';O=R/'all_edge_20261021'
def run(name,args,device=None):
 write(R/'delivery_progress.json',dict(stage=name,time=time.time()))
 with (R/(name+'.log')).open('ab') as log:
  p=subprocess.Popen([PY,'-B','-u',*map(str,args)],cwd=C,env={**ENV,**({'CUDA_VISIBLE_DEVICES':device} if device else {})},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 write(R/'delivery_progress.json',dict(stage=name,pid=p.pid,time=time.time()))
 if p.wait():raise RuntimeError(name+' failed')
try:
 while not (R/'pipeline_progress.json').exists() or read(R/'pipeline_progress.json').get('stage')!='mapping_complete_selection_not_started':
  if (R/'pipeline_failure.json').exists():raise RuntimeError((R/'pipeline_failure.json').read_text())
  time.sleep(5)
 run('selection_driver',[C/'run_selection.py'])
 run('causality_audit',[C/'verify_online.py'])
 run('all_edge_prepare',[A/'screen.py',80,'prepare'])
 run('all_edge_selection',[A/'run_variant.py',80,devices()[0]])
 run('all_edge_render',[A/'render.py',80])
 write(R/'delivery_progress.json',dict(stage='ready_for_review_QA',time=time.time(),summary=read(O/'complete.json')))
except Exception:
 write(R/'delivery_failure.json',dict(error=traceback.format_exc(),time=time.time()))
 raise
