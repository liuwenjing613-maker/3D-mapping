from online_common import *
from run_selection import ENV
import subprocess,time,traceback
try:
 while not (R/'ply/manifest.json').exists():
  if (R/'pipeline_failure.json').exists():raise RuntimeError((R/'pipeline_failure.json').read_text())
  time.sleep(3)
 subprocess.run([PY,'-B','-u',str(C/'prepare_3d80.py')],cwd=C,env=ENV,check=True)
except Exception:
 write(R/'ply80_failure.json',dict(error=traceback.format_exc()))
 raise
