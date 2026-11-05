"""Limit writes from adapted experiment processes to this experiment's owned roots."""
from pathlib import Path
import os,sys
ROOTS=('/data/chenkejun/CVPR/results/ovimap_aligned_main_onlinev2_20261101/gpu_locks','/home/chenkejun/CVPR/experiments/unified_full_depth_20261101','/data/chenkejun/CVPR/results/unified_full_depth_20261101')
def inside(p,r):return p==r or p.startswith(r+'/')
def install():
 if getattr(sys,'_cvpr_aligned_write_fence',False):return
 sys._cvpr_aligned_write_fence=True
 def validate(p):
  if not isinstance(p,(str,bytes,os.PathLike)):return
  path=os.path.realpath(os.fsdecode(p))
  if (inside(path,'/home/chenkejun') or inside(path,'/data/chenkejun')) and not any(inside(path,r) for r in ROOTS):
   raise PermissionError('ALIGNED_EXPERIMENT_READ_ONLY_EXTERNAL_PATH: '+path)
 def audit(event,args):
  if event=='open':
   mode=args[1];flags=args[2]
   if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):validate(args[0])
  elif event in ['os.remove','os.rmdir','os.mkdir','os.chmod','os.utime']:validate(args[0])
  elif event in ['os.rename','os.link']:
   for p in args[:2]:validate(p)
  elif event=='os.symlink':validate(args[1])
 sys.addaudithook(audit)
install()
