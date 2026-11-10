from pathlib import Path
import os,subprocess,sys,json,shlex,hashlib,time
C=Path('/home/chenkejun/CVPR/experiments/scannet_ovimap_strict_dbscan_20261104')
O=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104')
S=Path('/home/chenkejun/beauty/ovimap_aligned_eval_20260908')
R=Path('/data/chenkejun/ovimap_runtime_20260908')
def container(inner,build=False,extra=None):
 args=['bwrap','--unshare-user','--uid','0','--gid','0','--ro-bind',str(R/'rootfs'),'/', '--dev-bind','/dev','/dev','--tmpfs','/tmp','--tmpfs','/dev/shm','--unshare-ipc','--proc','/proc','--ro-bind','/sys','/sys','--ro-bind','/data','/data','--bind',str(O),str(O),'--ro-bind',str(S),'/workspace','--ro-bind',str(S),str(S),'--bind' if build else '--ro-bind',str(C),'/experiment','--ro-bind','/home/chenkejun/CVPR/experiments/ovimap_full_depth_control_20261101/build','/experiment/build','--ro-bind','/home/chenkejun/beauty/conceptgraphs/data/Replica','/datasets/Replica','--chdir','/workspace/official_ovimap','--setenv','PATH','/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','--setenv','HOME','/tmp','--setenv','PYTHONDONTWRITEBYTECODE','1','--unsetenv','LD_LIBRARY_PATH']
 for k,v in dict(OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,**(extra or {})).items():args+=['--setenv',k,str(v)]
 shell='source /workspace/official_ovimap/mapping_ros_ws/devel/setup.bash\nexport PYTHONPATH=/experiment/build:/experiment:/workspace/official_ovimap/mapping_ros_ws/devel/lib:$PYTHONPATH\nexport LD_LIBRARY_PATH=/workspace/official_ovimap/mapping_ros_ws/devel/lib:/workspace/official_ovimap/mapping_ros_ws/build/protobuf_catkin/protobuf_src-prefix/src/protobuf_src/src/.libs:$LD_LIBRARY_PATH\nexec '+shlex.join(inner)
 return args+['/bin/bash','-c',shell]