from pathlib import Path
import os,subprocess,sys,json,shlex
C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031')
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
S=Path('/home/chenkejun/beauty/ovimap_aligned_eval_20260908')
R=Path('/data/chenkejun/ovimap_runtime_20260908')
def container(inner,readonly_code=True,extra=None):
 args=['bwrap','--unshare-user','--uid','0','--gid','0','--bind',str(R/'rootfs'),'/', '--dev-bind','/dev','/dev','--tmpfs','/dev/shm','--unshare-ipc','--proc','/proc','--ro-bind','/sys','/sys','--bind','/data','/data','--ro-bind',str(S),'/workspace','--ro-bind',str(S),str(S),'--ro-bind' if readonly_code else '--bind',str(C),'/experiment','--ro-bind','/home/chenkejun/beauty/conceptgraphs/data/Replica','/datasets/Replica','--chdir','/workspace/official_ovimap','--setenv','PATH','/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','--setenv','HOME','/root','--setenv','PYTHONDONTWRITEBYTECODE','1','--unsetenv','LD_LIBRARY_PATH']
 for k,v in (extra or {}).items():args+=['--setenv',k,str(v)]
 shell='source /workspace/official_ovimap/mapping_ros_ws/devel/setup.bash\nexport PYTHONPATH=/experiment/build:/experiment:/workspace/official_ovimap/mapping_ros_ws/devel/lib:$PYTHONPATH\nexport LD_LIBRARY_PATH=/workspace/official_ovimap/mapping_ros_ws/devel/lib:/workspace/official_ovimap/mapping_ros_ws/build/protobuf_catkin/protobuf_src-prefix/src/protobuf_src/src/.libs:$LD_LIBRARY_PATH\nexec '+shlex.join(inner)
 if (extra or {}).get('CVPR_ORIGINAL'):
  shell=shell.replace('PYTHONPATH=/experiment/build:/experiment:', 'PYTHONPATH=')
 return args+['/bin/bash','-c',shell]
if __name__=='__main__':
 if sys.argv[1]=='build':
  subprocess.run(container(['/usr/bin/python3','-B','/experiment/build_inside.py'],False),check=True)
 else:
  scene,count,mode=sys.argv[1:4];count=int(count)
  dst=O/'native'/f'{scene}_{count}_{mode}'
  if dst.exists():raise FileExistsError(dst)
  dst.mkdir(parents=True)
  script='/experiment/native_instrumented.py' if mode!='original' else '/workspace/official_ovimap/scripts/panoptic_mapping_aligned.py'
  inner=['/usr/bin/python3','-B',script,'--dataset','replica','--task','Nyu40','--scene_num',scene,'--data_folder','/datasets/Replica','--result_folder',str(dst),'--start','0','--end',str(count*5),'--step','5','--data_association','2','--inst_association','4','--seg_graph_confidence','3','--temp_panoptics_folder',str(R/'sem_aligned'/scene),'--use_temp_geometrics','--temp_geometrics_folder',str(R/'results'/(scene+'_full400')/'geometrics'),'--intermediate_seg_folder',str(dst/'fused'),'--num_threads','1','--log','aligned','--skip_feature_extraction']
  extra={'CVPR_READOUT_MODE':mode,'CVPR_EXPECTED_COUNT':count,'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1}
  if mode=='original':extra['CVPR_ORIGINAL']='1'
  with (dst/'run.log').open('w') as f:subprocess.run(container(inner,extra=extra),stdout=f,stderr=subprocess.STDOUT,check=True)
  print('NATIVE_DONE',dst,flush=True)
