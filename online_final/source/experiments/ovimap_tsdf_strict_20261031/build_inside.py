from pathlib import Path
import shlex,subprocess,json
C=Path('/experiment')
B=Path('/workspace/official_ovimap/mapping_ros_ws/build/consistent_gsm')
d={}
for line in (B/'CMakeFiles/consistent_gsm.dir/flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);d[k]=shlex.split(v)
obj=str(C/'build/global_segment_map_py.cpp.o')
cmd=['/usr/bin/c++']+d['CXX_DEFINES']+['-I/experiment/include']+d['CXX_INCLUDES']+d['CXX_FLAGS']+['-c',str(C/'global_segment_map_py.cpp'),'-o',obj]
subprocess.run(cmd,check=True)
cmd=shlex.split((B/'CMakeFiles/consistent_gsm.dir/link.txt').read_text())
for i,x in enumerate(cmd):
 if i and cmd[i-1]=='-o':cmd[i]=str(C/'build/consistent_gsm.cpython-38-x86_64-linux-gnu.so')
 elif x=='CMakeFiles/consistent_gsm.dir/src/global_segment_map_py.cpp.o':cmd[i]=obj
 elif x.startswith('CMakeFiles/'):cmd[i]=str(B/x)
subprocess.run(cmd,cwd=B,check=True)
print('BUILD_PASS',flush=True)
