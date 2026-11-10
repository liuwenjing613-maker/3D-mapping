from pathlib import Path
import hashlib,json,shlex,subprocess,difflib
C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031')
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
S=Path('/home/chenkejun/beauty/ovimap_aligned_eval_20260908/official_ovimap')
P=S/'mapping_ros_ws/src/consistent_panoptic_mapping/consistent_gsm'
C.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
(C/'include/consistent_mapping').mkdir(parents=True,exist_ok=True)
(C/'build').mkdir(exist_ok=True)
h=(P/'include/consistent_mapping/global_segment_map_py.h').read_text()
h2=h.replace('bool integrateFrame();','bool integrateFrame();\npybind11::dict cvprTsdfState();\npybind11::dict cvprSurface();')
assert h2!=h
(C/'include/consistent_mapping/global_segment_map_py.h').write_text(h2)
src=(P/'src/global_segment_map_py.cpp').read_text()
extra=(C/'readout_methods.cpp.txt').read_text()
out=src.replace('PYBIND11_MODULE(consistent_gsm, m)',extra+'\nPYBIND11_MODULE(consistent_gsm, m)')
out=out.replace('.def("integrateFrame", &GlobalSegmentMap_py::integrateFrame)', '.def("integrateFrame", &GlobalSegmentMap_py::integrateFrame)\n        .def("cvprTsdfState", &GlobalSegmentMap_py::cvprTsdfState)\n        .def("cvprSurface", &GlobalSegmentMap_py::cvprSurface)')
(C/'global_segment_map_py.cpp').write_text(out)
(C/'readout.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),out.splitlines(True),fromfile='original',tofile='cvpr_readout')))
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'src/global_segment_map_py.cpp',P/'include/consistent_mapping/global_segment_map_py.h',S/'mapping_ros_ws/devel/lib/consistent_gsm.cpython-38-x86_64-linux-gnu.so']}
(C/'original_hashes.json').write_text(json.dumps(manifest,indent=2))
print('SOURCE_READY')
