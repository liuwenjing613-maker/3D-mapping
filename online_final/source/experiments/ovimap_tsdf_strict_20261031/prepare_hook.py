from pathlib import Path
import difflib
C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031')
S=Path('/home/chenkejun/beauty/ovimap_aligned_eval_20260908/official_ovimap/scripts/panoptic_mapping_aligned.py')
s=S.read_text()
prefix="import sys\nsys.path.insert(0, '/workspace/official_ovimap/scripts')\nsys.path.insert(0, '/experiment/build')\nfrom native_hook import on_frame\n"
out=prefix+s
needle='        gsm_node.clearTemporaryMemory()'
assert out.count(needle)==1
out=out.replace(needle,needle+"\n        on_frame(gsm_node, result_folder, f_i, data_loader, K_depth, inst_seg)")
needle='            logging.warning(f"[Skip] No segment found in frame {f_i}")'
assert out.count(needle)==1
out=out.replace(needle,needle+"\n            on_frame(gsm_node, result_folder, f_i, data_loader, K_depth, inst_seg)")
(C/'native_instrumented.py').write_text(out)
(C/'native_loop.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile='original',tofile='instrumented')))
