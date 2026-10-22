from pathlib import Path
import unittest,json,sys,hashlib
import test_strict_repair,test_history_core,test_joint_batch_ops
R=Path('/data/chenkejun/CVPR/results/mask_gate_no_foreign_v2_20261010/room0')
for module,folder,expected in [(test_strict_repair,'original_repair',10),(test_history_core,'modified_repair',8),(test_joint_batch_ops,'batch_tests',None)]:
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
 assert result.wasSuccessful()
 if expected is not None:assert result.testsRun==expected
 p=R/folder/'tests_complete.json';p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps({'status':'PASS','tests':result.testsRun,'code_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}))
