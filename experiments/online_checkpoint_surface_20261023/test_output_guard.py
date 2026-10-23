"""Exercise output protection against real completed outputs without writing them."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')
CODE=Path(os.environ.get('CVPR_TEST_CODE',str(HERE.parents[1]/'experiments/online_room0_repair50_to100_20261022/continuation300')))
SUPPORT=CODE/'map_support.py'

def load_support(root):
    env=dict(os.environ,CVPR_CHECKPOINT='200',CVPR_TRACK_GROUP='A')
    env.pop('CVPR_MAPPING_RUN_ROOT',None)
    if root is not None:env['CVPR_MAPPING_RUN_ROOT']=str(root)
    with patch.dict(os.environ,env,clear=True):
        spec=importlib.util.spec_from_file_location('candidate_support',SUPPORT)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module

class OutputGuardTests(unittest.TestCase):
    def test_requires_explicit_output(self):
        with self.assertRaisesRegex(ValueError,'fresh directory'):load_support(None)

    def test_forbids_external_and_bare_data_root(self):
        for root in ['/tmp/unrelated_output','/data/chenkejun/CVPR']:
            with self.subTest(root=root),self.assertRaisesRegex(ValueError,'subdirectory'):
                load_support(root)

    def test_protects_archive_even_with_different_checkpoint_or_path_alias(self):
        archive='/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022'
        for root in [archive,archive+'/continuation300/300/B',archive+'/continuation300/300/B/../A']:
            with self.subTest(root=root),self.assertRaisesRegex(ValueError,'archived experiment'):
                load_support(root)

    def test_import_of_new_output_does_not_create_it(self):
        root=OUT/'unused_guard_test_output'
        self.assertFalse(root.exists())
        module=load_support(root)
        self.assertEqual(module.R,root/'mapping')
        self.assertFalse(root.exists())

    def test_completed_setup_is_read_only(self):
        root=OUT/'checkpoint_200_A'
        module=load_support(root)
        files=[root/'mapping'/name for name in ['config.json','prior_commit.json','input_guard.json','mapping_complete.json']]
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        with self.assertRaisesRegex(FileExistsError,'immutable'):module.setup()
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.assertEqual(before,after)

    def test_failed_rerun_does_not_mark_completed_output_failed(self):
        root=OUT/'checkpoint_200_A'
        failure=root/'mapping/mapping_failure.json'
        self.assertFalse(failure.exists())
        protected=root/'mapping/mapping_complete.json'
        before=protected.read_bytes()
        guard=HERE.parents[1]/'experiments/online_room0_repair50_to100_20261022/continuation300/map_guard'
        env=dict(os.environ,CVPR_CHECKPOINT='200',CVPR_TRACK_GROUP='A',
            CVPR_MAPPING_RUN_ROOT=str(root),CVPR_SURFACE_POLICY='invalid',
            PYTHONPATH=str(guard),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        env.pop('CVPR_ONLINE_GUARD',None)
        proc=subprocess.run([sys.executable,'-B',str(CODE/'map_continue.py')],env=env,capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('CVPR_SURFACE_POLICY must be',proc.stderr)
        self.assertFalse(failure.exists())
        self.assertEqual(before,protected.read_bytes())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OutputGuardTests))
    (OUT/'output_guard_tests.json').write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAILED',
        code_directory=str(CODE),support_sha256=hashlib.sha256(SUPPORT.read_bytes()).hexdigest(),
        tests=result.testsRun,errors=len(result.errors),failures=len(result.failures)),indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
