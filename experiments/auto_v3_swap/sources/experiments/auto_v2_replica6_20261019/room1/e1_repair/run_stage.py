from bootstrap import C,R,F,load_script,verify_sources,digest
from pathlib import Path
import json,sys,unittest,time,traceback
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def tests(names,out):
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
 with (R/(out.replace('/','_')+'_tests.log')).open('w') as stream:res=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 assert res.wasSuccessful(),(names,res.errors,res.failures)
 dump(R/out/'tests_complete.json',{'status':'PASS','tests':res.testsRun,'modules':names})
 print('TESTS',out,res.testsRun,flush=True)
def main(stage):
 verify_sources()
 if stage=='tests':
  tests(['test_strict_repair'],'original_repair')
  tests(['test_history_core'],'modified_repair')
  tests(['test_observation_ops','test_adaptive_core'],'e1')
  tests(['test_gate_v1'],'gate_tests')
  dump(R/'tests_complete.json',{'status':'PASS','sets':['original_repair','modified_repair','e1','gate_tests'],'frozen_sources_verified':True})
 elif stage=='identity':load_script('_auto_batch',C/'run_batch.py').main(identity_only=True)
 elif stage=='raw_repair':load_script('_auto_batch',C/'run_batch.py').main()
 elif stage=='strict_repair':load_script('_auto_strict',C/'run_strict_repair.py').main()
 elif stage=='history_core':
  import run_history_core as h
  h.main()
 elif stage=='expand':load_script('_auto_expand',C/'expand_room1.py').main()
 elif stage=='E1':load_script('_auto_E1',C/'run_observation_repair.py').main()
 else:raise ValueError(stage)
 verify_sources()
 dump(R/('stage_'+stage+'_complete.json'),{'status':'PASS','stage':stage,'unix':time.time()})
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception:dump(R/('stage_'+sys.argv[1]+'_failure.json'),{'traceback':traceback.format_exc()});raise
