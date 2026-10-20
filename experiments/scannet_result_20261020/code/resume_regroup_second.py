from pathlib import Path
import sys,json,time,subprocess,traceback
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
s='scene0608_00'
sys.path.insert(0,str(C/s))
from run_v1 import run,check
sys.path.insert(0,str(C))
from controller import stage,save,main
try:
    save(R/'status.json',dict(status='RUNNING',scene=s,stage='approved_regroup_resume',time=time.time()))
    save(R/s/'status.json',dict(status='RUNNING',stage='approved_regroup_resume',time=time.time()))
    subprocess.run([sys.executable,'-B',str(C/s/'test_regroup_policy.py')],check=True)
    for name in ['grouping_r3','v1']:
        check(run(name,'compute.py'))
        audit=json.loads((R/s/name/'audit.json').read_text())
        assert audit['status']=='PASS'
        if name=='grouping_r3':
            src=json.loads((R/s/name/'semantic_exclusions.json').read_text())
            assert {e['id'] for e in src['post_crop_exclusions'] if e['entire_mask_excluded']}=={'M0138','M0141','M0163'}
            assert audit['stats']['unique_masks']==432
    save(R/s/'v1_complete.json',dict(status='PASS',stats=audit['stats'],time=time.time()))
    save(R/s/'v1_status.json',dict(stage='complete',time=time.time()))
    save(R/s/'upstream_stage_complete.json',dict(status='PASS',time=time.time(),resumed_after_approved_source_regroup=True))
    stage(s,'E1',[C/s/'run_remaining.py'],C/s)
    stage(s,'partial',[C/s/'partial/controller.py'],C/s/'partial')
    main()
    subprocess.run([sys.executable,'-B',str(C/'export.py')],check=True)
    save(R/'regroup_resume_complete.json',dict(status='PASS',time=time.time()))
except Exception:
    save(R/'regroup_resume_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),time=time.time()))
    raise
