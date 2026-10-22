"""Actual attempted reads verify the input fence, not just manifest claims."""
from online_common import *
import sitecustomize

m=read(R/'inputs_manifest.json')
last=next(x for x in m['files'] if x['frame']==395 and x['kind']=='rgb')
assert Path(last['source']).open('rb').read(1)
checks=[]
for name,path,cutoff in [
    ('future_after_trigger',last['source'].replace('frame000395','frame000400'),395),
    ('future_during_increment',last['source'],240),
    ('complete_sequence_baseline','/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room0/abstain/final/instance_surface.npz',395),
    ('GT','/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/ground_truth/gt.npz',395),
]:
    sitecustomize.current_cutoff=cutoff
    try:Path(path).open('rb')
    except PermissionError as error:
        assert 'ONLINE_INPUT_FENCE' in str(error)
        checks.append(dict(test=name,blocked=True))
    else:raise AssertionError('Input guard did not reject '+name)
sitecustomize.current_cutoff=395
write(R/'guard_test.json',dict(status='PASS',checks=checks))
print('INPUT_FENCE_TESTS_PASS',len(checks))
