"""Resume a CUDA-initialization-only failure without changing model inputs."""
from online_common import *
from run_selection import run, launch, CROP
import time

run('cropformer_retry','frontend/infer_cropformer.py',python=CROP,
    device='GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129')
deadline=time.time()+3600
while not (R/'frontend/sam2_1/audit.json').exists():
    assert time.time()<deadline, 'SAM recovery wait exceeded'
    time.sleep(5)
assert read(R/'frontend/sam2_1/audit.json')['status']=='PASS'
assert read(R/'frontend/cropformer/audit.json')['status']=='PASS'
while not (R/'selection_failure.json').exists():
    assert time.time()<deadline
    time.sleep(2)
(R/'selection_failure.json').rename(R/'selection_models_first_attempt_failure.json')
write(R/'model_recovery.json',dict(status='PASS',reason='initial CropFormer GPU CUDA initialization failed',
    retry_device='GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129',model_weights_unchanged=True,input_crops_unchanged=True))
run('selection_resume','run_selection.py',['--start','summary'])
