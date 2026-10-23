"""Run the existing continuation main with a one-frame range and the new commit pointer."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
CODE=REPO/'experiments/online_room0_repair50_to100_20261022/continuation300'
BENCH=REPO/'experiments/online_checkpoint_surface_20261023'
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
MAPS=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['reference','checkpoint'])
    p.add_argument('stage',choices=['setup','run'])
    args=p.parse_args()
    root=OUT/args.mode/'frame201'
    os.environ.update(CVPR_CHECKPOINT='300',CVPR_TRACK_GROUP='A',CVPR_MAPPING_RUN_ROOT=str(root),
                      CVPR_SURFACE_POLICY='every_frame' if args.mode=='reference' else 'checkpoint')
    sys.path.insert(0,str(CODE))
    import map_support as support
    # Only the range and prior pointer are adapted. Neither main nor association is replaced.
    support.TARGET=201
    support.PRIOR_COUNT=200
    support.PRIOR_RAW=995
    support.FIDS=tuple(range(0,1001,5))
    support.CUTOFF=1000
    pointer=OUT/args.mode/'repair/commit_complete.json'
    previous=MAPS/(args.mode+'_200_A')/'mapping'
    support.prior_paths=lambda:(pointer,previous)
    original=BENCH/'reference/map_continue.py'
    source=original if args.mode=='reference' else CODE/'map_continue.py'
    if args.stage=='setup':
        assert not root.exists(), 'Use a fresh frame-201 output'
        prior=support.setup()
        assert prior['processed_frames']==200 and prior['last_raw_frame']==995
        support.write(root/'probe_provenance.json',dict(status='READY',source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),main_function_unchanged=True,
            only_overrides=['TARGET=201','PRIOR_COUNT=200','FIDS=0..1000 stride5','new repair200 commit pointer'],
            committed_state=str(pointer),committed_state_sha256=support.sha(pointer)))
    else:
        assert os.environ.get('CVPR_ONLINE_GUARD')==str(root/'mapping/input_guard.json')
        import sitecustomize
        assert sitecustomize.current_cutoff==995
        runpy.run_path(str(source),run_name='__main__')

if __name__=='__main__':main()
