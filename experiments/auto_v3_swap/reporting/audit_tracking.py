"""Measure retained-mask differences when removal changes a whole tracking case."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
C=Path('/home/chenkejun/CVPR/experiments/replica8_swap34_20261020')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for scene,cfg in read(C/'config.json').items():
    if cfg['action']!='rerun':continue
    root=R/scene;out=root/'tracking_difference_audit.json';new=root/'e1_repair';old=Path(cfg['old_E1'])
    if out.exists():continue
    reuse=new/'tracking_cache_reuse.json'
    if not reuse.exists():continue
    cache=read(reuse);pending=cache['pending']
    reports=[new/'cases'/uid/'tracking/complete.json' for uid in pending]
    if not all(p.exists() for p in reports):continue
    old_by_track={}
    for row in read(old/'seed_diagnostics.json')['cases']:
        config=read(row['config_path']);p=old/'cases'/row['case_uid']/'tracking/complete.json'
        for obj in config['objects']:old_by_track[obj['track_id']]=p
    cases=[]
    for p in reports:
        report=read(p);assert report['status']=='PASS'
        before={t:read(old_by_track[t]) for t in report['track_ids']}
        count={t:dict(track_id=t,changed_frames=0,total_xor_pixels=0,max_xor_pixels=0,
             previous_case_track_ids=before[t]['track_ids']) for t in report['track_ids']}
        assert len(report['frames'])==2000
        for i,frame in enumerate(report['frames']):
            assert frame['frame']==i and sha(frame['label_file'])==frame['label_sha256']
            current=np.asarray(Image.open(frame['label_file']))
            old_arrays={}
            for t,prior in before.items():
                ref=prior['frames'][i];path=ref['label_file']
                if path not in old_arrays:
                    assert ref['frame']==i and sha(path)==ref['label_sha256']
                    old_arrays[path]=np.asarray(Image.open(path))
                oldmask=old_arrays[path]==t;newmask=current==t
                pixels=int(np.count_nonzero(oldmask^newmask))
                count[t]['changed_frames']+=int(pixels>0)
                count[t]['total_xor_pixels']+=pixels
                count[t]['max_xor_pixels']=max(count[t]['max_xor_pixels'],pixels)
        cases.append(dict(case_uid=report['case_uid'],track_ids=report['track_ids'],tracks=list(count.values()),
            source_report_sha256=sha(p),previous_report_sha256={str(old_by_track[t]):sha(old_by_track[t]) for t in report['track_ids']}))
    audit=dict(status='PASS',scene=scene,reused_cases=cache['reused_cases'],new_inference_cases=len(cases),
        cases=cases,only_entire_identical_tracking_cases_reused=True,GT_used=False)
    out.write_text(json.dumps(audit,indent=2));print('TRACKING_DIFFERENCES',json.dumps(audit),flush=True)
