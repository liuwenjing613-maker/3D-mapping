"""Reuse only an entire identical tracking case; never slice a multi-object track."""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
from PIL import Image
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/e1_auto35_original_20261019')
SOURCES=[Path('/data/chenkejun/CVPR/results')/n for n in ['e1_user28_original_20261017_r67','e1_auto90_repair_20261015']]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def signature(cfg,seed_sha):
    # These are all variable inputs read by the unchanged track_case.py.
    return json.dumps({k:cfg[k] for k in ['scene','experiment_mode','seed_mode','seed_frame','objects']}|{'seed_sha256':seed_sha,'tracking':{k:cfg['tracking'][k] for k in ['checkpoint','checkpoint_sha256','repo','python','directions']}},sort_keys=True)

def main():
    started=time.time();freeze=read(R/'input_freeze.json');code_sha=sha(C/'track_case.py');cache={};sources=[]
    assert code_sha==freeze['tracking_code_sha256']
    for source in SOURCES:
        old=read(source/'input_freeze.json')
        assert old['raw_rgb_hashes']==freeze['raw_rgb_hashes'],'Tracking RGB inputs changed'
        assert old['tracking_code_sha256']==code_sha==sha(old['tracking_code_path'])
        for row in read(source/'seed_diagnostics.json')['cases']:
            p=source/'cases'/row['case_uid']/'tracking/complete.json'
            report=read(p);cfg=read(row['config_path']);seed=source/'cases'/row['case_uid']/'prepared_seed.png'
            assert report['status']=='PASS' and report['completed_frames']==report['total_frames']==2000
            assert report['code_sha256']==code_sha and report['case_config_sha256']==sha(row['config_path'])
            assert report['seed_sha256']==sha(seed) and report['track_ids']==[x['track_id'] for x in cfg['objects']]
            assert report['checkpoint_sha256']==cfg['tracking']['checkpoint_sha256']
            assert report['directions_use_independent_seed_only_states'] and not report['GT_used']
            cache.setdefault(signature(cfg,report['seed_sha256']),(p,report,cfg,seed))
        sources.append(dict(directory=str(source),input_freeze_sha256=sha(source/'input_freeze.json')))
    reused=[];pending=[]
    for row in read(R/'seed_diagnostics.json')['cases']:
        uid=row['case_uid'];cfg=read(row['config_path']);seed=R/'cases'/uid/'prepared_seed.png';key=signature(cfg,sha(seed))
        if key not in cache:pending.append(uid);continue
        p,source_report,oldcfg,oldseed=cache[key]
        assert signature(oldcfg,sha(oldseed))==key
        np.testing.assert_array_equal(np.asarray(Image.open(seed)),np.asarray(Image.open(oldseed)))
        assert [x['frame'] for x in source_report['frames']]==list(range(2000))
        for frame in source_report['frames']:
            assert sha(frame['label_file'])==frame['label_sha256']
        seed_frame=source_report['frames'][cfg['seed_frame']]
        np.testing.assert_array_equal(np.asarray(Image.open(seed_frame['label_file'])),np.asarray(Image.open(seed)))
        out=R/'cases'/uid/'tracking';out.mkdir(exist_ok=False)
        report=copy.deepcopy(source_report)
        report.update(case_uid=uid,case_config_sha256=sha(row['config_path']),seed_sha256=sha(seed),
            execution='REUSED_IDENTICAL_WHOLE_CASE_NOT_NEW_INFERENCE',source_report=str(p),source_report_sha256=sha(p),
            source_case_config_sha256=source_report['case_config_sha256'],source_case_uid=source_report['case_uid'],
            reuse_verified_inputs=['complete seed label image','ordered target IDs','source frame','2000 RGB hashes','tracking code','checkpoint','tracking settings'],
            GPU_visible='cache; original '+str(source_report.get('GPU_visible')))
        save(out/'complete.json',report);save(out/'status.json',{k:v for k,v in report.items() if k!='frames'})
        reused.append(dict(case_uid=uid,source_report=str(p),source_report_sha256=sha(p),targets=report['track_ids'],frames=2000))
        print('REUSED',uid,report['track_ids'],flush=True)
    result=dict(status='PASS',reused_cases=len(reused),new_inference_cases=len(pending),reused=reused,pending=pending,
                sources=sources,partial_multiobject_reuse=False,GT_used=False,seconds=time.time()-started)
    save(R/'tracking_cache_reuse.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['reused','sources']}),flush=True)
if __name__=='__main__':main()
