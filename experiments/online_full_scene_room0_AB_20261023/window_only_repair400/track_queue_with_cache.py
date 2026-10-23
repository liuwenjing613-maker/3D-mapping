"""Use the existing exact tracking fingerprint and unchanged reuse/remap function."""
from common_continue import *
import threading
import track_queue as queue

AUDIT=[]
AUDIT_LOCK=threading.Lock()
SOURCES=[Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')/'400'/g/'tracking' for g in [GROUP,('B' if GROUP=='A' else 'A')]]+[ROOT.parent/('B' if GROUP=='A' else 'A')/'tracking']

def prior_case(case):
    data=case['fingerprint_data']
    computed=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert computed==case['fingerprint']
    for peer in SOURCES:
        path=peer/'seed_diagnostics.json'
        if not path.exists():continue
        for other in read(path)['cases']:
            if other['fingerprint']!=case['fingerprint']:continue
            assert other['fingerprint_data']==data
            complete=peer/'cases'/other['case_uid']/'tracking/complete.json'
            if not complete.exists():continue
            old=read(complete)
            if old['status']!='PASS':continue
            assert old['completed_frames']==old['total_frames']==RAW_COUNT
            assert sorted(row['frame'] for row in old['frames'])==list(range(RAW_COUNT))
            assert old['GT_used'] is False
            assert old['code_sha256']==data['tracking_code_sha256']==digest(C/'track_case.py')
            assert old['case_config_sha256']==digest(other['config_path'])
            seed=peer/'cases'/other['case_uid']/'prepared_seed.png'
            assert old['seed_sha256']==other['seed_sha256']==digest(seed)
            with AUDIT_LOCK:
                AUDIT.append(dict(case_uid=case['case_uid'],source=str(complete),source_sha256=digest(complete),
                                  fingerprint=case['fingerprint'],entire_fingerprint_data_exact=True,
                                  cached_config_seed_code_hashes_verified=True))
            return peer,other,complete
    return None

if __name__=='__main__':
    assert CHECKPOINT in (200,300,400) and GROUP in ('A','B')
    assert digest('/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt')=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
    queue.prior_case=prior_case
    queue.main()
    dump(R/'cache_fingerprint_audit.json',dict(status='PASS',sources=list(map(str,SOURCES)),matches=AUDIT,
        matching_rule='Existing complete fingerprint_data equality; unchanged reuse verifies every label hash and target seed',
        model_weight_hash_verified=True))
