"""Independent full-frame accumulation versus materialized delta vote tables."""
from pathlib import Path
import hashlib,json,time
import numpy as np

ROOT=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    freeze=json.loads((ROOT/'predictions_freeze.json').read_text())
    assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    results=[];artifacts={}
    for scene in ('room0','room2'):
        started=time.monotonic()
        native=BASE/f'native/{scene}/P1-A1/surface_p0'
        report=json.loads((native/'materialization_report.json').read_text())
        assigned={}
        for line in (BASE/f'native/{scene}/P1-A1/association/associations.jsonl').read_text().splitlines():
            r=json.loads(line);assigned[(r['frame_id'],r['mask_local_id'])]=r['instance_id']
        base=max(assigned.values())+1
        original=read_npz(native/'surface_instance_frame_votes.npz')
        master=original['surface_point_index'].astype(np.int64)*base+original['instance_id']
        order=np.argsort(master);master=master[order]
        full={p:np.zeros(len(master),np.int64) for p in ('original','abstain','depth')}
        for record in report['frame_records']:
            fid=record['frame_id'];path=Path(record['support_file'])
            assert sha(path)==record['support_sha256']
            raw=read_npz(path);points=raw['surface_point_index'];local=raw['mask_local_id']
            table=np.array([assigned.get((fid,i),-1) for i in range(int(local.max())+1)],np.int64)
            keys=np.unique(points.astype(np.int64)*base+table[local])
            point=keys//base
            starts=np.r_[0,np.flatnonzero(point[1:]!=point[:-1])+1]
            lengths=np.diff(np.r_[starts,len(keys)])
            single=keys[starts[lengths==1]]
            collision_points=point[starts[lengths>1]]
            preference_path=ROOT/'ablation'/scene/'collision_frames'/f'f{fid:06d}.npz'
            artifacts[str(preference_path)]=sha(preference_path)
            preference=read_npz(preference_path)
            np.testing.assert_array_equal(collision_points,preference['colliding_surface_point_index'])
            preferred=preference['depth_preferred_surface_point_index'].astype(np.int64)*base+preference['depth_preferred_instance_id']
            assert len(preferred)==len(np.unique(preferred//base))
            assert np.all(np.isin(preferred,keys)) and np.all(np.isin(preferred//base,collision_points))
            depth=np.union1d(single,preferred)
            for policy,frame_keys in (('original',keys),('abstain',single),('depth',depth)):
                if policy!='original':assert len(frame_keys)==len(np.unique(frame_keys//base))
                positions=np.searchsorted(master,frame_keys)
                np.testing.assert_array_equal(master[positions],frame_keys)
                full[policy][positions]+=1
        np.testing.assert_array_equal(full['original'],original['frame_votes'][order])
        policy_rows={}
        for policy in ('abstain','depth'):
            path=ROOT/'ablation'/scene/policy/'surface_instance_frame_votes.npz'
            artifacts[str(path)]=sha(path)
            pair=read_npz(path);keys=pair['surface_point_index'].astype(np.int64)*base+pair['instance_id']
            order=np.argsort(keys);keys=keys[order];votes=pair['frame_votes'][order]
            mask=full[policy]>0
            np.testing.assert_array_equal(keys,master[mask]);np.testing.assert_array_equal(votes,full[policy][mask])
            policy_rows[policy]={'full_frame_sum_exactly_equals_delta_materialization':True,'frame_point_uniqueness_verified':True,
                'effective_votes':int(votes.sum()),'vote_file_sha256':artifacts[str(path)]}
        mapping=json.loads((ROOT/'ablation'/scene/'mapping_complete.json').read_text())
        for path,digest in mapping['source_sha256'].items():assert sha(path)==digest
        results.append({'scene':scene,'frame_count':400,'policies':policy_rows,'seconds':round(time.monotonic()-started,2)})
        print(json.dumps(results[-1]),flush=True)
    assert all(sha(path)==digest for path,digest in artifacts.items())
    value={'status':'PASS','GT_read':False,'independent_reference_does_not_import_one_vote_module':True,
        'verified_actual_frames':800,'baseline_modified':False,'per_scene':results,'audited_artifact_sha256':artifacts,
        'audit_code_sha256':sha(__file__),'prediction_freeze_sha256':sha(ROOT/'predictions_freeze.json')}
    (ROOT/'runtime_equivalence.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
