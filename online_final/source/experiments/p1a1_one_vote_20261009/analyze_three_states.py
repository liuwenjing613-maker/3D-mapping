"""Read-only paired audit of frozen one-vote predictions; never modifies maps."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
DEST=OUT/'three_state_analysis'
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
OLD=Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009')
CODE=Path('/home/chenkejun/CVPR/worktrees/v3-object-observed-repair-20261009')
sys.path.insert(0,str(CODE))
from unified_eval.io import load_gt,load_prediction
from unified_eval.metrics import build_overlap,owner_coverage
from unified_eval.schema import Protocol

STATE_NAMES=('unobserved','tentative','assigned_or_confirmed','conflict')
QUALITY_NAMES=('no_geometry','unassigned_with_geometry','correct_owner','wrong_owner')
POLICIES=('original','abstain','depth')
hashes={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def record(path,expected=None):
    digest=sha(path)
    if expected is not None:assert digest==expected,str(path)
    hashes[str(path)]=digest

def read(path):
    record(path)
    return json.loads(Path(path).read_text())

def arrays(path):
    record(path)
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def counts(values,n=4):return np.bincount(values.astype(np.int64),minlength=n).astype(int).tolist()

def transition(before,after,mask=None):
    if mask is not None:before,after=before[mask],after[mask]
    matrix=np.bincount(before.astype(np.int64)*4+after,minlength=16).reshape(4,4)
    assert int(matrix.sum())==len(before)
    assert np.array_equal(matrix.sum(axis=1),np.bincount(before,minlength=4))
    assert np.array_equal(matrix.sum(axis=0),np.bincount(after,minlength=4))
    return matrix.astype(int).tolist()

def sparse(matrix,names):
    return {names[i]+' -> '+names[j]:int(matrix[i][j]) for i in range(4) for j in range(4)
            if i!=j and matrix[i][j]}

def stats(evidence,indices):
    if not np.any(indices):return {'count':0}
    return {'count':int(indices.sum()),
            **{key+'_median':float(np.median(evidence[key][indices])) for key in
               ('top1_votes','top2_votes','total_frame_votes','confidence','margin')}}

def quality(scene,policy,stage,gt,native,nearest,protocol,expected):
    path=OUT/'evaluation'/scene/(policy+'_'+stage)/'canonical_prediction.npz'
    record(path)
    pred=load_prediction(path)
    overlap=build_overlap(gt,pred,protocol)
    scoring=owner_coverage(overlap)
    alignment={str(r['instance_uid']):int(r['raw_gt_id']) for r in scoring['scoring_owner_alignment']}
    ref=np.full(gt.vertex_count,-1,np.int64)
    mapped_gt=np.full(gt.vertex_count,-1,np.int64)
    for instance in pred.instances:
        indices=instance.vertex_indices
        assert np.all(ref[indices]<0),'Non-partition'
        identity=int(instance.metadata['native_instance_id'])
        assert identity==int(instance.instance_uid)
        ref[indices]=identity
        mapped_gt[indices]=alignment.get(str(instance.instance_uid),-1)
    matched=nearest>=0
    reconstructed=np.full(gt.vertex_count,-1,np.int64)
    reconstructed[matched]=native[nearest[matched]]
    reconstructed[reconstructed<=0]=-1
    np.testing.assert_array_equal(ref,reconstructed)
    target=overlap.gt_object_mask
    status=np.zeros(gt.vertex_count,np.uint8)
    status[target&matched&(ref<=0)]=1
    status[target&matched&(ref>0)]=3
    status[target&matched&(ref>0)&(mapped_gt==gt.instance_id)]=2
    c=counts(status[target])
    for key,value in [('correct_owner_vertices',c[2]),('wrong_owner_vertices',c[3]),
                      ('unassigned_with_geometry_target_vertices',c[1])]:
        assert value==expected[key],(scene,policy,stage,key,value,expected[key])
    assert c[2]==scoring['correct_owner_vertices'] and c[3]==scoring['wrong_owner_vertices']
    return {'status':status,'target':target,'ref_identity':ref,'alignment':alignment,
            'counts':c,'scoring':scoring}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),'Audit output already exists; preserve it'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    assert not DEST.exists(),'Use existing audited results instead of overwriting'
    freeze=read(OUT/'predictions_freeze.json')
    comparison=read(OUT/'comparison.json')
    config=read(CODE/'unified_eval/configs/replica_ca_v3.object_observed_repair.development.json')
    assert comparison['status']=='PASS' and comparison['baseline_modified'] is False
    assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT'
    assert config['frozen'] is False
    protocol=Protocol.from_dict(config)
    for path,digest in freeze['code_sha256'].items():record(path,digest)
    record(__file__)
    result={'status':'PASS','baseline_modified':False,'mapping_rerun':False,'GT_used_only_for_posthoc_scoring':True,
            'protocol_status':comparison['protocol_status'],
            'state_order':list(STATE_NAMES),'quality_order':list(QUALITY_NAMES),
            'native_definition':'Raw state in surface_evidence; assigned_or_confirmed means state=2',
            'final_definition':'Final label >0 is assigned_or_confirmed; otherwise use this policy own native state 0/1/3',
            'unobserved_caveat':'Zero effective identity votes; after abstention this does not mean no raw observations',
            'quality_definition':'Unchanged v3 maximum-intersection one-to-one scoring alignment; fixed GT scope and full-geometry correspondence',
            'decisions':[
                'Use previous final-unassigned three-state PLY definition, and also audit native state transitions.',
                'Compute exact paired per-point matrices, so conflict-to-tentative/unobserved cannot be mistaken for identity recovery.',
                'Reuse all frozen predictions, settings and v3 scope; no threshold search or new predictions.',
                'Recompute official owner alignment only for posthoc scoring; also report whether alignment changed.',
                'Count TSDF points and GT reference vertices separately; never equate their denominators.',
                'Interpret zero-vote points after abstention as no effective vote, preserving raw observation provenance.'
            ],'scenes':{}}
    native_matrices={p:np.zeros((4,4),np.int64) for p in POLICIES[1:]}
    final_matrices={p:np.zeros((4,4),np.int64) for p in POLICIES[1:]}
    quality_matrices={stage:{p:np.zeros((4,4),np.int64) for p in POLICIES[1:]} for stage in ('native','final')}
    for scene in ('room0','room2'):
        print('AUDIT_START',scene,flush=True)
        mapping=read(OUT/'ablation'/scene/'mapping_complete.json')
        assert sha(OUT/'ablation'/scene/'mapping_complete.json')==freeze['scenes'][scene]['mapping_complete_sha256']
        ev_path=BASE/'native'/scene/'P1-A1/surface_p0/surface_evidence.npz'
        record(ev_path,mapping['source_sha256'][str(ev_path)])
        gt_path=OLD/'mesh'/scene/'gt.npz';record(gt_path);gt=load_gt(gt_path)
        cor=arrays(gt_path.parent/'fixed_gt_to_tsdf.npz');nearest=cor['nearest_native_index']
        collision=arrays(OUT/'ablation'/scene/'collision_point_counts.npz')['colliding_frame_count']>0
        masks={};evidence={};labels={};qualities={};summaries={}
        for policy in POLICIES:
            evidence[policy]=arrays(ev_path if policy=='original' else OUT/'ablation'/scene/policy/'surface_evidence.npz')
            s=evidence[policy]['state']
            assert np.all(np.isin(s,[0,1,2,3]))
            np.testing.assert_array_equal(evidence[policy]['xyz_m'],evidence['original']['xyz_m'])
            assert int((s==0).sum())==mapping['mapping_summaries'][policy]['state_counts']['unobserved']
            assert np.all((s==0)==(evidence[policy]['total_frame_votes']==0))
            labels[policy]={};qualities[policy]={};masks[policy]={}
            for stage in ('native','final'):
                rec=freeze['scenes'][scene]['predictions'][policy+'_'+stage]
                record(rec['path'],rec['sha256'])
                pred=arrays(rec['path'])
                np.testing.assert_array_equal(pred['xyz_m'],evidence['original']['xyz_m'])
                lab=pred['instance_id'];labels[policy][stage]=lab
                if stage=='native':
                    np.testing.assert_array_equal(lab,np.where(s==2,evidence[policy]['top1_instance_id'],-1))
                    masks[policy][stage]=s.copy()
                else:
                    assert np.all(s[lab<=0]!=2),'Confirmed point lost in final'
                    masks[policy][stage]=np.where(lab>0,2,s).astype(np.uint8)
                qualities[policy][stage]=quality(scene,policy,stage,gt,lab,nearest,protocol,
                    comparison['per_scene'][scene][policy+'_'+stage])
            summaries[policy]={'native_counts':counts(s),'final_counts':counts(masks[policy]['final']),
                'native_three_state_total':int((s!=2).sum()),
                'final_three_state_total':int((labels[policy]['final']<=0).sum()),
                'postprocessing_assigned_by_native_state':counts(s[labels[policy]['final']>0]),
                'quality_native_counts':qualities[policy]['native']['counts'],
                'quality_final_counts':qualities[policy]['final']['counts']}
        if scene=='room0':assert summaries['original']['final_counts']==[75821,10081,1487655,59477]
        oldstate=evidence['original']['state']
        collstate=counts(oldstate[collision]);conf=int((oldstate==3).sum())
        rec={'surface_points':len(oldstate),'policies':summaries,
            'original_collision_affected_native_counts':collstate,
            'original_conflict_without_any_same_frame_collision':conf-collstate[3],
            'original_conflict_without_same_frame_collision_fraction':(conf-collstate[3])/conf,
            'paired_vs_original':{}}
        for policy in POLICIES[1:]:
            m1=transition(oldstate,evidence[policy]['state'])
            m2=transition(masks['original']['final'],masks[policy]['final'])
            native_matrices[policy]+=np.asarray(m1);final_matrices[policy]+=np.asarray(m2)
            assert np.all(collision[evidence[policy]['state']!=oldstate]),'State changes outside collisions'
            paired={'native_matrix':m1,'native_flows':sparse(m1,STATE_NAMES),
                'final_matrix':m2,'final_flows':sparse(m2,STATE_NAMES),
                'native_same_state_identity_changes':int(np.count_nonzero((oldstate==2)&(evidence[policy]['state']==2)&
                    (labels['original']['native']!=labels[policy]['native']))),
                'final_assigned_identity_changes':int(np.count_nonzero((labels['original']['final']>0)&
                    (labels[policy]['final']>0)&(labels['original']['final']!=labels[policy]['final']))),
                'final_three_state_exits':int(np.count_nonzero((labels['original']['final']<=0)&(labels[policy]['final']>0))),
                'final_three_state_entries':int(np.count_nonzero((labels['original']['final']>0)&(labels[policy]['final']<=0))),
                'new_zero_effective_vote_points':int(np.count_nonzero((oldstate!=0)&(evidence[policy]['state']==0))),
                'quality':{}}
            for stage in ('native','final'):
                qold=qualities['original'][stage];qnew=qualities[policy][stage];target=qold['target']
                assert np.array_equal(target,qnew['target'])
                mat=transition(qold['status'],qnew['status'],target)
                quality_matrices[stage][policy]+=np.asarray(mat)
                alignmentchanges={uid:[qold['alignment'].get(uid),qnew['alignment'].get(uid)]
                    for uid in sorted(set(qold['alignment'])|set(qnew['alignment']))
                    if qold['alignment'].get(uid)!=qnew['alignment'].get(uid)}
                grouped={}
                for state in (0,1,3):
                    cohort=target&(nearest>=0)
                    cohort[cohort]=masks['original'][stage][nearest[cohort]]==state
                    assert np.all(qold['status'][cohort]==1)
                    grouped[STATE_NAMES[state]]={'target_reference_vertices':int(cohort.sum()),
                        'after_quality_counts':counts(qnew['status'][cohort]),
                        'after_policy_state_counts':counts(masks[policy][stage][nearest[cohort]])}
                paired['quality'][stage]={'matrix':mat,'flows':sparse(mat,QUALITY_NAMES),
                    'owner_alignment_changes':alignmentchanges,'original_three_state_cohorts':grouped,
                    'target_reference_vertices_with_different_identity':int(np.count_nonzero(target&(qold['ref_identity']!=qnew['ref_identity'])))}
            native_gain=(oldstate==3)&(evidence[policy]['state']==2)
            paired['native_conflict_to_confirmed_evidence']={'before':stats(evidence['original'],native_gain),
                'after':stats(evidence[policy],native_gain),
                'top1_identity_changed_points':int(np.count_nonzero(native_gain&
                    (evidence['original']['top1_instance_id']!=evidence[policy]['top1_instance_id'])))}
            rec['paired_vs_original'][policy]=paired
        result['scenes'][scene]=rec
        print('AUDIT_COMPLETE',scene,json.dumps({'counts':summaries,
            'flows':{p:rec['paired_vs_original'][p]['final_flows'] for p in POLICIES[1:]}}),flush=True)
    result['pooled']={'native_matrices':{p:m.astype(int).tolist() for p,m in native_matrices.items()},
        'final_matrices':{p:m.astype(int).tolist() for p,m in final_matrices.items()},
        'quality_matrices':{stage:{p:m.astype(int).tolist() for p,m in rows.items()} for stage,rows in quality_matrices.items()}}
    for path,digest in hashes.items():assert sha(path)==digest,'Input modified during audit: '+path
    result['input_hashes_verified_before_and_after']=hashes
    dump(DEST/'three_state_analysis.json',result)
    print('PASS',json.dumps({'output':str(DEST/'three_state_analysis.json'),'verified_inputs':len(hashes)}),flush=True)

if __name__=='__main__':main()
