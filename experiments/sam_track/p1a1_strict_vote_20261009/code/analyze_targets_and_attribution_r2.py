"""Correct legacy target IDs by exact fixed-reference lineage; audit scoring alignment."""
from pathlib import Path
import json,sys,time
import numpy as np

SOURCE=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
ROOT=SOURCE/'evaluation_r2'
CODE=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(CODE))
from unified_eval.current_protocol import CURRENT_CONFIG,CURRENT_PROTOCOL,require_current_protocol,require_current_gt
from unified_eval.io import load_gt,load_prediction,sha256_file
from unified_eval.metrics import build_overlap,owner_coverage
from unified_eval.schema import Protocol

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists()
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def matrix(a,b,mask):
    return np.bincount(a[mask].astype(np.int64)*4+b[mask],minlength=16).reshape(4,4).astype(int)
def quality(gt,pred,protocol,nearest):
    overlap=build_overlap(gt,pred,protocol);score=owner_coverage(overlap)
    alignment={str(r['instance_uid']):int(r['raw_gt_id']) for r in score['scoring_owner_alignment']}
    identity=np.full(gt.vertex_count,-1,np.int64);owner=np.full(gt.vertex_count,-1,np.int64)
    for instance in pred.instances:
        v=instance.vertex_indices;assert np.all(identity[v]<0)
        identity[v]=int(instance.metadata['native_instance_id']);owner[v]=alignment.get(str(instance.instance_uid),-1)
    target=overlap.gt_object_mask;exists=nearest>=0
    q=np.zeros(gt.vertex_count,np.uint8);q[target&exists&(identity<=0)]=1;q[target&exists&(identity>0)]=3
    q[target&exists&(identity>0)&(owner==gt.instance_id)]=2
    assert np.count_nonzero(target&(q==2))==score['correct_owner_vertices']
    assert np.count_nonzero(target&(q==3))==score['wrong_owner_vertices']
    return q,identity,target,overlap.gt_ids

def main():
    assert require_current_protocol(CURRENT_CONFIG)
    complete=json.loads((ROOT/'evaluation_complete.json').read_text());assert complete['status']=='PASS' and complete['profile_revision']==2
    comparison_sha=sha256_file(ROOT/'comparison.json');assert comparison_sha==complete['comparison_sha256']
    freeze=json.loads((SOURCE/'predictions_freeze.json').read_text())
    policy=json.loads((SOURCE/'policy.json').read_text())
    protocol=Protocol.from_dict(json.loads(CURRENT_CONFIG.read_text()))
    protected={str(Path(__file__)):sha256_file(__file__),str(CURRENT_CONFIG):sha256_file(CURRENT_CONFIG)}
    results={};corrected_files={};physical={};total_pairs=0
    for scene,rec in freeze['scenes'].items():
        gtpath=Path(CURRENT_PROTOCOL['canonical_GT'][scene]['file']);assert require_current_gt(gtpath)==scene
        protected[str(gtpath)]=sha256_file(gtpath);gt=load_gt(gtpath)
        meta=gt.metadata;reference=Path(meta['source_reference'])
        assert sha256_file(reference)==meta['source_reference_sha256'];protected[str(reference)]=sha256_file(reference)
        oldsummary=Path(policy['repair_source_roots'][scene])/'evaluation_summary.json'
        oldtargets=json.loads(oldsummary.read_text())['target_GT_ids'];protected[str(oldsummary)]=sha256_file(oldsummary)
        scopepath=gtpath.parent/'gt_scope.json';scope=json.loads(scopepath.read_text());protected[str(scopepath)]=sha256_file(scopepath)
        objects={r['raw_gt_id']:r for r in scope['objects']}
        with np.load(reference) as z:
            np.testing.assert_array_equal(z['xyz'],gt.xyz_ref)
            legacy=z['instance'].copy();encoded_source=z['raw_instance'].copy()
        with np.load(gtpath) as z:raw=z['raw_instance_id'].copy()
        assert meta['source_instance_encoding']=='original_semantic_id*1000+raw_object_id'
        np.testing.assert_array_equal(encoded_source%1000,raw)
        mapping=[];mapped=[]
        for legacy_id in oldtargets:
            vertices=legacy==legacy_id;assert np.any(vertices)
            physical_ids=np.unique(raw[vertices]);assert len(physical_ids)==1,(scene,legacy_id,'Not one physical GT')
            pid=int(physical_ids[0]);assert pid in objects
            mapped.append(pid);r=objects[pid]
            mapping.append({'legacy_compact_evaluation_id':int(legacy_id),'physical_raw_GT_id':pid,
                'reference_vertices_checked':int(vertices.sum()),'evaluable_in_current_scope':r['evaluable'],
                'semantic_class':r['semantic_class']})
        assert len(mapped)==len(set(mapped)),'Selected legacy targets are not a one-to-one physical list'
        with np.load(ROOT/'geometry_cache'/scene/'fixed_gt_to_tsdf_r2.npz') as z:nearest=z['nearest_native_index'].copy()
        qualities={}
        for name,p in rec['predictions'].items():
            assert sha256_file(p['path'])==p['sha256']
            canonical=ROOT/'evaluation'/scene/name/'canonical_prediction.npz'
            manifest=json.loads((canonical.parent/'adapter_manifest.json').read_text())
            assert sha256_file(canonical)==manifest['canonical_prediction_sha256']
            protected[str(canonical)]=sha256_file(canonical)
            qualities[name]=quality(gt,load_prediction(canonical),protocol,nearest)
        qualified=set(int(x) for x in qualities['original_baseline_final'][3])
        assert qualified=={pid for pid,r in objects.items() if r['evaluable']}
        targets=set(mapped)&qualified
        results[scene]={'legacy_selected_target_ids':oldtargets,'selected_physical_raw_GT_ids':sorted(targets),
            'mapped_but_not_evaluable':[pid for pid in mapped if pid not in qualified],'reference_vertex_verified_mapping':mapping}
        for paired_path in sorted((ROOT/'evaluation'/scene).glob('*/paired_vs_*.json')):
            d=json.loads(paired_path.read_text());protected[str(paired_path)]=sha256_file(paired_path)
            after=paired_path.parent.name;semantic=after.split('_',1)[0];stage=after.rsplit('_',1)[1]
            before=semantic+'_baseline_'+stage if paired_path.name=='paired_vs_same_rule_baseline.json' else 'original'+after[len('strict'):]
            a,aid,target,_=qualities[before];b,bid,other,_=qualities[after];np.testing.assert_array_equal(target,other)
            all_matrix=matrix(a,b,target);np.testing.assert_array_equal(all_matrix,np.asarray(d['quality_transition_matrix']))
            changed=target&(aid!=bid);unchanged=target&~changed
            changed_matrix=matrix(a,b,changed);same_matrix=matrix(a,b,unchanged)
            np.testing.assert_array_equal(changed_matrix+same_matrix,all_matrix)
            for row in d['per_gt']:
                row['scene_id']=scene;row['semantic_class']=objects[row['raw_gt_id']]['semantic_class']
            d['groups']={group:{'objects':len(rr),'rows':rr} for group,rr in
                [(g,[r for r in d['per_gt'] if (r['raw_gt_id'] in targets)==pick]) for g,pick in [('selected_targets',True),('other_targets',False)]]}
            d['fixed_selected_target_GT_ids']=sorted(targets)
            d['target_definition']='Frozen legacy human-seed target list mapped one-to-one through exact same reference vertices to physical GT, intersect locked evaluable scope'
            d['target_lineage_source_sha256']=sha256_file(reference)
            d['quality_transition_on_changed_identity_vertices']=changed_matrix.tolist()
            d['quality_transition_on_unchanged_identity_vertices']=same_matrix.tolist()
            d['correct_owner_gain_at_changed_identity_vertices']=int(changed_matrix[:,2].sum()-changed_matrix[2,:].sum())
            d['correct_owner_gain_at_unchanged_identity_vertices']=int(same_matrix[:,2].sum()-same_matrix[2,:].sum())
            d['changed_identity_target_reference_vertices']=int(changed.sum())
            d['original_pair_sha256']=sha256_file(paired_path)
            d['supersedes_original_grouping_only_and_adds_attribution']=True
            output=ROOT/'analysis/corrected_pairs'/scene/after/paired_path.name
            dump(output,d);corrected_files[str(output)]=sha256_file(output);total_pairs+=1
            if after.endswith('full_track_final') and paired_path.name=='paired_vs_same_rule_baseline.json':
                physical[scene+'_'+semantic]={k:d[k] for k in ('correct_owner_gain_at_changed_identity_vertices',
                    'correct_owner_gain_at_unchanged_identity_vertices','changed_identity_target_reference_vertices',
                    'quality_transition_on_changed_identity_vertices','quality_transition_on_unchanged_identity_vertices')}
        print(json.dumps({'scene':scene,'selected_target_objects':len(targets),'legacy_ids_mapped_exactly':len(mapping)}),flush=True)
    assert total_pairs==28
    assert sha256_file(ROOT/'comparison.json')==comparison_sha
    for path,digest in protected.items():assert sha256_file(path)==digest
    for rec in freeze['scenes'].values():
        for p in rec['predictions'].values():assert sha256_file(p['path'])==p['sha256']
    result={'status':'PASS','profile_revision':2,'GT_used_to_change_repairs':False,'main_metrics_sha256_unchanged':comparison_sha,
        'diagnostic_correction':'Initial grouping compared compact legacy evaluation IDs directly with physical R2 IDs. Corrected groups use verified reference-vertex lineage, no modulo guess on compact IDs. Main metrics and predictions were correct and unchanged.',
        'corrected_pairs':total_pairs,'scenes':results,'full_track_physical_attribution':physical,
        'protected_sha256':protected,'corrected_pair_sha256':corrected_files,'created_unix':time.time()}
    dump(ROOT/'analysis/targets_and_attribution_complete.json',result)
    print(json.dumps({'status':'PASS','corrected_pairs':total_pairs,'full_track_physical_attribution':physical}),flush=True)

if __name__=='__main__':main()
