"""Compare actual seeds, cached/inferred dense tracks, and fresh Gate decisions."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
from PIL import Image

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def seed_records(root,canonical=False):
    manifest=read(root/'candidates/all_edge_20261021/v2/seed_manifest.json')
    uid_to_spatial={}
    if canonical:
        catalog=read(root/'candidates/catalog/room0/candidate_manifest.json')
        membership=load(root/'candidates/catalog/room0/membership.npz')
        xyz=load(Path(catalog['surface_file']))['xyz_m']
        for c in catalog['unique_components']:
            k=c['component_index'];ix=membership['core_point_index'][membership['core_offsets'][k]:membership['core_offsets'][k+1]]
            points=xyz[ix];order=np.lexsort((points[:,2],points[:,1],points[:,0]))
            uid_to_spatial[c['uid']]=hashlib.sha256(np.ascontiguousarray(points[order]).tobytes()).hexdigest()
    rows=[]
    for g in manifest['groups']:
        p=g['primary'];path=root/'candidates/all_edge_20261021/v1'/p['mask']
        assert sha(path)==p['mask_sha256']
        rep=p['representative']
        if canonical:
            rep=json.loads(json.dumps(rep).replace(str((root/'candidates').resolve()),'<CANDIDATES>').replace(str(root/'candidates'),'<CANDIDATES>'))
            rep['uid']=uid_to_spatial[rep['uid']]
        rows.append(dict(group_id=g['id'],frame=p['frame'],mask_sha256=p['mask_sha256'],
                         mask_id=p['id'],area=p['area'],representative=rep))
    return rows

def compare(left,right):
    seeds=[seed_records(p) for p in [left,right]]
    cases=[read(p/'tracking/seed_diagnostics.json')['cases'] for p in [left,right]]
    records=[]
    def check(name,value,**extra):records.append(dict(name=name,exact=bool(value),**extra))
    check('selected_seed_raw_metadata',seeds[0]==seeds[1],counts=list(map(len,seeds)),required=False,
          note='Raw ROI UID hashes temporary point indices; source paths contain the isolated run root.')
    check('selected_seed_sources_spatial_UID_and_path_normalized',seed_records(left,True)==seed_records(right,True))
    keys=lambda s:Counter((r['frame'],r['mask_sha256']) for r in s)
    check('seed_frame_and_pixel_multiset',keys(seeds[0])==keys(seeds[1]))
    duplicates=[sum(v-1 for v in keys(s).values()) for s in seeds]
    check('no_new_exact_duplicate_seeds',duplicates[1]<=duplicates[0],duplicate_counts=duplicates)
    signature=lambda cs:[dict(case_uid=c['case_uid'],fingerprint=c['fingerprint'],data=c['fingerprint_data']) for c in cs]
    same_cases=signature(cases[0])==signature(cases[1])
    check('ordered_cases_and_full_inference_fingerprints',same_cases,counts=list(map(len,cases)))
    family=lambda cs:[(c['case_uid'],[(o['track_id'],o['old_family_id'],o['old_pixel_overlap']) for o in c['objects']]) for c in cs]
    check('source_family_id_and_pixel_overlap',family(cases[0])==family(cases[1]))
    if same_cases:
        label_count=0;differences=[];metric_differences=[]
        for a,b in zip(cases[0],cases[1]):
            x=read(left/'tracking/cases'/a['case_uid']/'tracking/complete.json')
            y=read(right/'tracking/cases'/b['case_uid']/'tracking/complete.json')
            assert [r['frame'] for r in x['frames']]==[r['frame'] for r in y['frames']]==list(range(996))
            assert [o['track_id'] for o in a['objects']]==[o['track_id'] for o in b['objects']]
            for u,v in zip(x['frames'],y['frames']):
                pa,pb=Path(u['label_file']),Path(v['label_file'])
                assert sha(pa)==u['label_sha256'] and sha(pb)==v['label_sha256']
                if u['label_sha256']!=v['label_sha256']:
                    aa,bb=np.asarray(Image.open(pa)),np.asarray(Image.open(pb))
                    if not np.array_equal(aa,bb):differences.append(dict(case=a['case_uid'],frame=u['frame'],different_pixels=int(np.count_nonzero(aa!=bb))))
                if u['areas']!=v['areas']:metric_differences.append([a['case_uid'],u['frame']])
                label_count+=1
        check('dense_SAM2_label_pixels',not differences,files=label_count,differences=differences)
        check('dense_SAM2_areas',not metric_differences,differences=metric_differences)
    gates=[read(p/'tracking/gate/frame_decisions.json') for p in [left,right]]
    decision_diff=[];metric_diff=[];mask_diff=[];reason_diff=[]
    for a,b in zip(gates[0]['frames'],gates[1]['frames']):
        assert a['frame']==b['frame'];fid=a['frame']
        if a['accepted_track_ids']!=b['accepted_track_ids']:
            decision_diff.append(dict(frame=fid,left=a['accepted_track_ids'],right=b['accepted_track_ids']))
        for field in ['objects','initial_objects']:
            decisions=lambda row:{tid:{k:obj.get(k) for k in ['accepted','reasons']} for tid,obj in row.get(field,{}).items()}
            if decisions(a)!=decisions(b):reason_diff.append(dict(frame=fid,field=field,left=decisions(a),right=decisions(b)))
        normalized=[json.loads(json.dumps(row).replace(str(root),'<RUN>')) for row,root in [(a,left),(b,right)]]
        if normalized[0]!=normalized[1]:metric_diff.append(fid)
        pa=left/'tracking/gate/accepted_masks'/f'f{fid:06d}.png'
        pb=right/'tracking/gate/accepted_masks'/f'f{fid:06d}.png'
        if sha(pa)!=sha(pb):
            aa,bb=np.asarray(Image.open(pa)),np.asarray(Image.open(pb))
            if not np.array_equal(aa,bb):mask_diff.append(dict(frame=fid,pixels=int(np.count_nonzero(aa!=bb))))
    check('Gate_accept_reject_decisions',not decision_diff,differences=decision_diff)
    check('Gate_object_decisions_and_rejection_reasons',not reason_diff,differences=reason_diff)
    check('Gate_accepted_mask_pixels',not mask_diff,differences=mask_diff)
    check('Gate_all_frame_records_including_metrics',not metric_diff,different_frames=metric_diff)
    assocs=[read(p/'tracking/global_association.json') for p in [left,right]]
    check('repair_persistent_ID_objects',assocs[0]['objects']==assocs[1]['objects'])
    check('repair_instance_base',assocs[0]['instance_base']==assocs[1]['instance_base'])
    return dict(status='PASS' if all(x['exact'] for x in records if x.get('required',True)) else 'DIFFERENT',left=str(left),right=str(right),
                checks=records,candidate_unification_performed=False,zero_float_tolerance=True,
                metadata_alignment='Run-root paths normalized; ROI UID aligned by exact full coordinate multiset from its original catalog surface. Raw metadata difference retained as diagnostic.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--left',default='reference');parser.add_argument('--right',default='checkpoint');parser.add_argument('--name',default='tracking_comparison')
    args=parser.parse_args()
    result=compare(OUT/args.left,OUT/args.right)
    (OUT/(args.name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],checks=[{k:v for k,v in r.items() if k!='differences'} for r in result['checks']]),indent=2))
