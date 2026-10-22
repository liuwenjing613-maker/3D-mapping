"""Filter whole masks, then reselect seeds on the unchanged frozen TSDF."""
from pathlib import Path
from collections import Counter
import json, hashlib, shutil, zipfile, time
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from grouping_rules import group_seeds, same_instance
from source_policy import assert_sam_only_groups
from crop_filter import has_crop_contact
from noise_filter import shape_features, classify, is_review_control, RULE

C=Path(__file__).resolve().parent
P=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/300/A/candidates/grouping_r3')
R=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/300/A/candidates/v1')
R.mkdir(parents=True,exist_ok=True)
started=time.time()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    (R/name).write_text(json.dumps(value,ensure_ascii=False,separators=(',',':')),encoding='utf-8')

rule=json.loads((C/'config.json').read_text())
assert rule['noise_filter']==RULE and rule['iou']==.25
old=json.loads((P/'grouping.json').read_text())
prior_audit=json.loads((P/'audit.json').read_text())
assert prior_audit['status']=='PASS'
for name,digest in prior_audit['output_sha256'].items():assert sha(P/name)==digest
source_sha=dict(old['source_sha256'])
for name in ['grouping.json','audit.json','surface_supports.npz','pair_metrics.npz']:
    source_sha[str(P/name)]=sha(P/name)
for p,digest in source_sha.items():assert sha(p)==digest,('upstream changed',p)
with np.load(P/'surface_supports.npz') as z:
    old_indices=z['indices'];old_offsets=z['offsets']
seeds=[];excluded=[];features=[];supports=[];old_active_indices=[]
for folder in ['masks','excluded_masks','noise_masks']:(R/folder).mkdir(exist_ok=True)
for s0 in old['seeds']:
    s=dict(s0)
    a=np.asarray(Image.open(P/s['mask']))>0
    assert sha(P/s['mask'])==s['mask_sha256']
    assert hashlib.sha256(s['frame'].to_bytes(4,'little')+np.packbits(a.reshape(-1)).tobytes()).hexdigest()==s['digest']
    assert a.shape==(680,1200) and int(a.sum())==s['area']
    f=shape_features(a);decision=classify(f)
    s['noise_features']=f;s['noise_decision']=decision
    s['previous_group']=old['groups'][s['group']]['id']
    s['previous_primary']=old['groups'][s['group']]['primary']==s['index']
    features.append(dict(id=s['id'],features=f,decision=decision))
    support=old_indices[old_offsets[s['index']]:old_offsets[s['index']+1]]
    assert len(support)==s['support_count'] and len(np.unique(support))==len(support)
    assert hashlib.sha256(support.tobytes()).hexdigest()==s['support_sha256']
    if decision['exclude']:
        s.update(index=len(excluded),group=None,reason=decision['reason'],mask=f"noise_masks/{s['id']}.png")
        (R/s['mask']).symlink_to((P/s0['mask']).resolve());excluded.append(s)
    else:
        old_active_indices.append(s0['index']);s['index']=len(seeds)
        s['noise_review_control']=is_review_control(f)
        (R/s['mask']).symlink_to((P/s['mask']).resolve());seeds.append(s);supports.append(support)
    source_sha[str(P/s0['mask'])]=s['mask_sha256']
assert not any(has_crop_contact(s['aliases']) for s in seeds+excluded)
print('FILTER',len(old['seeds']),'-',len(excluded),'=',len(seeds),Counter(s['reason'] for s in excluded),flush=True)

counts=np.array([s['support_count'] for s in seeds],dtype=np.int64)
assert np.all(counts[:-1]>=counts[1:])
offsets=np.concatenate(([0],np.cumsum(counts)))
indices=np.concatenate(supports).astype(np.int32)
matrix=csr_matrix((np.ones(len(indices),np.int32),indices,offsets),shape=(len(seeds),old['visibility_rule']['vertex_count']))
inter=(matrix@matrix.T).toarray().astype(np.int64)
union=counts[:,None]+counts[None,:]-inter
iou=np.divide(inter,union,out=np.zeros(inter.shape,float),where=union>0)
passes=(union>0)&(4*inter>union)
with np.load(P/'pair_metrics.npz') as z:
    assert np.array_equal(inter,z['intersection'][np.ix_(old_active_indices,old_active_indices)])
groups_raw=group_seeds(range(len(seeds)),lambda a,b:bool(passes[a,b]))
assert_sam_only_groups(groups_raw,seeds)
group_of={j:gi for gi,members in enumerate(groups_raw) for j in members}
groups=[]
for gi,members in enumerate(groups_raw):
    primary=members[0];vframes={};sameframe=[]
    assert counts[primary]==max(counts[members])
    for j in members[1:]:
        exact=len(np.intersect1d(supports[primary],supports[j],assume_unique=True))
        assert exact==inter[primary,j] and same_instance(exact,counts[primary],counts[j])
        if seeds[j]['frame']==seeds[primary]['frame']:sameframe.append(j)
        else:vframes.setdefault(seeds[j]['frame'],[]).append(j)
    aux_below=sum(not passes[a,b] for ai,a in enumerate(members[1:]) for b in members[ai+2:])
    nearest=sorted((j for j in range(len(seeds)) if group_of[j]!=gi and inter[primary,j]>0),key=lambda j:(-iou[primary,j],j))[:5]
    values=[float(iou[a,b]) for ai,a in enumerate(members) for b in members[ai+1:]]
    groups.append(dict(id=f'G{gi+1:03d}',index=gi,primary=primary,members=members,member_count=len(members),
       frames=sorted({seeds[j]['frame'] for j in members}),
       validation_frames=[dict(frame=fr,members=js,representative=js[0]) for fr,js in sorted(vframes.items())],
       same_frame_references=sameframe,min_primary_iou=min((float(iou[primary,j]) for j in members[1:]),default=None),
       min_all_pair_iou=min(values,default=None),below_threshold_aux_pairs=int(aux_below),
       warnings=['auxiliary_pairs_below_threshold_allowed'] if aux_below else [],
       nearest_outside=[dict(member=j,group=group_of[j],iou=float(iou[primary,j])) for j in nearest],
       semantic_window_present=any(a['semantic_hit'] for j in members for a in seeds[j]['aliases']),
       semantic_families=sorted({a['semantic_family'] for j in members for a in seeds[j]['aliases'] if a['semantic_hit']})))
for s in seeds:s['group']=group_of[s['index']]
crop_excluded=old['excluded_crop_masks']
for s in crop_excluded:
    assert sha(P/s['mask'])==s['mask_sha256']
    (R/s['mask']).symlink_to((P/s['mask']).resolve())
active_ids={s['id'] for s in seeds}
emptied=[dict(id=g['id'],mask_ids=[old['seeds'][j]['id'] for j in g['members']]) for g in old['groups'] if not any(old['seeds'][j]['id'] in active_ids for j in g['members'])]
stats={**old['stats'],
    'before_noise_filter_masks':len(old['seeds']),'excluded_noise_masks':len(excluded),
    'excluded_noise_occurrences':sum(len(s['aliases']) for s in excluded),
    'noise_reason_counts':dict(Counter(s['reason'] for s in excluded)),
    'previous_groups_emptied_by_crop':old['stats']['previous_groups_emptied'],
    'previous_groups_emptied':len(emptied),'previous_primaries_removed':sum(s['previous_primary'] for s in excluded),
    'selected_occurrences':sum(len(s['aliases']) for s in seeds),'unique_masks':len(seeds),
    'exact_duplicate_occurrences_removed':sum(len(s['aliases']) for s in seeds)-len(seeds),
    'groups':len(groups),'primary_seeds':len(groups),'auxiliary_masks':len(seeds)-len(groups),
    'multi_groups':sum(len(g['members'])>1 for g in groups),'singleton_groups':sum(len(g['members'])==1 for g in groups),
    'zero_support_masks':int(sum(counts==0)),
    'groups_without_other_frame':sum(not g['validation_frames'] for g in groups),
    'groups_with_low_aux_pair':sum(g['below_threshold_aux_pairs']>0 for g in groups),
    'semantic_groups':sum(g['semantic_window_present'] for g in groups),
    'semantic_groups_non_sam_primary':0,'semantic_groups_cf_members':0,
    'review_control_masks':sum(s['noise_review_control'] for s in seeds)}
save('grouping.json',dict(rule=rule,stats=stats,seeds=seeds,groups=groups,intersection=inter.tolist(),
    excluded_crop_masks=crop_excluded,excluded_noise_masks=excluded,source_sha256=source_sha,visibility_rule=old['visibility_rule']))
save('noise_exclusions.json',dict(policy=RULE,stats=stats,excluded=excluded,previous_groups_emptied=emptied,
    retained_review_controls=[s['id'] for s in seeds if s['noise_review_control']],
    caveat='Morphology is reviewable evidence, not semantic ground truth. Retained masks are not pixel-eroded.'))
save('noise_features.json',dict(policy=RULE,masks=features))
for name in ['crop_exclusions.json','semantic_exclusions.json']:shutil.copy2(P/name,R/name)
save('seed_manifest.json',dict(status='awaiting_visual_review',tracking_requested=False,map_write=False,rule=rule,
    manual_source_corrections=json.loads((R.parent/'grouping_r2/semantic_exclusions.json').read_text()).get('manual_source_corrections'),
    groups=[dict(id=g['id'],primary=seeds[g['primary']],auxiliary_ids=[seeds[j]['id'] for j in g['members'][1:]],validation_frames=g['validation_frames']) for g in groups]))
np.savez_compressed(R/'surface_supports.npz',indices=indices,offsets=offsets)
np.savez_compressed(R/'pair_metrics.npz',intersection=inter,union=union,iou=iou,passes=passes)
assert {s['digest'] for s in seeds+excluded}=={s['digest'] for s in old['seeds']}
assert len(seeds)+len(excluded)==len(old['seeds'])
assert sorted(j for g in groups for j in g['members'])==list(range(len(seeds)))
for p,digest in source_sha.items():assert sha(p)==digest,('input changed during run',p)
names=['grouping.json','seed_manifest.json','surface_supports.npz','pair_metrics.npz','semantic_exclusions.json','crop_exclusions.json','noise_exclusions.json','noise_features.json']
audit=dict(status='PASS',stats=stats,elapsed_seconds=round(time.time()-started,2),
    checks=['unchanged verified r3 inputs and upstream source hashes','each full mask binary digest verified','native-resolution distance transform with zero padding',
       'area strictly less than 100; integer 95% tests','active plus excluded exactly partition previous-stage masks',
       'no pixel editing; full mask/support hashes preserved','prior crop and SAM-only exclusions preserved',
       'all pair intersections recomputed from TSDF supports and equal prior submatrix','fixed-primary IoU > 0.25 grouping and max-TSDF seed',
       'every primary-member pair independently recounted','SAM-only semantic groups','all excluded masks included in audit'],
    output_sha256={name:sha(R/name) for name in names})
save('audit.json',audit)
with zipfile.ZipFile(R/'grouping_data.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for name in names+['audit.json']:z.write(R/name,name)
    for folder in ['masks','excluded_masks','noise_masks']:
        for p in sorted((R/folder).glob('*.png')):z.write(p,p.relative_to(R).as_posix())
save('receipt.json',dict(sha256=sha(R/'grouping_data.zip')))
print('COMPLETE',json.dumps(stats),flush=True)
