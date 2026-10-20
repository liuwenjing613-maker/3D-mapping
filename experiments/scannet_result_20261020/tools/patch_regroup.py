"""Approved scene0608 adapter; deliberately leaves completed scene0011 intact."""

def replace_once(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)


def adapt(s, rel):
    if rel not in ('grouping_r3/compute.py', 'v1/compute.py'):
        return s
    s = replace_once(s, 'C=Path(__file__).resolve().parent' if rel.startswith('v1') else
                     'C = Path(__file__).resolve().parent',
                     "C=Path(__file__).resolve().parent\nimport sys\nsys.path.insert(0,str(C.parent))\nfrom regroup_policy import resolve, POLICY_PATH")
    anchor = "assert_sam_only_groups(groups_raw,seeds)" if rel.startswith('v1') else "assert_sam_only_groups(groups_raw, seeds)"
    stage = 'post_noise' if rel.startswith('v1') else 'post_crop'
    block = f"""seeds, inter, groups_raw, retained_indices, regroup_exclusions, semantic_dropped, regroup_passes = resolve(seeds, inter, group_seeds, {stage!r})
for p in [C.parent/'regroup_policy.py', POLICY_PATH, C/'compute.py']:
    source_sha[str(p)] = sha(p)
counts = np.asarray([s['support_count'] for s in seeds], dtype=np.int64)
offsets = np.concatenate(([0], np.cumsum(counts)))
"""
    if rel.startswith('v1'):
        block += """supports = [supports[i] for i in retained_indices]
old_active_indices = [old_active_indices[i] for i in retained_indices]
indices = np.concatenate(supports).astype(np.int32)
with np.load(P/'pair_metrics.npz') as z:
    assert np.array_equal(inter,z['intersection'][np.ix_(old_active_indices,old_active_indices)])
for s in seeds:
    s['rois'] = sorted({a['uid'] for a in s['aliases']})
    s['candidate_ids'] = sorted({a['candidate_id'] for a in s['aliases']})
"""
    else:
        block += """indices = np.concatenate([s['support'] for s in seeds]).astype(np.int32)
excluded.extend(regroup_exclusions)
assert {s['digest'] for s in seeds+crop_excluded+semantic_dropped} == {s['digest'] for s in previous['seeds']}
assert len(seeds)+len(crop_excluded)+len(semantic_dropped) == len(previous['seeds'])
"""
    block += """union = counts[:,None] + counts[None,:] - inter
iou = np.divide(inter,union,out=np.zeros(inter.shape,float),where=union>0)
passes = (union>0)&(4*inter>union)
print('REGROUP_SAM_ONLY', json.dumps(regroup_passes), flush=True)
""" + anchor
    s = replace_once(s, anchor, block)
    if rel.startswith('grouping_r3'):
        s = replace_once(s, "before_crop_filter_masks=source_eligible_masks,",
            "before_crop_filter_masks=source_eligible_masks,\n    post_crop_source_excluded_masks=len(semantic_dropped), post_crop_source_excluded_aliases=sum(len(e['removed_aliases']) for e in regroup_exclusions),")
        s = replace_once(s, "save('semantic_exclusions.json', dict(policy=rule['source_policy'], excluded=excluded,",
            "save('semantic_exclusions.json', dict(policy=rule['source_policy'], excluded=excluded,\n    post_crop_regroup_passes=regroup_passes, post_crop_exclusions=regroup_exclusions,")
        s = replace_once(s, 'active plus crop-excluded masks exactly reproduce previous source-eligible input',
            'active plus crop-excluded plus post-crop source-excluded masks exactly partition previous source-eligible input')
    else:
        s = replace_once(s, "'before_noise_filter_masks':len(old['seeds']),",
            "'post_noise_source_excluded_masks':len(semantic_dropped),'post_noise_source_excluded_aliases':sum(len(e['removed_aliases']) for e in regroup_exclusions),\n    'excluded_cf_masks':old['stats']['excluded_cf_masks']+len(semantic_dropped),\n    'excluded_cf_occurrences':old['stats']['excluded_cf_occurrences']+sum(len(e['removed_aliases']) for e in regroup_exclusions),\n    'before_noise_filter_masks':len(old['seeds']),")
        s = replace_once(s, "excluded_crop_masks=crop_excluded,excluded_noise_masks=excluded,",
            "excluded_crop_masks=crop_excluded,excluded_noise_masks=excluded,excluded_semantic_masks=semantic_dropped,")
        anchor = "for name in ['crop_exclusions.json','semantic_exclusions.json']:shutil.copy2(P/name,R/name)"
        s = replace_once(s, anchor, anchor + """
source_report=json.loads((R/'semantic_exclusions.json').read_text())
source_report['excluded'].extend(regroup_exclusions)
source_report.update(post_noise_regroup_passes=regroup_passes,post_noise_exclusions=regroup_exclusions,stats=stats)
save('semantic_exclusions.json',source_report)""")
        s = replace_once(s, "for s in seeds+excluded}==", "for s in seeds+excluded+semantic_dropped}==")
        s = replace_once(s, "assert len(seeds)+len(excluded)==", "assert len(seeds)+len(excluded)+len(semantic_dropped)==")
    return s
