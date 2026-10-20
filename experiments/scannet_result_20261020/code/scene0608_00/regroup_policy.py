"""Reapply the existing r2 SAM-only policy after approved whole-mask filters."""
from pathlib import Path
import importlib.util
import numpy as np

POLICY_PATH = Path(__file__).resolve().parent / 'grouping_r2/source_policy.py'
spec = importlib.util.spec_from_file_location('existing_r2_source_policy', POLICY_PATH)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def resolve(seeds, intersections, group_seeds, stage):
    original = seeds
    seeds = [dict(s, aliases=list(s['aliases'])) for s in seeds]
    kept = list(range(len(seeds)))
    inter = intersections.copy()
    records, dropped, reports = [], [], []
    for pass_number in range(1, len(seeds) + 2):
        counts = np.asarray([s['support_count'] for s in seeds], dtype=np.int64)
        union = counts[:, None] + counts[None, :] - inter
        passes = (union > 0) & (4 * inter > union)
        groups = group_seeds(range(len(seeds)), lambda a, b: bool(passes[a, b]))
        removals = policy.mixed_group_source_exclusions(groups, seeds, inter, pass_number)
        if not removals:
            policy.assert_sam_only_groups(groups, seeds)
            for i, s in enumerate(seeds):
                s['index'] = i
                before = original[kept[i]]
                assert s['digest'] == before['digest']
                assert s['support_count'] == before['support_count']
            assert np.array_equal(inter, intersections[np.ix_(kept, kept)])
            assert {s['id'] for s in seeds + dropped} == {s['id'] for s in original}
            return seeds, inter, groups, kept, records, dropped, reports
        selected = []
        for i, seed in enumerate(seeds):
            if i in removals:
                evidence = removals[i]
                aliases = evidence['retained_aliases']
                records.append(dict(
                    id=seed['id'], digest=seed['digest'], frame=seed['frame'],
                    support_count=seed['support_count'], area=seed['area'],
                    stage=stage, reason='GLOBAL_SEMANTIC_SAM_ONLY_CROSS_VIEW',
                    entire_mask_excluded=not aliases, **evidence))
                if not aliases:
                    dropped.append(seed)
                    continue
                seed['aliases'] = aliases
                seed['representative'] = min(aliases, key=lambda a: (
                    not a['semantic_hit'], a['model'], a['touches_crop'], a['crop_key'], a['label']))
            selected.append(i)
        selected.sort(key=lambda i: (-seeds[i]['support_count'], seeds[i]['frame'],
                                     seeds[i]['representative']['model'], seeds[i]['digest']))
        reports.append(dict(stage=stage, pass_number=pass_number, before_masks=len(seeds),
                            after_masks=len(selected), affected_mask_ids=[seeds[i]['id'] for i in removals],
                            removed_aliases=sum(len(e['removed_aliases']) for e in removals.values())))
        seeds = [seeds[i] for i in selected]
        kept = [kept[i] for i in selected]
        inter = inter[np.ix_(selected, selected)]
    raise AssertionError('Existing SAM-only source policy failed to converge')
