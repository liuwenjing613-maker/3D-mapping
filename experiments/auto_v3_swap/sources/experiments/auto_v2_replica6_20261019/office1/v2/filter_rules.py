"""The accepted auto-v2 filters, evaluated from immutable measured counts.

The measurements contain no decisions or human labels. The only manual input is
the explicit, case-scoped override applied AFTER the five automatic stages.
"""
from itertools import combinations


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_count(value, name, allow_zero=False):
    require(type(value) is int and value >= (0 if allow_zero else 1),
            'Invalid count: ' + name)
    return value


def semantic_excluded(high_excluded_pixels, area):
    positive_count(area, 'area')
    positive_count(high_excluded_pixels, 'high_excluded_pixels', True)
    require(high_excluded_pixels <= area, 'Semantic count exceeds mask area')
    return 10 * high_excluded_pixels >= 9 * area


def iou_duplicate(intersection, union):
    positive_count(intersection, 'intersection', True)
    positive_count(union, 'union', True)
    require(intersection <= union, 'Intersection exceeds union')
    # There is deliberately no individual visibility fraction or point gate.
    return union > 0 and 5 * intersection >= 4 * union


def validate_coverage(c, n, i):
    positive_count(c, 'intersection', True)
    positive_count(n, 'support_count')
    positive_count(i, 'instance_visible_count', True)
    require(c <= min(n, i), 'Coverage intersection exceeds denominator')


def skip_repair(c, n, i):
    validate_coverage(c, n, i)
    return i > 0 and 20 * c >= 19 * n and 20 * c >= 19 * i


def pr_excluded(c, n, i):
    validate_coverage(c, n, i)
    return i > 0 and 10 * c > n and 5 * c < i


def containment_small(a, na, b, nb, common):
    positive_count(na, 'count_a')
    positive_count(nb, 'count_b')
    positive_count(common, 'original_intersection', True)
    require(common <= min(na, nb), 'Full intersection exceeds smaller support')
    if 10 * common <= 7 * min(na, nb):
        return None
    require(na != nb, 'Equal-size qualifying containment needs user choice')
    return (a, b) if na < nb else (b, a)


def fixed_primary_groups(ordered_ids, passes):
    remaining = list(ordered_ids)
    groups = []
    while remaining:
        primary = remaining[0]
        direct = [candidate for candidate in remaining[1:] if passes(primary, candidate)]
        groups.append([primary] + direct)
        removed = set(direct) | {primary}
        remaining = [candidate for candidate in remaining if candidate not in removed]
    return groups


def identity(record):
    return {key: record[key] for key in
            ('group_id', 'original_track_id', 'mask_id', 'frame', 'mask_sha256')}


def run_filters(measurements, overrides=None):
    require(measurements['schema'] == 'auto-v2-count-measurements-v1', 'Wrong input schema')
    records = measurements['seeds']
    seeds = {record['group_id']: record for record in records}
    require(len(seeds) == len(records), 'Duplicate seed ID')
    require(len({r['original_track_id'] for r in records}) == len(records), 'Duplicate track ID')
    require(len({r['mask_id'] for r in records}) == len(records), 'Duplicate mask ID')
    for record in records:
        positive_count(record['support_count'], 'support_count')
        positive_count(record['original_track_id'], 'original_track_id')
        require(record['frame'] >= 0, 'Negative frame')

    audit = {gid: dict(**identity(seed), stage_decisions=[], decision=None)
             for gid, seed in seeds.items()}
    stages = []

    def finish_stage(name, incoming, removed, reasons=None):
        removed = set(removed)
        require(removed <= set(incoming), 'Stage reintroduced an excluded seed')
        retained = sorted(set(incoming) - removed)
        stages.append(dict(stage=name, input=len(incoming), removed=len(removed),
                           retained=len(retained), removed_group_ids=sorted(removed),
                           retained_group_ids=retained))
        for gid in incoming:
            decision = 'skip_repair' if gid in removed and name == 'skip_repair_B' else (
                'exclude' if gid in removed else 'retain')
            audit[gid]['stage_decisions'].append(dict(stage=name, decision=decision,
                evidence=(reasons or {}).get(gid)))
            if gid in removed:
                audit[gid].update(decision=decision, terminal_stage=name)
        return retained

    current = sorted(seeds)
    sem_removed = [gid for gid in current
                   if semantic_excluded(seeds[gid]['high_excluded_pixels'], seeds[gid]['area'])]
    current = finish_stage('semantic', current, sem_removed, {
        gid: dict(high_excluded_pixels=seeds[gid]['high_excluded_pixels'], area=seeds[gid]['area'])
        for gid in current})

    pairs = {}
    for pair in measurements['pairs']:
        a, b = pair['a'], pair['b']
        key = tuple(sorted((a, b)))
        require(a != b and a in current and b in current, 'Pair outside semantic survivor scope')
        require(key not in pairs, 'Duplicate pair')
        require(pair['count_a'] == seeds[a]['support_count'] and
                pair['count_b'] == seeds[b]['support_count'], 'Pair support identity mismatch')
        c, u, va, vb = (pair[k] for k in ('intersection', 'union', 'visible_a', 'visible_b'))
        require(0 <= c <= min(va, vb) and u == va + vb - c, 'Invalid common-visible counts')
        require(va <= pair['count_a'] and vb <= pair['count_b'], 'Visibility exceeds full support')
        require(c <= pair['original_intersection'] <= min(pair['count_a'], pair['count_b']),
                'Invalid full-support intersection')
        pairs[key] = pair
    require(set(pairs) == set(combinations(current, 2)), 'Missing complete pair measurements')
    ordered = sorted(current, key=lambda gid: (-seeds[gid]['support_count'], seeds[gid]['frame'],
                     seeds[gid]['source_model'], seeds[gid]['mask_digest']))
    groups = fixed_primary_groups(ordered, lambda a, b: iou_duplicate(
        pairs[tuple(sorted((a, b)))]['intersection'], pairs[tuple(sorted((a, b)))]['union']))
    representatives = {gid: group[0] for group in groups for gid in group}
    reasons = {}
    for gid in current:
        rep = representatives[gid]
        reasons[gid] = dict(primary_group_id=rep)
        if rep != gid:
            pair = pairs[tuple(sorted((rep, gid)))]
            reasons[gid].update(intersection=pair['intersection'], union=pair['union'])
    current = finish_stage('covisible_iou_only', current,
                           [gid for gid in current if representatives[gid] != gid], reasons)

    coverage = {row['group_id']: row for row in measurements['coverage']}
    require(len(coverage) == len(measurements['coverage']) and set(coverage) == set(current),
            'Coverage input differs from deduplicated seed scope')
    for gid in current:
        rec = coverage[gid]
        require(identity(rec) == identity(seeds[gid]), 'Coverage identity mismatch: ' + gid)
        require(rec['support_count'] == seeds[gid]['support_count'], 'Coverage support changed')
        validate_coverage(rec['intersection'], rec['support_count'], rec['instance_visible_count'])
        require((rec['instance_id'] > 0) == (rec['instance_visible_count'] > 0),
                'Invalid matched baseline instance')
    args = lambda gid: tuple(coverage[gid][key] for key in
                            ('intersection', 'support_count', 'instance_visible_count'))
    current = finish_stage('skip_repair_B', current,
                           [gid for gid in current if skip_repair(*args(gid))], coverage)

    witnesses = {}
    for a, b in combinations(current, 2):
        pair = pairs[a, b]
        small_large = containment_small(a, seeds[a]['support_count'], b,
                                       seeds[b]['support_count'], pair['original_intersection'])
        if small_large:
            small, large = small_large
            witnesses.setdefault(small, []).append(dict(larger_group_id=large,
                intersection=pair['original_intersection'], small_count=seeds[small]['support_count']))
    require(not any(w['larger_group_id'] in witnesses for rows in witnesses.values() for w in rows),
            'Chained qualifying containment needs user review; no hidden ancestor policy')
    current = finish_stage('full_support_containment', current, witnesses, witnesses)
    before_pr = current[:]
    automatic_removed = [gid for gid in current if pr_excluded(*args(gid))]
    current = finish_stage('pr_filter', current, automatic_removed, coverage)
    automatic_retained = current[:]

    applied = []
    if overrides is not None:
        require(overrides['case_id'] == measurements['case_id'], 'Manual override case mismatch')
        require(overrides['stage'] == 'pr_filter', 'Only the final p/r stage permits this override')
        require(overrides['input_group_ids'] == before_pr, 'Manual override input scope changed')
        require(overrides['automatic_excluded_group_ids'] == sorted(automatic_removed),
                'Manual override automatic exclusions changed')
        chosen = set()
        for entry in overrides['retain']:
            gid = entry['group_id']
            require(gid in automatic_removed and gid not in chosen, 'Invalid repeated/manual restore')
            require(identity(entry) == identity(seeds[gid]), 'Manual override identity changed')
            chosen.add(gid)
            audit[gid].update(decision='retain', terminal_stage='manual_room2_override')
            audit[gid]['stage_decisions'].append(dict(stage='manual_room2_override',
                decision='retain', evidence=dict(user_quote=overrides['user_quote'])))
            applied.append(gid)
        current = sorted(set(current) | chosen)
        require(set(automatic_removed) - chosen == set(overrides['final_excluded_group_ids']),
                'Manual override final exclusion list mismatch')

    for gid in current:
        if audit[gid]['decision'] is None:
            audit[gid].update(decision='retain', terminal_stage='automatic_complete')
    return dict(schema='auto-v2-stage-selection-v1', case_id=measurements['case_id'],
                automatic_retained_group_ids=automatic_retained,
                retained_group_ids=current, manual_restored_group_ids=sorted(applied),
                final_pr_excluded_group_ids=sorted(set(automatic_removed) - set(applied)),
                stages=stages, dedup_groups=groups, records=[audit[gid] for gid in sorted(audit)])
