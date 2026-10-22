"""Version selection on overlap groups. Crop contact is diagnostic only."""
from collections import defaultdict
from itertools import combinations, product


def noise_reason(area, width=None, length=None):
    if area <= 25:
        return 'TINY_AREA_NOISE'
    if area <= 100 and width is not None and length is not None:
        if width <= 2.00001 and length / max(width, 1e-12) >= 5 - 1e-5:
            return 'THIN_LINE_NOISE'
    return None


def choose(nodes, edges):
    by = {n['key']: n for n in nodes}
    active = {k for k, n in by.items() if not n.get('noise_reason')}
    eligible = {k for k in active if by[k]['eligible']}
    edges = [e for e in edges if e['a'] in active and e['b'] in active]
    parent = {k: k for k in eligible}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for e in edges:
        if e['group_link'] and e['a'] in eligible and e['b'] in eligible:
            join(e['a'], e['b'])
    sets = defaultdict(list)
    for k in sorted(eligible):
        sets[find(k)].append(k)

    out = []

    def emit(members, selected, reason, parent_group, original_members):
        member_set = set(members)
        pairs = [e for e in edges if e['a'] in member_set and e['b'] in member_set]
        refs = [e for e in edges if ((e['a'] in member_set) != (e['b'] in member_set)) and e['group_link']]
        warnings = []
        if any(by[k]['touches_crop'] for k in members):
            warnings.append('TOUCHES_ARTIFICIAL_CROP')
        if any(by[k]['touches_frame'] for k in members):
            warnings.append('TOUCHES_IMAGE_EDGE')
        if len(members) == 1 and selected and any(e['containment'] for e in refs):
            warnings.append('SINGLE_HAS_NONINTERSECTING_CONTAINMENT_REFERENCE')
        out.append(dict(id=f'G{len(out)+1:03d}', members=members, selected=selected,
                        status='selected' if selected else 'deferred', reason=reason,
                        warnings=warnings, relations=pairs, references=refs,
                        parent_group=parent_group, original_members=original_members))

    for index, members in enumerate(sorted(sets.values(), key=lambda x: x[0]), 1):
        member_set = set(members)
        pairs = [e for e in edges if e['a'] in member_set and e['b'] in member_set]
        refs = [e for e in edges if ((e['a'] in member_set) != (e['b'] in member_set)) and e['group_link']]
        containment = [e for e in pairs + refs if e['containment']]
        near = {frozenset((e['a'], e['b'])) for e in pairs if e['near']}

        def same(a, b):
            return frozenset((a, b)) in near

        parent_group = f'P{index:03d}'
        selected, reason = None, 'UNRESOLVED_DISAGREEMENT'
        if len(members) == 1:
            selected, reason = members[0], 'SINGLE_REMAINING'
        elif containment:
            # A crop edge no longer vetoes selection or creates a separate reason.
            # Actual whole/part conflicts still defer, as clarified by the user.
            reason = 'WHOLE_PART_CONFLICT'
        elif all(same(a, b) for a, b in combinations(members, 2)):
            selected = min(members, key=lambda k: (by[k]['model'], by[k]['label']))
            reason = 'THREE_AGREE' if len(members) == 3 else ('CROPFORMERS_AGREE' if {by[k]['model'] for k in members} == {0, 1} else 'CROSS_MODEL_AGREE')
        else:
            sources = [[k for k in members if by[k]['model'] == m] for m in range(3)]
            triples = [list(t) for t in product(*sources) if all(same(a, b) for a, b in combinations(t, 2))]
            if triples:
                # Mutual unique best matches make these triples disjoint.
                flattened = [k for t in triples for k in t]
                assert len(flattened) == len(set(flattened))
                for t in sorted(triples):
                    emit(t, t[0], 'AGREED_MAIN_KEEP', parent_group, members)
                extras = sorted(member_set - set(flattened))
                if extras:
                    emit(extras, None, 'EXTRA_FRAGMENT_REVIEW', parent_group, members)
                continue
            supported = [(a, b) for a in members if by[a]['model'] in (0, 1)
                         for b in members if by[b]['model'] == 2 and same(a, b)]
            if len(supported) == 1 and len(members) <= 3 and len({by[k]['model'] for k in members}) == len(members):
                selected, reason = supported[0][0], 'ONE_CROPFORMER_WITH_SAM'
            elif len(supported) > 1:
                reason = 'MULTIPLE_SUPPORTED_VERSIONS'
        emit(members, selected, reason, parent_group, members)

    assert sorted(k for g in out for k in g['members']) == sorted(eligible)
    assert all(g['selected'] is None or by[g['selected']]['eligible'] for g in out)
    return out
