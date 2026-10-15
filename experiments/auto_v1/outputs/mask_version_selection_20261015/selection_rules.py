"""Deterministic rules; grouping is a comparison group, never a proven object ID."""
from collections import defaultdict

def choose(nodes, edges):
    by={n['key']:n for n in nodes};eligible={k for k,n in by.items() if n['eligible']}
    parent={k:k for k in eligible}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for e in edges:
        if e['group_link'] and e['a'] in eligible and e['b'] in eligible:join(e['a'],e['b'])
    sets=defaultdict(list)
    for k in sorted(eligible):sets[find(k)].append(k)
    out=[]
    for index,members in enumerate(sorted(sets.values(),key=lambda x:x[0]),1):
        member_set=set(members);pairs=[e for e in edges if e['a'] in member_set and e['b'] in member_set]
        references=[e for e in edges if (e['a'] in member_set) != (e['b'] in member_set) and e['group_link']]
        containment=[e for e in pairs+references if e['containment']]
        near={frozenset((e['a'],e['b'])) for e in pairs if e['near']}
        def same(a,b):return frozenset((a,b)) in near
        warnings=[]
        if any(by[k]['touches_crop'] for k in members):warnings.append('TOUCHES_ARTIFICIAL_CROP')
        if any(by[k]['touches_frame'] for k in members):warnings.append('TOUCHES_IMAGE_EDGE')
        selected=None;reason='UNRESOLVED_DISAGREEMENT'
        if len(members)==1:
            selected=members[0];reason='SINGLE_REMAINING'
            if containment:warnings.append('SINGLE_HAS_NONINTERSECTING_CONTAINMENT_REFERENCE')
        elif containment:
            # An internal crop edge is unknown outside the crop, not evidence of a part.
            cut_relation=any(by[e['a']]['touches_crop'] or by[e['b']]['touches_crop'] for e in containment)
            reason='CROP_TRUNCATION_UNCERTAIN' if cut_relation else 'WHOLE_PART_CONFLICT'
        elif all(same(a,b) for i,a in enumerate(members) for b in members[i+1:]):
            selected=min(members,key=lambda k:(by[k]['model'],by[k]['label']))
            reason='THREE_AGREE' if len(members)==3 else ('CROPFORMERS_AGREE' if {by[k]['model'] for k in members}=={0,1} else 'CROSS_MODEL_AGREE')
        elif any(by[k]['touches_crop'] for k in members):reason='CROP_TRUNCATION_UNCERTAIN'
        else:
            supported=[(a,b) for a in members if by[a]['model'] in [0,1] for b in members if by[b]['model']==2 and same(a,b)]
            if len(supported)==1 and len(members)<=3 and len({by[k]['model'] for k in members})==len(members):
                selected=supported[0][0];reason='ONE_CROPFORMER_WITH_SAM'
            elif len(supported)>1:reason='MULTIPLE_SUPPORTED_VERSIONS'
        group=dict(id=f'G{index:03d}',members=members,selected=selected,status='selected' if selected else 'deferred',reason=reason,warnings=warnings,relations=pairs,references=references)
        out.append(group)
    assert sorted(k for g in out for k in g['members'])==sorted(eligible)
    assert all(g['selected'] is None or by[g['selected']]['eligible'] for g in out)
    return out
