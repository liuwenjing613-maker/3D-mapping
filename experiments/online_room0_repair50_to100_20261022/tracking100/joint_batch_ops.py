"""Combine human cases before observation retirement and persistent-ID dedup."""
import numpy as np

def canonical_map(track_ids, aliases):
    ids=set(map(int,track_ids)); aliases={int(a):int(b) for a,b in aliases.items()}
    if not set(aliases).issubset(ids) or not set(aliases.values()).issubset(ids):
        raise ValueError('Alias must refer to an actual human-selected mask')
    result={}
    for oid in sorted(ids):
        current=oid; seen=set()
        while current in aliases:
            if current in seen: raise ValueError('Alias cycle')
            seen.add(current);current=aliases[current]
        result[oid]=current
    return result

def merge_foregrounds(proposals, canonical):
    """Same-object masks union; pixels claimed by different objects abstain."""
    if not proposals: raise ValueError('Need at least one proposal')
    shape=np.asarray(proposals[0][1]).shape
    groups={}
    for oid,mask in proposals:
        mask=np.asarray(mask,bool)
        if mask.shape!=shape: raise ValueError('Proposal shapes differ')
        cid=canonical[int(oid)]
        groups[cid]=groups.get(cid,np.zeros(shape,bool)) | mask
    labels=np.zeros(shape,np.uint16); claims=np.zeros(shape,np.uint16)
    for cid,mask in sorted(groups.items()): labels[mask]=cid;claims+=mask
    conflict=claims>1;labels[conflict]=0
    return labels,conflict

def combined_plan(case_actions, required_canonical_ids, accepted_canonical_ids, conflict_intersects_observation):
    """Never let one case retire another case's unexplained target observation."""
    if not case_actions: return 'retain'
    if (any(a=='whole' for a in case_actions) and
            set(required_canonical_ids).issubset(set(accepted_canonical_ids)) and
            not conflict_intersects_observation): return 'whole'
    return 'partial'

def count_vote_keys(frame_arrays):
    if not frame_arrays: return np.empty(0,np.int64),np.empty(0,np.int64)
    if any(len(x)!=len(np.unique(x)) for x in frame_arrays):
        raise ValueError('Each frame must deduplicate before counting')
    k,v=np.unique(np.concatenate(frame_arrays),return_counts=True)
    return k,v.astype(np.int64)
