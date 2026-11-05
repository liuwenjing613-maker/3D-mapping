"""Experimental policy for candidate surface-point IDs; not yet wired to mapper."""
import numpy as np

def partition_candidates(points, local_ids, trusted_owner, fitted_ids, minimum=10, fraction=.05):
    points=np.asarray(points, dtype=np.int64)
    local_ids=np.asarray(local_ids, dtype=np.int64)
    trusted_owner=np.asarray(trusted_owner, dtype=np.int64)
    fitted_ids=np.asarray(fitted_ids, dtype=np.int64)
    if points.ndim != 1 or not (len(points)==len(local_ids)==len(trusted_owner)==len(fitted_ids)):
        raise ValueError('Inputs must be equal-length 1D arrays')
    if np.any(points<0) or np.any(local_ids<=0) or np.any(fitted_ids<=0) or minimum<1 or not 0<fraction<=1:
        raise ValueError('Invalid inputs or thresholds')
    # Each point contributes once to the threshold even if a projection repeats it.
    result=fitted_ids.copy(); audit=[]
    for lid in np.unique(local_ids):
        s=np.flatnonzero(local_ids==lid)
        if len(np.unique(fitted_ids[s]))!=1: raise ValueError('Inconsistent fitted ID per mask')
        pairs=np.unique(np.column_stack((points[s],trusted_owner[s])),axis=0)
        # A point associated with more than one owner is treated as ambiguous.
        unique,counts=np.unique(pairs[:,0],return_counts=True)
        ambiguous=unique[counts>1]
        clean=pairs[~np.isin(pairs[:,0],ambiguous)]
        n=len(unique); positive=clean[clean[:,1]>0,1]
        ids,support=np.unique(positive,return_counts=True)
        significant=ids[(support>=minimum)&(support>=fraction*n)]
        fitted=int(fitted_ids[s[0]])
        if len(significant)>=2:
            mode='multi'; result[s]=np.where(np.isin(trusted_owner[s],significant),trusted_owner[s],-1)
            result[s[np.isin(points[s],ambiguous)]]=-1
        elif len(significant)==1 and int(significant[0])!=fitted:
            mode='single_mismatch'; result[s]=-1
        elif np.any((trusted_owner[s]>0)&(trusted_owner[s]!=fitted)) or np.any(trusted_owner[s]==-2):
            mode='partial_conflict'
            bad=((trusted_owner[s]>0)&(trusted_owner[s]!=fitted))|(trusted_owner[s]==-2)
            result[s[bad]]=-1
        else:
            mode='normal'
        audit.append(dict(mask_local_id=int(lid),mode=mode,projected_points=n,owners=ids.tolist(),owner_support=support.tolist(),significant=significant.tolist(),fitted_id=fitted,abstentions=int(np.count_nonzero(result[s]<0))))
    return result,audit

def strict_keys(points, pids, base):
    points=np.asarray(points,dtype=np.int64); pids=np.asarray(pids,dtype=np.int64)
    valid=pids>0
    if np.any(pids[valid]>=base): raise ValueError('ID exceeds base')
    pairs=np.unique(np.column_stack((points[valid],pids[valid])),axis=0)
    if len(pairs)==0: return np.empty(0,dtype=np.int64)
    pts,cnt=np.unique(pairs[:,0],return_counts=True)
    keep=np.isin(pairs[:,0],pts[cnt==1])
    return np.sort(pairs[keep,0]*base+pairs[keep,1])
