"""Frame-level set operations preserve shared votes when observations change."""
import numpy as np
from evidence_replacement import EvidenceReplacement

def keys_from_pairs(points,ids,base):
    points=np.asarray(points,np.int64);ids=np.asarray(ids,np.int64)
    assert points.shape==ids.shape and np.all(points>=0) and np.all((ids>0)&(ids<base))
    return np.unique(points*base+ids)

def corrected_frame(old_points,old_local,old_table,retired_local_ids,new_points,new_ids,base):
    keep=~np.isin(old_local,retired_local_ids)
    retained=keys_from_pairs(old_points[keep],old_table[old_local[keep]],base)
    new=keys_from_pairs(new_points,new_ids,base)
    return np.union1d(retained,new),retained

def batch_delta(keys,votes,removed,added):
    k,v=EvidenceReplacement.add_delta(keys,votes,removed,-1)
    return EvidenceReplacement.add_delta(k,v,added,1)

def self_test():
    base=1000
    p=np.array([0,0,1,2,3]);local=np.array([1,2,1,3,4])
    table=np.array([0,52,52,9,10])
    old=keys_from_pairs(p,table[local],base)
    new,keep=corrected_frame(p,local,table,[1],np.array([1,4]),np.array([355,355]),base)
    expected=np.array([52,1355,2009,3010,4355])
    np.testing.assert_array_equal(new,expected)
    # A second untouched observation still supplies (point 0, ID 52).
    assert 52 in keep and 52 in new
    votes=np.array([3,2,5,1],np.int64)
    removed=np.setdiff1d(old,new);added=np.setdiff1d(new,old)
    k,v=batch_delta(old,votes,removed,added)
    original=EvidenceReplacement(old,votes)
    assert original.replace('f0',old,new)
    np.testing.assert_array_equal(k,original.keys);np.testing.assert_array_equal(v,original.votes)
    assert not original.replace('f0',old,new)
    back_k,back_v=batch_delta(k,v,added,removed)
    np.testing.assert_array_equal(back_k,old);np.testing.assert_array_equal(back_v,votes)
    try:batch_delta(old,votes,np.array([999999]),np.array([999999]))
    except ValueError:pass
    else:raise AssertionError('Missing original vote must fail before addition')

if __name__=='__main__':
    self_test();print('Shared-vote preservation, frame deduplication, idempotency and exact rollback PASS')
