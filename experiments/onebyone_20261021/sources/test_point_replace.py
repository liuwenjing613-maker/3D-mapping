import json
import numpy as np
from point_replace import replace_frame_points


def oracle(old, new, base):
    a, b = {}, {}
    for key in old: a.setdefault(int(key)//base, set()).add(int(key)%base)
    for key in new: b.setdefault(int(key)//base, set()).add(int(key)%base)
    for point, ids in b.items():
        if len(ids) == 1: a[point] = ids
    return np.array(sorted(p*base+i for p, ids in a.items() for i in ids), np.int64)


def run():
    cases = [([110, 137, 210, 310], [1800, 1800], 1000),
             ([101,102,201,301], [109,209,208,409],100),
             ([101,201], [],100), ([],[109,109,209,208],100)]
    rng=np.random.default_rng(20261021)
    for _ in range(500):
        a=rng.integers(0,40,150)*100+rng.integers(1,15,150)
        b=rng.integers(0,40,40)*100+rng.integers(1,15,40)
        cases.append((a,b,100))
    for old,new,base in cases:
        result,retained,committed,conflict=replace_frame_points(old,new,base)
        np.testing.assert_array_equal(result,oracle(old,new,base))
        old=np.unique(old); covered=committed//base
        np.testing.assert_array_equal(result[~np.isin(result//base,covered)],old[~np.isin(old//base,covered)])
        np.testing.assert_array_equal(result[np.isin(result//base,conflict)],old[np.isin(old//base,conflict)])
        p,n=np.unique(old//base,return_counts=True); effective=p[n==1]
        p,n=np.unique(result//base,return_counts=True); after=p[n==1]
        assert not len(np.setdiff1d(effective,after))
        np.testing.assert_array_equal(replace_frame_points(result,new,base)[0],result)
    print(json.dumps({'status':'PASS','independent_set_oracle_cases':len(cases),'conflict_preservation':True,'no_effective_vote_loss':True,'idempotent':True}))

if __name__=='__main__':run()
