"""GT-free manual core claims and source-pixel history reassociation."""
import numpy as np

def core_claims(core_points, target_ids, surface_count, instance_base):
    """Same target cores union; distinct target IDs leave a point ambiguous."""
    if len(core_points)!=len(target_ids):raise ValueError('Claim arrays differ')
    chunks=[]
    for p,pid in zip(core_points,target_ids):
        p=np.asarray(p,np.int64)
        if not 0<int(pid)<instance_base or np.any(p<0) or np.any(p>=surface_count):raise ValueError('Invalid claim')
        chunks.append(p*instance_base+int(pid))
    raw=np.unique(np.concatenate(chunks)) if chunks else np.empty(0,np.int64)
    point,first,n=np.unique(raw//instance_base,return_index=True,return_counts=True)
    owner=np.full(surface_count,-1,np.int32)
    unique=n==1;owner[point[unique]]=(raw[first[unique]]%instance_base).astype(np.int32)
    owner[point[~unique]]=-2
    return owner,raw

def retarget_selection(pixel_points, old_ids, retained_pixels, core_owner):
    """Only still-retained original pixels on uniquely owned cores may move."""
    point=np.asarray(pixel_points,np.int64);old=np.asarray(old_ids,np.int64);retained=np.asarray(retained_pixels,bool)
    if point.shape!=old.shape or point.shape!=retained.shape:raise ValueError('Pixel arrays differ')
    owner=core_owner[point]
    return retained&(old>0)&(owner>0)&(old!=owner)

def strict_frame(candidate_keys, instance_base):
    """One vote per point; multiple distinct identities abstain."""
    keys=np.unique(np.asarray(candidate_keys,np.int64));p=keys//instance_base
    _,inv,n=np.unique(p,return_inverse=True,return_counts=True)
    return keys[n[inv]==1]
