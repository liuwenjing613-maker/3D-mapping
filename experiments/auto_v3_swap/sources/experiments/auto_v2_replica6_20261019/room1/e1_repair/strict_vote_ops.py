"""Effective votes are reduced from the COMPLETE raw candidate set of a frame."""
import numpy as np
from one_vote import effective_keys, pair_keys


def strict_frame_keys(candidate_keys, base):
    keys=np.asarray(candidate_keys,np.int64)
    if keys.ndim!=1:raise ValueError('Frame candidate keys must be one dimensional')
    return effective_keys(keys//base,keys%base,base,'abstain')


def revised_frame(retained_points,retained_ids,new_points,new_ids,base):
    raw=np.union1d(pair_keys(retained_points,retained_ids,base),pair_keys(new_points,new_ids,base))
    return strict_frame_keys(raw,base),raw


def frame_delta(before_candidates,after_candidates,base):
    before=strict_frame_keys(before_candidates,base)
    after=strict_frame_keys(after_candidates,base)
    return before,after,np.setdiff1d(before,after),np.setdiff1d(after,before)
