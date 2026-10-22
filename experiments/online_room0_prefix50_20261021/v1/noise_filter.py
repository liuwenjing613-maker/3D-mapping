"""Conservative whole-mask morphology screening in native 1200x680 pixels.

Distances measure each foreground pixel to the nearest background pixel,
not RGB gradients. Features are evidence for review, not semantic proof.
Retained masks are never eroded or otherwise edited.
"""
import numpy as np
from scipy import ndimage as ndi

RULE = {
    'area_lt': 100,
    'max_core_fraction': .05,
    'thin_radius': 3, 'thin_min_extent': 80,
    'sparse_radius': 8, 'sparse_max_bbox_fill': .10, 'sparse_min_extent': 80,
    'large_sparse_radius': 12, 'large_sparse_max_bbox_fill': .05, 'large_sparse_min_extent': 200,
}

def shape_features(mask):
    mask = np.asarray(mask, dtype=bool)
    y,x=np.nonzero(mask)
    assert len(x)>0
    w,h=int(x.max()-x.min()+1),int(y.max()-y.min()+1)
    sub=np.pad(mask[y.min():y.max()+1,x.min():x.max()+1],1)
    distance=ndi.distance_transform_edt(sub)
    inside=distance[sub]
    labels,n=ndi.label(sub,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel())[1:]
    return dict(area=int(len(x)),bbox_width=w,bbox_height=h,bbox_area=w*h,
        extent=max(w,h),bbox_fill=float(len(x)/(w*h)),components=int(n),
        largest_component=int(sizes.max()),max_radius=float(inside.max()),
        p95_radius=float(np.quantile(inside,.95)),
        core_pixels={str(r):int(np.sum(inside>r)) for r in [2,3,8,12]})

def classify(f):
    area=f['area'];core=f['core_pixels'];extent=f['extent'];box=f['bbox_area']
    flags=[]
    if area<100:flags.append('AREA_LT_100')
    if extent>=80 and 20*core['3']<=area:flags.append('THIN_EDGE_BAND')
    if extent>=80 and 10*area<=box and 20*core['8']<=area:flags.append('SPARSE_EDGE_OUTLINE')
    if extent>=200 and 20*area<=box and 20*core['12']<=area:flags.append('LARGE_SPARSE_EDGE_OUTLINE')
    # Priority makes counts mutually exclusive; keep every matching condition.
    reason=flags[0] if flags else None
    return dict(exclude=bool(flags),reason=reason,matched_rules=flags)

def is_review_control(f):
    return not classify(f)['exclude'] and (f['area']<=1000 or
        f['extent']>=80 and f['bbox_fill']<=.20 and f['p95_radius']<=12)
