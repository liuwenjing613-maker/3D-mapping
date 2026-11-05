"""Map-versus-seed measurements only; no skip decision or human labels."""
import numpy as np


def measure(support, visible, labels, state):
    """S includes unassigned vertices; I is the entire source-visible instance."""
    assert support.ndim==visible.ndim==1 and len(support)>0
    assert np.all(support[:-1]<support[1:]) and np.all(visible[:-1]<visible[1:])
    at=np.searchsorted(visible,support)
    assert np.all(at<len(visible)) and np.array_equal(visible[at],support)
    seed_labels=labels[support]
    positive=seed_labels[seed_labels>0]
    ids,counts=np.unique(positive,return_counts=True)
    order=np.lexsort((ids,-counts))
    ids=ids[order];counts=counts[order]
    instance=int(ids[0]) if len(ids) else None
    intersection=int(counts[0]) if len(counts) else 0
    instance_visible=int(np.sum(labels[visible]==instance)) if instance is not None else 0
    p=intersection/len(support)
    r=intersection/instance_visible if instance_visible else 0.
    return dict(support_count=len(support),instance_id=instance,intersection=intersection,
        instance_visible_count=instance_visible,p=p,r=r,
        both95=bool(instance is not None and 20*intersection>=19*len(support) and 20*intersection>=19*instance_visible),
        tied_largest_ids=ids[counts==intersection].astype(int).tolist(),
        other_instance_support=[{'instance_id':int(i),'count':int(n)} for i,n in zip(ids[1:],counts[1:])],
        final_unassigned_count=int(np.sum(seed_labels<=0)),
        native_state_counts={name:int(np.sum(state[support]==code)) for name,code in [('unobserved',0),('tentative',1),('confirmed',2),('conflict',3)]},
        assigned_but_native_conflict_count=int(np.sum((state[support]==3)&(seed_labels>0))))
