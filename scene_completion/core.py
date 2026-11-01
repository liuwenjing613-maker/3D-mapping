"""GT-free single-pass completion. Only immutable original CONFIRMED seeds."""
from dataclasses import dataclass,asdict
from collections import Counter
import time
import numpy as np
from scipy.spatial import cKDTree

U,T,CONFIRMED,CONFLICT=0,1,2,3
@dataclass(frozen=True)
class Settings:
    neighbors:int=20
    radius_m:float=.025
    normal_cos:float=.92
    plane_tolerance_m:float=.003
    min_dominance:float=.90
    min_same_id:int=6
    chunk_size:int=10000
    workers:int=4

def complete(xyz,normals,labels,evidence,settings=Settings()):
    start=time.perf_counter()
    n=len(xyz); xyz=np.asarray(xyz); normals=np.asarray(normals,dtype=np.float64)
    assert xyz.shape==normals.shape==(n,3) and labels.shape==(n,)
    assert np.isfinite(xyz).all() and np.isfinite(normals).all()
    norms=np.linalg.norm(normals,axis=1)
    assert np.all(norms>.5)
    normals=normals/norms[:,None]
    state=evidence['state']; top=evidence['top1_instance_id']
    assert state.shape==top.shape==(n,) and np.isin(state,[0,1,2,3]).all()
    seed=np.flatnonzero(state==CONFIRMED)
    assert len(seed)>0 and np.array_equal(labels[seed],top[seed]) and np.all(labels[seed]>0)
    target=np.flatnonzero(labels==-1)
    result=labels.copy()
    candidate=np.full(len(target),-1,np.int32)
    support=np.zeros(len(target),np.int16)
    valid_count=np.zeros(len(target),np.int16)
    dominance=np.zeros(len(target),np.float32)
    reason=np.full(len(target),1,np.uint8)
    tree=cKDTree(xyz[seed])
    nearest=tree.query(xyz[target],k=1,workers=settings.workers)[0] if len(target) else np.empty(0)
    for lo in range(0,len(target),settings.chunk_size):
        hi=min(lo+settings.chunk_size,len(target));q=target[lo:hi]
        d,j=tree.query(xyz[q],k=settings.neighbors,distance_upper_bound=settings.radius_m,workers=settings.workers)
        j=np.atleast_2d(j); d=np.atleast_2d(d)
        good=np.isfinite(d)&(j<len(seed))
        ix=seed[np.minimum(j,len(seed)-1)]
        delta=xyz[ix]-xyz[q,None]
        nq=normals[q,None];nj=normals[ix]
        dot=np.sum(nq*nj,axis=2)
        plane_q=np.abs(np.sum(delta*nq,axis=2))
        plane_j=np.abs(np.sum(delta*nj,axis=2))
        good &= (dot>=settings.normal_cos)&(plane_q<=settings.plane_tolerance_m)&(plane_j<=settings.plane_tolerance_m)
        ids=labels[ix]
        counts=np.sum((ids[:,:,None]==ids[:,None,:])&good[:,None,:],axis=2)
        counts[~good]=0
        choice=np.argmax(counts,axis=1);rows=np.arange(len(q))
        best=counts[rows,choice];total=good.sum(axis=1)
        frac=np.divide(best,total,out=np.zeros(len(q)),where=total>0)
        cid=np.where(best>0,ids[rows,choice],-1)
        candidate[lo:hi]=cid;support[lo:hi]=best;valid_count[lo:hi]=total;dominance[lo:hi]=frac
        r=np.zeros(len(q),np.uint8)
        r[total==0]=1
        r[(r==0)&(best<settings.min_same_id)]=2
        r[(r==0)&(frac<settings.min_dominance)]=3
        r[(r==0)&(state[q]==T)&(cid!=top[q])]=4
        r[(r==0)&(state[q]==CONFLICT)]=5
        assert not np.any(state[q]==CONFIRMED)
        result[q[r==0]]=cid[r==0]
        reason[lo:hi]=r
    assert np.array_equal(result[labels!=-1],labels[labels!=-1])
    changed=(result!=labels)
    assert np.all(labels[changed]==-1) and np.all(result[changed]>0)
    assert set(np.unique(result[result>0])).issubset(set(np.unique(labels[labels>0])))
    reasons={0:'accepted',1:'no_geometric_confirmed_neighbor',2:'fewer_than_6_same_id',3:'dominance_below_90pct',4:'tentative_top1_mismatch',5:'conflict_needs_projective_evidence'}
    stats={'seconds_compute':time.perf_counter()-start,'settings':asdict(settings),'original_confirmed_seeds':len(seed),'target_points':len(target),'newly_assigned':int(changed.sum()),'immutable_existing_labels':True,'recursive_propagation':False,'by_state':{}}
    for value,name in [(U,'U'),(T,'T'),(CONFLICT,'CONFLICT')]:
        use=state[target]==value
        rem=use&(reason!=0)
        ds=nearest[rem]
        stats['by_state'][name]={'target':int(use.sum()),'newly_assigned':int(np.count_nonzero(use&(reason==0))),'remaining':int(rem.sum()),'reasons':{reasons[int(k)]:int(v) for k,v in Counter(reason[use].tolist()).items()},'remaining_nearest_confirmed_distance_quantiles_m':dict(zip(['p0','p25','p50','p75','p90','p99','p100'],np.quantile(ds,[0,.25,.5,.75,.9,.99,1]).tolist())) if len(ds) else {},'remaining_within_25mm':int(np.count_nonzero(ds<=settings.radius_m))}
    diagnostics={'target_index':target.astype(np.int32),'candidate_id':candidate,'same_id_count':support,'valid_neighbor_count':valid_count,'dominance':dominance,'rejection_code':reason,'nearest_confirmed_distance_m':nearest.astype(np.float32)}
    return result,diagnostics,stats
