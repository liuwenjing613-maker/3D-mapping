"""GT-free visibility, multi-view seeded expansion and source observation revision."""
import numpy as np
from scipy.spatial import cKDTree

def visible_samples(xyz, camera_to_world, camera, depth, tolerance=.02):
    cam=(np.asarray(xyz)-camera_to_world[:3,3])@camera_to_world[:3,:3]
    positive=np.isfinite(cam).all(axis=1)&(cam[:,2]>0)
    u=np.zeros(len(cam),dtype=int);v=u.copy()
    u[positive]=np.rint(camera.fx*cam[positive,0]/cam[positive,2]+camera.cx).astype(int)
    v[positive]=np.rint(camera.fy*cam[positive,1]/cam[positive,2]+camera.cy).astype(int)
    inside=positive&(u>=0)&(v>=0)&(u<depth.shape[1])&(v<depth.shape[0])
    visible=np.zeros(len(cam),bool);i=np.flatnonzero(inside);d=depth[v[i],u[i]]
    visible[i]=np.isfinite(d)&(d>0)&(d<10)&(np.abs(d-cam[i,2])<=tolerance)
    return visible,u,v

def choose_view_representatives(centers, translation=.05):
    """Deterministic greedy separated camera poses; nearby repeated frames count once."""
    selected=[]
    for i in range(len(centers)):
        if all(np.linalg.norm(centers[i]-centers[j])>=translation for j in selected):selected.append(i)
    return np.asarray(selected,dtype=int)

def connected_expansion(xyz, core, eligible, radius=.02, normals=None, max_normal_angle_deg=45):
    """Expand only through independently qualified points; never through weak bridges."""
    core=np.asarray(core,bool);eligible=np.asarray(eligible,bool);allowed=core|eligible
    tree=cKDTree(np.asarray(xyz));chosen=core.copy();front=np.flatnonzero(core);rounds=[]
    cosine=np.cos(np.deg2rad(max_normal_angle_deg))
    while len(front):
        accepted=[]
        for i,neighbors in zip(front,tree.query_ball_point(np.asarray(xyz)[front],radius)):
            for j in neighbors:
                if not allowed[j] or chosen[j]:continue
                if normals is not None:
                    ni=np.asarray(normals[i]);nj=np.asarray(normals[j]);den=np.linalg.norm(ni)*np.linalg.norm(nj)
                    if den<=1e-9 or abs(float(np.dot(ni,nj)/den))<cosine:continue
                chosen[j]=True;accepted.append(j)
        front=np.asarray(accepted,dtype=int)
        if len(front):rounds.append(len(front))
    return chosen,rounds

def revise_history(points, old_ids, available, prior_owner, expanded_book, visible, book_id=817):
    """Return per-source-pixel IDs; a different pixel sharing a key remains independent."""
    ids=np.asarray(old_ids).copy();available=np.asarray(available,bool)
    prior=prior_owner[points];take=available&(prior>0)&(ids!=prior);ids[take]=prior[take]
    expand=available&expanded_book[points]&visible[points]&(ids!=book_id)
    ids[expand]=book_id
    return ids,expand

def constrain_new(points, ids, owner, visible, book_id=817):
    """Remove conflicting source pixels only. Ambiguous book overlap is reported separately."""
    o=owner[points];focus=(ids==book_id)|(o==book_id)
    reject=visible[points]&focus&(o>0)&(ids!=o)
    ambiguous=(ids==book_id)&(o==-2)&visible[points]
    return reject,ambiguous

def keyset(points, ids, base):
    return np.unique(np.asarray(points,dtype=np.int64)*base+np.asarray(ids,dtype=np.int64))

def strict_frame(keys, base):
    keys=np.unique(np.asarray(keys,dtype=np.int64));_,inv,n=np.unique(keys//base,return_inverse=True,return_counts=True)
    return keys[n[inv]==1]

def mask_support(xyz,camera_to_world,camera,depth,mask,tolerance=.02):
    visible,u,v=visible_samples(xyz,camera_to_world,camera,depth,tolerance)
    positive=np.zeros(len(xyz),bool);positive[visible]=mask[v[visible],u[visible]]
    return visible,positive

def resolve_claims(point_sets,identities,npoints):
    owner=np.full(npoints,-1,np.int32)
    for points,pid in zip(point_sets,identities):
        p=np.unique(points);old=owner[p];free=old==-1;conflict=(old!=-1)&(old!=pid)
        owner[p[free]]=pid;owner[p[conflict]]=-2
    return owner

def revise_all_history(points,old_ids,available,prior_owner,owner,visible):
    ids=np.asarray(old_ids).copy();prior=prior_owner[points]
    take=available&(prior>0);ids[take]=prior[take]
    o=owner[points];change=available&visible[points]&(o>0)&(ids!=o)
    ids[change]=o[change]
    return ids,change

def constrain_all_new(points,ids,owner,visible):
    o=owner[points];reject=visible[points]&(o>0)&(ids!=o)
    ambiguous=visible[points]&(o==-2)
    return reject,ambiguous
