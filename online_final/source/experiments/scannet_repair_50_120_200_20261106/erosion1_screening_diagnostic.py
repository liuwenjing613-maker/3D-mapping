from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from PIL import Image

ROOT=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
OUT=ROOT/'erosion1_screen_diagnostic_20261106';OUT.mkdir(exist_ok=True)
SCENES=['scene0011_00','scene0050_00']
def groups(passes,n):
 rem=list(range(n));out=[]
 while rem:
  lead=rem[0];yes=[lead];no=[]
  for j in rem[1:]:(yes if passes[lead,j] else no).append(j)
  out.append(yes);rem=no
 return out
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=[];start=time.monotonic()
 for scene in SCENES:
  for method in ['main','online']:
   for checkpoint in ([200] if method=='main' else [50,120,200]):
    work=ROOT/method/scene/str(checkpoint)/('A' if method=='main' else 'B')/'candidates';stage=work/'v1';data=json.loads((stage/'grouping.json').read_text());seeds=data['seeds'];n=len(seeds);vertices=data['visibility_rule']['vertex_count'];t0=time.monotonic();visible={};uvs={}
    for frame in sorted({s['frame'] for s in seeds}):
     with np.load(work/'overmerge'/('frame%06d_visible.npz'%frame)) as z:visible[frame]=z['vertex_index'].astype(np.int32);uvs[frame]=z['uv'].astype(np.int32)
     assert np.all(visible[frame][1:]>visible[frame][:-1])
    with np.load(stage/'surface_supports.npz') as z:existing=z['indices'];offsets=z['offsets']
    with np.load(stage/'pair_metrics.npz') as z:orig=z['passes'].astype(bool);orig_iou=z['iou']
    baseline=groups(orig,n);assert baseline==[g['members'] for g in data['groups']]
    supports=[];radii=[];eroded_counts=[];full_counts=[]
    for i,s in enumerate(seeds):
     image=np.asarray(Image.open(stage/s['mask']))>0; yy,xx=np.nonzero(image);short=int(min(xx.max()-xx.min()+1,yy.max()-yy.min()+1));radius=max(1,round(short*.01));distance=distance_transform_edt(np.pad(image,1))[1:-1,1:-1];eroded=distance>radius;v=visible[s['frame']];uv=uvs[s['frame']]
     orig_support=v[image[uv[:,1],uv[:,0]]];assert np.array_equal(orig_support,existing[offsets[i]:offsets[i+1]])
     support=v[eroded[uv[:,1],uv[:,0]]];supports.append(support);radii.append(radius);eroded_counts.append(int(eroded.sum()));full_counts.append(int(image.sum()))
    offsets2=np.concatenate(([0],np.cumsum([len(s) for s in supports])));idx=np.concatenate(supports).astype(np.int32);mat=csr_matrix((np.ones(len(idx),np.int32),idx,offsets2),shape=(n,vertices));intersection=(mat@mat.T).toarray().astype(np.int32)
    cache={}
    def count_cross(i,frame):
     key=(i,frame)
     if key not in cache:
      arr=supports[i];vis=visible[frame];ind=np.searchsorted(vis,arr);good=ind<len(vis);cache[key]=int((vis[ind[good]]==arr[good]).sum())
     return cache[key]
    covis=np.zeros((n,n),np.float32);common_union=np.zeros((n,n),np.int32)
    for i in range(n):
     for j in range(i+1,n):
      v=int(intersection[i,j]);
      if not v:continue
      a=count_cross(i,seeds[j]['frame']);b=count_cross(j,seeds[i]['frame']);den=a+b-v;assert den>=v>0
      covis[i,j]=covis[j,i]=v/den;common_union[i,j]=common_union[j,i]=den
    new_common=(covis>.25);np.fill_diagonal(new_common,False);updated=orig|new_common;rerun=groups(updated,n);edge_new=new_common&~orig;edge_count=int(np.triu(edge_new,1).sum())
    def membership(gs):
     return {seeds[i]['id']:k for k,g in enumerate(gs) for i in g}
    before=membership(baseline);after=membership(rerun)
    pairs=[]
    for i,j in zip(*np.where(np.triu(edge_new,1))):
     pairs.append(dict(a=seeds[i]['id'],b=seeds[j]['id'],ordinary_iou=float(orig_iou[i,j]),eroded_common_iou=float(covis[i,j]),common_intersection=int(intersection[i,j]),common_union=int(common_union[i,j]),before_same_group=before[seeds[i]['id']]==before[seeds[j]['id']],after_same_group=after[seeds[i]['id']]==after[seeds[j]['id']]))
    result=dict(status='PASS',scope='fixed original checkpoint input; 1% erosion only on 3D covisible branch; original ordinary 3D IoU unchanged; fixed-primary greedy regrouping only',scene=scene,method=method,checkpoint=checkpoint,raw_frame_count=checkpoint,gt_used=False,erosion_percent=1,erosion_basis='rounded 1% of mask bounding-box shorter side, minimum 1 pixel, Euclidean distance transform',ordinary_iou_threshold_strict_gt=.25,eroded_common_iou_threshold_strict_gt=.25,min_common_intersection=1,input_masks=n,baseline_groups=len(baseline),replayed_groups=len(rerun),additional_qualifying_pairs=edge_count,additional_pairs_with_under_10_common_points=sum(x['common_intersection']<10 for x in pairs),radii_range=[int(min(radii)),int(max(radii))],previous_and_new_groups=[dict(primary=seeds[g[0]]['id'],members=[seeds[i]['id'] for i in g]) for g in rerun],new_pairs=pairs,source_files={'grouping_sha256':digest(stage/'grouping.json'),'pair_metrics_sha256':digest(stage/'pair_metrics.npz'),'surface_supports_sha256':digest(stage/'surface_supports.npz')},seconds=round(time.monotonic()-t0,2),online_caveat='each node uses old experiment mapping/repair history; not a new sequential online run' if method=='online' else None)
    folder=OUT/method/scene/str(checkpoint);folder.mkdir(parents=True,exist_ok=True);tmp=folder/'report.json.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(folder/'report.json');np.savez_compressed(folder/'pair_diagnostic.npz',original_pass=orig,original_iou=orig_iou,eroded_common_iou=covis,eroded_intersection=intersection,eroded_common_union=common_union,updated_pass=updated,erosion_radius=np.asarray(radii,np.int16),eroded_pixel_count=np.asarray(eroded_counts,np.int32),original_pixel_count=np.asarray(full_counts,np.int32))
    summary={k:result[k] for k in ['scene','method','checkpoint','input_masks','baseline_groups','replayed_groups','additional_qualifying_pairs','additional_pairs_with_under_10_common_points','radii_range','seconds']};rows.append(summary);print('DONE',json.dumps(summary),flush=True)
    (OUT/'status.json').write_text(json.dumps(dict(status='RUNNING',completed=len(rows),total=8,rows=rows),ensure_ascii=False))
 (OUT/'status.json').write_text(json.dumps(dict(status='PASS',completed=len(rows),total=8,rows=rows,seconds=round(time.monotonic()-start,2),repair_executed=False,final_evaluation_executed=False),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
