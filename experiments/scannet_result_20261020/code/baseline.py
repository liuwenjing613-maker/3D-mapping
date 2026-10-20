"""Rebuild frozen forward-projection E1 evidence and strict abstention on ScanNet."""
from pathlib import Path
from dataclasses import asdict,replace
import sys,json,hashlib,time,gc
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import save_npz
from instance_repair.cli import verify_sources
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.surface_evidence import project_frame_regions,frame_instance_keys,reduce_surface_votes
from revisable_instance_map.offline_surface_assignment import AssignmentSettings,assign_surface,surface_edges
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,default=lambda x:x.item()))
def arr(p,**z):
    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**z)
def main(scene):
    verify_sources()
    out=R/'baseline/ablation'/scene
    if (out/'mapping_complete.json').exists():return
    config=C/'configs'/(scene+'.json');source=ReplicaFrameSource(config)
    geometry=R/'inputs'/scene/'geometry/surface.ply';cloud=o3d.io.read_point_cloud(str(geometry))
    xyz=np.asarray(cloud.points).astype(np.float32);rgb=np.asarray(cloud.colors).astype(np.float32)
    normals=np.asarray(cloud.normals).astype(np.float32);assert normals.shape==xyz.shape
    tree=cKDTree(xyz.astype(np.float64))
    native=R/'compat/native'/scene/'P1-A1/surface_p0'
    assoc=native.parent/'association/associations.jsonl'
    rows=[json.loads(x) for x in assoc.read_text().splitlines()]
    lookup={(x['frame_id'],x['mask_local_id']):x['instance_id'] for x in rows}
    base=max(lookup.values())+1
    allkeys=[];strictkeys=[];records=[];collision=0
    for index,frame in enumerate(source.iter_frames()):
        fid=frame.frame_id;points,local,stats=project_frame_regions(frame,tree,pixel_stride=2,max_distance_m=.015,workers=8)
        table=np.full(int(frame.mask_local.max())+1,-1,np.int32)
        for mid in np.unique(frame.mask_local):
            if mid:table[mid]=lookup[fid,int(mid)]
        assert (table[local]>0).all()
        keys=frame_instance_keys(points,local,table,base);allkeys.append(keys)
        pid,cnt=np.unique(keys//base,return_counts=True);bad=pid[cnt>1];collision+=int(np.sum(cnt-1))
        strict=keys[~np.isin(keys//base,bad)];assert len(strict)==len(np.unique(strict//base))
        strictkeys.append(strict)
        maskp=source.mask_root/source.config['source']['mask_pattern'].format(frame=fid)
        support=native/'frame_region_support'/f'f{fid:06d}.npz'
        arr(support,frame_id=np.array([fid],np.int32),surface_point_index=points,mask_local_id=local,
            source_mask_sha256=np.array([sha(maskp)]),pixel_stride=np.array([2],np.int32),max_surface_distance_m=np.array([.015]))
        records.append(dict(frame_id=fid,frame_instance_votes=len(keys),support_file=str(support),support_sha256=sha(support),**stats))
        if index%40==0:print('BASELINE_PROJECT',scene,index+1,len(source.frame_ids),flush=True)
    report=dict(status='PASS',scene_id=scene,ground_truth_used=False,frame_count=len(records),tsdf_surface_points=len(xyz),
        config_sha256=sha(config),association_decisions_sha256=sha(assoc),tsdf_surface_sha256=sha(geometry),
        pixel_stride=2,max_surface_distance_m=.015,min_confirmed_votes=2,min_confirmed_ratio=.67,
        frame_records=records,same_frame_multi_id_points=collision,projection_direction='source_pixels_to_fixed_surface',
        source_geometry='unchanged previous engineering-corrected ScanNet TSDF',
        historical_difference='Previous ScanNet used reverse projection; this run applies current E1 forward projection and strict one-vote.')
    settings=AssignmentSettings();settings_sha=hashlib.sha256(json.dumps(asdict(settings),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    predictions={}
    for policy,chunks in [('original',allkeys),('abstain',strictkeys)]:
        keys,votes=np.unique(np.concatenate(chunks),return_counts=True)
        evidence=reduce_surface_votes(keys,votes,len(xyz),base,2,.67)
        labels=np.where(evidence['state']==2,evidence['top1_instance_id'],-1).astype(np.int32)
        pair=dict(surface_point_index=(keys//base).astype(np.int32),instance_id=(keys%base).astype(np.int32),frame_votes=votes.astype(np.int32))
        destination=native if policy=='original' else out/policy
        arr(destination/'surface_evidence.npz',xyz_m=xyz,rgb=rgb,**evidence)
        arr(destination/'surface_instance_frame_votes.npz',**pair)
        nativefile=native/'instance_surface.npz' if policy=='original' else destination/'native/instance_surface.npz'
        arr(nativefile,xyz_m=xyz,rgb=rgb,instance_id=labels)
        print('BASELINE_POSTPROCESS',scene,policy,flush=True)
        variants,_,stats=assign_surface(xyz,normals,rgb,evidence,pair,labels,settings,
            progress=lambda s,d:print('POST',scene,policy,s,flush=True))
        finalfile=(R/'compat/final'/scene/'P1-A1/diffusion/holes_geodesic/instance_surface.npz') if policy=='original' else destination/'final/instance_surface.npz'
        final=variants['holes_geodesic'];inventory=np.union1d(np.unique(labels[labels>0]),np.unique(final[final>0])).astype(np.int32)
        arr(finalfile,xyz_m=xyz,rgb=rgb,instance_id=final,native_instance_ids=inventory)
        if policy=='abstain':arr(nativefile,xyz_m=xyz,rgb=rgb,instance_id=labels,native_instance_ids=inventory)
        for stage,p in [('native',nativefile),('final',finalfile)]:
            predictions[policy+'_'+stage]=dict(path=str(p),sha256=sha(p),inventory=inventory.tolist())
        if policy=='original':
            save(finalfile.parent.parent/'materialization_report.json',dict(settings_sha256=settings_sha,settings=asdict(settings),statistics=stats))
            report.update(instance_surface_sha256=sha(nativefile),surface_evidence_sha256=sha(native/'surface_evidence.npz'))
            save(native/'materialization_report.json',report)
        del variants,evidence;gc.collect()
    freeze=dict(status='PREDICTIONS_FROZEN_BEFORE_GT',GT_read_by_mapper=False,scenes={scene:dict(predictions=predictions)},
        source_hashes={str(config):sha(config),str(geometry):sha(geometry),str(assoc):sha(assoc)},frame_ids=list(source.frame_ids))
    save(R/'baseline'/(scene+'_predictions_freeze.json'),freeze)
    # Same surface-neighbour graph used by the validated C/U/T ROI selector.
    normals=normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    dist=tree.query(xyz,k=5,workers=8)[0][:,4];median=float(np.median(dist));scale=np.clip(dist,.75*median,1.25*median)
    graph_settings=replace(settings,normal_cos=.20,plane_tolerance_scale=.80,color_cost_weight=0.,normal_cost_weight=0.,query_chunk=25000)
    print('BUILD_ROI_GRAPH',scene,flush=True)
    graph,_=surface_edges(tree,xyz,normals,rgb,scale,np.arange(len(xyz),dtype=np.int32),graph_settings)
    graph=graph.maximum(graph.T).tocsr();graph.sort_indices()
    save_npz(R/'inputs'/scene/'review_surface_graph_n02_p08.npz',graph)
    save(out/'mapping_complete.json',dict(status='PASS',scene=scene,frame_count=len(records),strict_one_vote=True,
        mapping_GT_used=False,geometry_sha256=sha(geometry),frame_ids=list(source.frame_ids),predictions=predictions,
        code_sha256={str(p):sha(p) for p in [Path(__file__),C/'shared/frame_io.py',C/'shared/run_repairs.py']}))
    print('BASELINE_COMPLETE',scene,flush=True)
if __name__=='__main__':main(sys.argv[1])
