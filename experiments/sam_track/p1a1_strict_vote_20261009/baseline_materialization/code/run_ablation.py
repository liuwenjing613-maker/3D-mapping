"""GT-free three-policy ablation on the exact frozen P1-A1 surface and IDs."""
from pathlib import Path
from dataclasses import asdict
import gc
import hashlib
import json
import os
import sys
import time
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from one_vote import pair_keys, ambiguous_points, effective_keys, supported_depth_preferences

TASK = Path(__file__).resolve().parent
SNAPSHOT = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
BASE = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
INPUT = Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')
OUT = Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
sys.path.insert(0, str(SNAPSHOT/'revisable_instance_map/src'))
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.surface_evidence import reduce_surface_votes, CONFIRMED, STATE_NAMES
from revisable_instance_map.surface_projective_evidence import project_surface_regions
from revisable_instance_map.offline_surface_assignment import assign_surface, AssignmentSettings


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def emit(stage, **values):
    row = {'stage':stage,'updated_unix':time.time(),**values}
    dump(OUT/'ablation_progress.json',row)
    print(json.dumps(row),flush=True)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def image_coordinates(frame, xyz):
    camera = (xyz.astype(np.float64)-frame.camera_to_world[:3,3]) @ frame.camera_to_world[:3,:3]
    z = camera[:,2]
    good = np.isfinite(z)&(z>0)
    u,v = np.zeros(len(z),np.int32),np.zeros(len(z),np.int32)
    u[good] = np.rint(frame.camera.fx*camera[good,0]/z[good]+frame.camera.cx).astype(np.int32)
    v[good] = np.rint(frame.camera.fy*camera[good,1]/z[good]+frame.camera.cy).astype(np.int32)
    good &= (u>=0)&(u<frame.camera.width)&(v>=0)&(v<frame.camera.height)
    return u,v,good


def boundary_distances(mask):
    boundary = np.zeros(mask.shape,bool)
    horizontal = mask[:,1:] != mask[:,:-1]
    vertical = mask[1:,:] != mask[:-1,:]
    boundary[:,1:] |= horizontal; boundary[:,:-1] |= horizontal
    boundary[1:,:] |= vertical; boundary[:-1,:] |= vertical
    return distance_transform_edt(~boundary)


def render_example(frame, bad_points, xyz, preferred, base, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    u,v,inside = image_coordinates(frame,xyz[bad_points])
    resolved = np.isin(bad_points,preferred//base)
    palette = np.random.default_rng(20261009).uniform(.15,.95,(int(frame.mask_local.max())+1,3))
    palette[0] = [.3,.3,.3]
    fig,axes = plt.subplots(1,3,figsize=(18,4.8))
    axes[0].imshow(frame.rgb)
    axes[0].scatter(u[inside],v[inside],s=5,c='red',linewidths=0)
    axes[0].set_title('RGB: colliding surface points in red')
    axes[1].imshow(palette[frame.mask_local])
    axes[1].scatter(u[inside],v[inside],s=5,c='red',linewidths=0)
    axes[1].set_title('Frozen frame-local masks')
    axes[2].imshow(frame.rgb)
    for selected,color,label in ((inside&resolved,'lime','Depth resolved'),(inside&~resolved,'magenta','Abstain')):
        axes[2].scatter(u[selected],v[selected],s=7,c=color,linewidths=0,label=label)
    axes[2].legend(loc='upper right',fontsize=8)
    axes[2].set_title('Only existing collisions are processed')
    for ax in axes:ax.axis('off')
    fig.suptitle(f'{path.parent.name}, frame {frame.frame_id}: {len(bad_points)} point-frame collisions')
    fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=130,bbox_inches='tight')
    plt.close(fig)


def scene_mapping(scene):
    started = time.monotonic()
    folder = OUT/'ablation'/scene
    complete = folder/'mapping_complete.json'
    if complete.exists():
        result = json.loads(complete.read_text())
        assert result['GT_read'] is False and result['task_code_sha256'] == sha(__file__)
        for rec in result['predictions'].values():assert sha(rec['path'])==rec['sha256']
        return result
    folder.mkdir(parents=True,exist_ok=True)
    audit = json.loads((OUT/f'{scene}_frozen_vote_audit.json').read_text())
    hashes = dict(audit['source_sha256'])
    assert all(sha(path)==digest for path,digest in hashes.items())
    native = BASE/f'native/{scene}/P1-A1/surface_p0'
    final_path = BASE/f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
    final_report_path = final_path.parent.parent/'materialization_report.json'
    report = json.loads((native/'materialization_report.json').read_text())
    config_path = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs')/f'{scene}.json'
    assert sha(config_path)==report['config_sha256']
    source = ReplicaFrameSource(config_path)
    assert list(source.frame_ids)==list(range(0,2000,5))
    geometry = INPUT/scene/'geometry/surface.ply'
    assert sha(geometry)==report['tsdf_surface_sha256']
    for path in (config_path,geometry,final_path,final_report_path,source.scene_root/source.config['source']['trajectory']):hashes[str(path)]=sha(path)
    original = load(native/'instance_surface.npz')
    final = load(final_path)
    evidence = load(native/'surface_evidence.npz')
    pair = load(native/'surface_instance_frame_votes.npz')
    xyz,rgb = original['xyz_m'],original['rgb']
    for name in ('xyz_m','rgb'):
        np.testing.assert_array_equal(original[name],final[name])
        np.testing.assert_array_equal(original[name],evidence[name])
    cloud = o3d.io.read_point_cloud(str(geometry))
    assert cloud.has_normals() and len(cloud.points)==len(xyz)
    np.testing.assert_allclose(np.asarray(cloud.points),xyz,atol=1e-5,rtol=0)
    normals = np.asarray(cloud.normals).astype(np.float32)
    del cloud
    lookup = {}; maximum=0
    assoc_path = BASE/f'native/{scene}/P1-A1/association/associations.jsonl'
    for line in assoc_path.read_text().splitlines():
        row=json.loads(line);key=(int(row['frame_id']),int(row['mask_local_id']))
        assert key not in lookup
        lookup[key]=int(row['instance_id']);maximum=max(maximum,int(row['instance_id']))
    base=maximum+1
    keys = pair['surface_point_index'].astype(np.int64)*base+pair['instance_id']
    order=np.argsort(keys,kind='stable');keys=keys[order]
    original_votes=pair['frame_votes'][order].astype(np.int64)
    counts={'original':np.zeros_like(original_votes),'abstain':original_votes.copy(),'depth':original_votes.copy()}
    bad_count=np.zeros(len(xyz),np.int32)
    preferred_count=np.zeros(len(xyz),np.int32)
    rows=[];collision_distances=[];control_hits={str(t):0.0 for t in (0,1,2,5)};control_denominator=0
    audit_rows=sorted(audit['per_frame'],key=lambda r:(-r['distinct_multi_id_point_frame_events'],r['frame_id']))
    examples=[]
    for row in audit_rows:
        if all(abs(row['frame_id']-f)>=200 for f in examples):examples.append(row['frame_id'])
        if len(examples)==3:break
    emit('frame_policy_reduction',scene=scene,frames=0)
    for index,record in enumerate(report['frame_records']):
        fid=int(record['frame_id']);support_path=Path(record['support_file'])
        assert sha(support_path)==record['support_sha256']
        raw=load(support_path);points=raw['surface_point_index'];local=raw['mask_local_id']
        assert int(raw['frame_id'][0])==fid
        hashes[str(support_path)]=record['support_sha256']
        for name in ('rgb','depth','mask'):
            path=(source.mask_root/source.config['source']['mask_pattern'].format(frame=fid) if name=='mask'
                  else source.scene_root/source.config['source'][name+'_pattern'].format(frame=fid))
            digest=sha(path);hashes[str(path)]=digest
            if name=='mask':assert digest==str(raw['source_mask_sha256'][0])
        frame=source.load_frame(fid)
        table=np.full(int(frame.mask_local.max())+1,-1,np.int64)
        for mid in np.unique(frame.mask_local):
            if mid>0:table[mid]=lookup[(fid,int(mid))]
        frame_keys=pair_keys(points,table[local],base)
        np.testing.assert_equal(len(frame_keys),record['frame_instance_votes'])
        positions=np.searchsorted(keys,frame_keys)
        np.testing.assert_array_equal(keys[positions],frame_keys)
        counts['original'][positions]+=1
        bad_points=ambiguous_points(frame_keys,base)
        bad_count[bad_points]+=1
        if len(bad_points):
            relative,pixel_local,depth_stats=project_surface_regions(frame,xyz[bad_points],
                depth_tolerance_m=float(report['max_surface_distance_m']))
            preferred=supported_depth_preferences(points,local,bad_points[relative],pixel_local,table,base)
        else:preferred=np.empty(0,np.int64);depth_stats={}
        strict=effective_keys(points,table[local],base,'abstain')
        depth=effective_keys(points,table[local],base,'depth',preferred)
        assert len(strict)==len(np.unique(strict//base))
        assert len(depth)==len(np.unique(depth//base))
        # Raw observations are immutable; only full-frame effective votes change.
        for policy,new_keys in (('abstain',strict),('depth',depth)):
            removed=np.setdiff1d(frame_keys,new_keys,assume_unique=True)
            assert np.all(np.isin(removed//base,bad_points))
            counts[policy][np.searchsorted(keys,removed)]-=1
        preferred_count[preferred//base]+=1
        u,v,inside=image_coordinates(frame,xyz[bad_points])
        dist=boundary_distances(frame.mask_local)
        collision_distances.append(dist[v[inside],u[inside]])
        unambiguous=np.unique(strict//base)
        if len(unambiguous):
            samples=unambiguous[np.linspace(0,len(unambiguous)-1,min(1024,len(unambiguous)),dtype=np.int64)]
            cu,cv,ci=image_coordinates(frame,xyz[samples])
            values=dist[cv[ci],cu[ci]]
            weight=len(unambiguous)/len(samples)
            control_denominator+=int(np.count_nonzero(ci))*weight
            for threshold in (0,1,2,5):control_hits[str(threshold)]+=int(np.count_nonzero(values<=threshold))*weight
        row={'frame_id':fid,'colliding_point_frames':len(bad_points),'depth_resolved_point_frames':len(preferred),
             'original_votes':len(frame_keys),'abstain_votes':len(strict),'depth_votes':len(depth),
             'inside_image_collisions':int(np.count_nonzero(inside)),'depth_projection':depth_stats,
             'visible_positive_pixel_without_saved_raw_region_support':int(len(relative)-len(preferred)) if len(bad_points) else 0}
        rows.append(row)
        collision_file=folder/'collision_frames'/f'f{fid:06d}.npz';collision_file.parent.mkdir(exist_ok=True)
        np.savez_compressed(collision_file,colliding_surface_point_index=bad_points.astype(np.int32),
            depth_preferred_surface_point_index=(preferred//base).astype(np.int32),
            depth_preferred_instance_id=(preferred%base).astype(np.int32))
        if fid in examples:render_example(frame,bad_points,xyz,preferred,base,OUT/'visuals'/scene/f'f{fid:06d}.png')
        if (index+1)%100==0:emit('frame_policy_reduction',scene=scene,frames=index+1)
    np.testing.assert_array_equal(counts['original'],original_votes)
    assert int(bad_count.sum())==audit['distinct_multi_id_point_frame_events']
    assert int(np.count_nonzero(bad_count))==audit['distinct_multi_id_surface_points']
    distances=np.concatenate(collision_distances)
    collision_mask=bad_count>0
    distribution={'GT_read':False,'total_colliding_point_frames':int(bad_count.sum()),
        'depth_resolved_point_frames':int(preferred_count.sum()),
        'depth_resolution_fraction':float(preferred_count.sum()/bad_count.sum()),
        'boundary_distance_quantiles_px':{str(q):float(np.quantile(distances,q)) for q in (.1,.5,.9,.95)},
        'collision_boundary_fraction':{str(t):float(np.mean(distances<=t)) for t in (0,1,2,5)},
        'noncollision_boundary_fraction_estimated':{t:control_hits[t]/control_denominator for t in control_hits},
        'noncollision_control_sampling':'deterministic up to 1024 points/frame, weighted by frame point count; distribution diagnostic only',
        'collision_surface_baseline_states':{name:int(np.count_nonzero(collision_mask&(evidence['state']==i))) for i,name in enumerate(STATE_NAMES)},
        'collision_surface_baseline_assigned_points':int(np.count_nonzero(collision_mask&(original['instance_id']>0))),
        'visual_frame_selection':'highest collision counts, three frames separated by >=200 raw frames, selected before GT',
        'visual_frame_ids':examples,'per_frame':rows}
    discovery=json.loads(Path('/data/chenkejun/CVPR/results/manual_v3_comparison_20261009/source_discovery.json').read_text())
    scope_file=Path(discovery[scene]['root'])/'validated/full_track/allowed_surface_ids.npz'
    if scope_file.exists():
        scope=load(scope_file)['surface_point_index'];assert np.all((scope>=0)&(scope<len(xyz)))
        hashes[str(scope_file)]=sha(scope_file)
        scope_mask=np.zeros(len(xyz),bool);scope_mask[scope]=True
        distribution['repair_allowed_scope']={'source':str(scope_file),'surface_points':len(scope),
            'colliding_surface_points_in_scope':int(np.count_nonzero(collision_mask&scope_mask)),
            'colliding_surface_point_fraction_in_scope':float(np.count_nonzero(collision_mask&scope_mask)/np.count_nonzero(collision_mask)),
            'colliding_point_frame_fraction_in_scope':float(bad_count[scope_mask].sum()/bad_count.sum())}
    else:raise ValueError('Missing exact latest repair scope: '+str(scope_file))
    dump(folder/'collision_distribution.json',distribution)
    np.savez_compressed(folder/'collision_point_counts.npz',colliding_frame_count=bad_count,depth_resolved_frame_count=preferred_count)
    settings=AssignmentSettings()
    settings_sha=hashlib.sha256(json.dumps(asdict(settings),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert settings_sha==json.loads(final_report_path.read_text())['settings_sha256']
    predictions={};summaries={}
    for policy in ('original','abstain','depth'):
        emit('surface_aggregation_and_frozen_postprocessing',scene=scene,policy=policy)
        assert np.all(counts[policy]>=0)
        kept=counts[policy]>0
        policy_pair={'surface_point_index':(keys[kept]//base).astype(np.int32),
                     'instance_id':(keys[kept]%base).astype(np.int32),'frame_votes':counts[policy][kept].astype(np.int32)}
        policy_evidence=reduce_surface_votes(keys[kept],counts[policy][kept],len(xyz),base,
            report['min_confirmed_votes'],report['min_confirmed_ratio'])
        native_labels=np.where(policy_evidence['state']==CONFIRMED,policy_evidence['top1_instance_id'],-1).astype(np.int32)
        if policy=='original':
            for field,array in policy_evidence.items():np.testing.assert_array_equal(array,evidence[field])
            np.testing.assert_array_equal(native_labels,original['instance_id'])
        else:
            assert not np.any((native_labels!=original['instance_id'])&~collision_mask)
        def progress(stage, values):
            details=values if isinstance(values,dict) else {'value':float(values)}
            emit('postprocessing',scene=scene,policy=policy,substage=stage,**details)
        variants,_,stats=assign_surface(xyz,normals,rgb,policy_evidence,policy_pair,native_labels,settings,progress=progress)
        final_labels=variants['holes_geodesic']
        if policy=='original':np.testing.assert_array_equal(final_labels,final['instance_id'])
        pdir=folder/policy;pdir.mkdir(exist_ok=True)
        if policy!='original':
            np.savez_compressed(pdir/'surface_evidence.npz',xyz_m=xyz,rgb=rgb,**policy_evidence)
            np.savez_compressed(pdir/'surface_instance_frame_votes.npz',**policy_pair)
        for stage,labels,baseline in (('native',native_labels,original['instance_id']),('final',final_labels,final['instance_id'])):
            stage_dir=pdir/stage;stage_dir.mkdir(exist_ok=True)
            baseline_ids=np.unique(baseline[baseline>0])
            published_ids=np.unique(labels[labels>0])
            inventory=np.union1d(baseline_ids,published_ids).astype(np.int32)
            if policy=='original':path=native/'instance_surface.npz' if stage=='native' else final_path
            else:
                path=stage_dir/'instance_surface.npz'
                np.savez_compressed(path,xyz_m=xyz,rgb=rgb,instance_id=labels,native_instance_ids=inventory)
            predictions[f'{policy}_{stage}']={'path':str(path),'sha256':sha(path),'inventory':inventory.astype(int).tolist()}
        changed_native=native_labels!=original['instance_id'];changed_final=final_labels!=final['instance_id']
        summaries[policy]={'state_counts':{name:int(np.count_nonzero(policy_evidence['state']==i)) for i,name in enumerate(STATE_NAMES)},
            'native_label_changed_points':int(changed_native.sum()),'final_label_changed_points':int(changed_final.sum()),
            'final_changed_outside_collision_points':int(np.count_nonzero(changed_final&~collision_mask)),
            'final_assigned_points':int(np.count_nonzero(final_labels>0)),
            'native_assigned_points':int(np.count_nonzero(native_labels>0)),
            'total_effective_votes':int(counts[policy].sum()),'postprocessing_stats':stats}
        dump(pdir/'mapping_summary.json',summaries[policy])
        del variants,policy_evidence,policy_pair
        gc.collect()
    assert all(sha(p)==digest for p,digest in hashes.items())
    result={'status':'PASS','scene':scene,'GT_read':False,'baseline_modified':False,
        'original_native_and_final_controls_bit_identical':True,'strict_frame_point_uniqueness_verified_all_400_frames':True,
        'original_raw_observation_support_preserved':True,'geometry_and_rgb_shared_exactly':True,
        'frame_count':400,'settings_sha256':settings_sha,'source_sha256':hashes,
        'task_code_sha256':sha(__file__),'one_vote_code_sha256':sha(TASK/'one_vote.py'),
        'depth_tolerance_m':float(report['max_surface_distance_m']),
        'depth_policy':'only colliding points; frozen surface-to-frame projection with valid depth; center raw local mask must exist in saved candidate support, otherwise abstain',
        'inventory_policy':'retain all baseline exported stage IDs, append any newly published existing association IDs; never drop a vanished baseline ID',
        'predictions':predictions,'mapping_summaries':summaries,'seconds':round(time.monotonic()-started,2)}
    dump(complete,result)
    emit('scene_mapping_complete',scene=scene,seconds=result['seconds'])
    return result


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    frozen=OUT/'predictions_freeze.json'
    assert not frozen.exists(),'Predictions already frozen; preserve the existing run'
    codefiles=[TASK/'one_vote.py',TASK/'run_ablation.py']+[SNAPSHOT/'revisable_instance_map/src/revisable_instance_map'/f for f in (
        'surface_evidence.py','surface_projective_evidence.py','offline_surface_assignment.py','frame_io.py')]
    code_before={str(p):sha(p) for p in codefiles}
    dump(OUT/'ablation_policy.json',{'status':'FIXED_BEFORE_NEW_PREDICTIONS_AND_GT_READ','scene_ids':['room0','room2'],
        'policies':['original','abstain','depth'],'raw_frames':list(range(0,2000,5)),
        'only_same_frame_multi_ID_points_changed':True,'depth_tolerance_m':.015,
        'postprocessing_parameters':'exact frozen AssignmentSettings; original final control required',
        'no_new_association_or_geometry':True,'GT_read_for_mapping':False,'code_sha256':code_before})
    scenes=[scene_mapping(scene) for scene in ('room0','room2')]
    assert all(sha(p)==digest for p,digest in code_before.items())
    dump(frozen,{'status':'PREDICTIONS_FROZEN_BEFORE_GT','GT_read_for_mapping':False,'baseline_modified':False,
        'code_sha256':code_before,'scenes':{r['scene']:{'mapping_complete_sha256':sha(OUT/'ablation'/r['scene']/'mapping_complete.json'),
                                                 'predictions':r['predictions']} for r in scenes}})
    emit('all_predictions_frozen',scenes=['room0','room2'],GT_read=False)
