"""Immutable ScanNet exports, fixed observed-object scope, audited V3 revision 2."""
from collections import Counter
import ast
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from plyfile import PlyData

CODE = Path(__file__).resolve().parent
ROOT = Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104/ground_truth')
OLD = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_scannet_20261008')
FIX = Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_scannet_engineering_fix_20261008')
INPUT_CONFIG = Path('/home/chenkejun/CVPR/experiments/p1a1_scannet_20261008/configs')
MC = Path('/data/chenkejun/CVPR/datasets/MaskClustering')
RAW = Path('/data/chenkejun/datasets/ScanNet')
CONFIG = CODE / 'scannet_ca_v3.object_observed_repair.audit_r2.json'
SCENES = ['scene0011_00','scene0608_00']
METHODS = ['MaskClustering','P1-A1_original','P1-A1_engineering_fixed']
sys.dont_write_bytecode = True
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from unified_eval.evaluate import evaluate_scenes
from unified_eval.gt_scope import apply_observed_scope,build_gt_scope,coincident_label_ambiguity,scope_sha256
from unified_eval.io import load_gt,load_prediction,save_gt,sha256_array,sha256_file,repair_evaluator_code_hashes
from unified_eval.mesh_visibility import GTMeshVisibility
from unified_eval.observable_surface import ObservationFrame,accumulate_observed_support,save_observed_support
from unified_eval.repair_profile import adapt_surface,write_json
from unified_eval.schema import CanonicalGT,Protocol
from scripts.run_object_observed_repair_development import direct_gt_oracle

INPUT_HASHES = {}


def track(path):
    path=Path(path)
    digest=sha256_file(path)
    if str(path) in INPUT_HASHES: assert INPUT_HASHES[str(path)]==digest
    INPUT_HASHES[str(path)]=digest
    return digest


def emit(stage,**extra):
    value=dict(stage=stage,**extra)
    write_json(ROOT/'progress.json',value)
    print(json.dumps(value),flush=True)


def classes():
    path=Path('/home/chenkejun/CVPR/repos/MaskClustering/evaluation/constants.py')
    track(path)
    tree=ast.parse(path.read_text())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ['SCANNET_LABELS','SCANNET_IDS']}
    assert len(constants['SCANNET_IDS'])==len(constants['SCANNET_LABELS'])==198
    selected=dict(zip(constants['SCANNET_IDS'],constants['SCANNET_LABELS']))
    mapping=RAW/'scannetv2-labels.combined.tsv'
    track(mapping)
    with mapping.open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    categories={int(r['id']):selected.get(int(r['id']),r['category']) for r in rows}
    semantic=[None]*max(categories)
    for key,value in categories.items(): semantic[key-1]=value
    return selected,{r['raw_category']:int(r['id']) for r in rows},semantic


def make_source(scene,selected,raw_category,semantic):
    root=RAW/'scans'/scene
    mesh_path=root/(scene+'_vh_clean_2.ply')
    agg_path=root/(scene+'.aggregation.json')
    seg_path=root/(scene+'_vh_clean_2.0.010000.segs.json')
    for p in [mesh_path,agg_path,seg_path]: track(p)
    mesh=PlyData.read(mesh_path)
    xyz=np.column_stack([mesh['vertex'][key] for key in ['x','y','z']]).astype(np.float64)
    faces=np.stack(mesh['face']['vertex_indices'])
    groups=json.loads(agg_path.read_text())['segGroups']
    segments=np.asarray(json.loads(seg_path.read_text())['segIndices'])
    assert len(segments)==len(xyz)
    raw_ids=np.full(len(xyz),-1,np.int64)
    semantic_ids=np.zeros(len(xyz),np.int64)
    historical_expected=np.ones(len(xyz),np.int64)
    annotation={}
    for group in groups:
        identity=int(group['objectId'])+1
        assert group['id']==group['objectId'] and identity not in annotation
        selected_vertices=np.isin(segments,group['segments'])
        assert np.all(raw_ids[selected_vertices]<0), 'Overlapping source annotation groups'
        category_id=raw_category[group['label']]
        raw_ids[selected_vertices]=identity
        semantic_ids[selected_vertices]=category_id
        historical_expected[selected_vertices]=(category_id if category_id in selected else 0)*1000+identity+1
        annotation[identity]=dict(original_object_id=group['objectId'],source_group_id=group['id'],
            source_raw_category=group['label'],official_category_id=category_id,
            historical_gt_label=(category_id if category_id in selected else 0)*1000+identity+1)
    historical_path=MC/'scannet/gt'/(scene+'.txt')
    track(historical_path)
    historical=np.loadtxt(historical_path,dtype=np.int64)
    assert np.array_equal(historical_expected,historical), 'Raw scene annotation does not reproduce historical GT labels'
    source=CanonicalGT(scene,xyz,raw_ids.copy(),semantic_ids,np.ones(len(xyz),bool),raw_ids<=0,
        raw_instance_id=raw_ids,raw_semantic_id=semantic_ids.copy(),metadata=dict(dataset='ScanNet200',
            source_reference_sha256=INPUT_HASHES[str(mesh_path)],source_mesh=str(mesh_path),
            source_aggregation=str(agg_path),source_segmentation=str(seg_path),
            source_annotation_encoding='raw_instance_id=official_objectId+1; -1 is unannotated; historical export adds another +1',
            coordinate_transform='identity; no axisAlignment',historical_gt_labels_reproduced=True))
    targets=[name for name in selected.values() if name not in ['wall','floor','ceiling']]
    scope=build_gt_scope(source,semantic,targets)
    for row in scope['objects']:
        identity=row['raw_gt_id']
        if identity>0:
            row['official_annotation']=annotation[identity]
            cid=annotation[identity]['official_category_id']
            if cid not in selected and row['object_status']!='NON_TARGET':
                row.update(object_status='UNKNOWN',reason='OFFICIAL_CATEGORY_OUTSIDE_FIXED_SCANNET200_TARGET_CLASSES')
    scope.update(dataset='ScanNet200',historical_size_filter_removed=True,
        qualification_source='Official aggregation/segmentation/TSV plus existing 198-class ScanNet200 object set; structural ceiling excluded by repair profile',
        source_file_sha256={str(p):INPUT_HASHES[str(p)] for p in [mesh_path,agg_path,seg_path,historical_path]})
    return source,scope,faces


def observation_frames(scene,source,faces,config):
    root=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104/inputs')/scene/'input'
    poses=np.loadtxt(root/'traj.txt').reshape(-1,4,4)
    intr=np.loadtxt(root/'intrinsics.txt').reshape(-1,3,3)
    renderer=GTMeshVisibility(source.xyz_ref,faces,source.raw_instance_id)
    for sequence,identity in enumerate(range(0,1000,5)):
        dp=root/'results'/f'depth{identity:06d}.png'
        depth=cv2.imread(str(dp),-1).astype(np.float32)/1000
        pose=poses[identity];k=intr[identity]
        visible,rendered=renderer.render_reference_pixels(source.xyz_ref,k,pose,depth.shape)
        if sequence%20==0:emit('observation',scene=scene,done=sequence+1,total=200)
        yield ObservationFrame(identity,depth,k,pose,visible,rendered,{str(dp):track(dp)})


def make_gt(scene,selected,raw_category,semantic,config):
    folder=ROOT/scene/'ground_truth';folder.mkdir(parents=True)
    source,scope,faces=make_source(scene,selected,raw_category,semantic)
    write_json(folder/'candidate_scope.json',scope)
    counts,metadata=accumulate_observed_support(source.xyz_ref,source.raw_instance_id,
        observation_frames(scene,source,faces,config),config['observation']['depth_tolerance_m'])
    ambiguity=coincident_label_ambiguity(source.xyz_ref,source.raw_instance_id)
    gt,qualified=apply_observed_scope(source,scope,counts,observation_metadata=metadata,ambiguity_mask=ambiguity)
    gt.metadata.update(profile_revision=config['profile_revision'],
        profile_semantics_sha256=Protocol.from_dict(config).profile_semantics_sha256,
        evaluator_code_sha256=repair_evaluator_code_hashes())
    save_gt(folder/'gt.npz',gt)
    save_observed_support(folder/'gt_observed_support.npz',source.xyz_ref,source.raw_instance_id,counts,metadata,ambiguity)
    write_json(folder/'gt_scope.json',qualified)
    oracle=direct_gt_oracle(gt,Protocol.from_dict(config))
    write_json(folder/'discrete_gt_oracle.json',oracle)
    summary=dict(scene=scene,raw_objects=len([r for r in qualified['objects'] if r['raw_gt_id']>0]),
        evaluable_objects=sum(r['evaluable'] for r in qualified['objects']),
        status_counts=dict(Counter(r['object_status'] for r in qualified['objects'])),
        observed_target_vertices=int(np.count_nonzero(gt.evaluation_region==1)),
        observed_non_target_vertices=int(np.count_nonzero(gt.evaluation_region==2)),
        ambiguity_vertices=int(ambiguity.sum()),gt_scope_sha256=gt.metadata['gt_scope_sha256'],
        fixed_frame_count=metadata['frame_count'],oracle_passed=True)
    write_json(folder/'scope_summary.json',summary)
    emit('scope_ready',**summary)
    return gt,summary



if __name__=='__main__':
    from unified_eval.current_protocol import CURRENT_CONFIG,require_current_protocol
    require_current_protocol(CURRENT_CONFIG)
    ROOT.mkdir(parents=True,exist_ok=True)
    config=json.loads(CURRENT_CONFIG.read_text())
    selected,raw,semantic=classes()
    scope={}
    for scene in ['scene0011_00','scene0050_00']:
        folder=ROOT/scene/'ground_truth'
        if (folder/'scope_summary.json').exists():
            scope[scene]=json.loads((folder/'scope_summary.json').read_text());continue
        gt,scope[scene]=make_gt(scene,selected,raw,semantic,config)
    write_json(ROOT/'scope_lock.json',dict(status='PASS',scope=scope,evaluator_code_sha256=repair_evaluator_code_hashes(),current_config_sha256=sha256_file(CURRENT_CONFIG),predictions_used=False,frame_count=200,source_dataset='ScanNet200'))
