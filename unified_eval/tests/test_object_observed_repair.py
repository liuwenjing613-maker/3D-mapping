"""Acceptance counterexamples for the explicit observed-object repair profile."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import (attach_reference_support, build_fixed_surface_correspondence,
    load_fixed_surface_correspondence, map_fixed_surface_labels, native_reference_region_support,
    save_fixed_surface_correspondence)
from unified_eval.gt_scope import build_gt_scope, apply_observed_scope, coincident_label_ambiguity
from unified_eval.io import save_gt, load_gt, save_prediction, load_prediction
from unified_eval.metrics import build_overlap
from unified_eval.observable_surface import ObservationFrame, accumulate_observed_support, observed_reference_vertices_from_frame
from unified_eval.repair_profile import adapt_surface, load_surface_export, paired_revision_metrics
from unified_eval.schema import CanonicalGT, EvaluationError, Protocol

CONFIG = Path(__file__).parents[1] / "configs/replica_ca_v3.object_observed_repair.development.json"


def protocol():
    return replace(Protocol.from_dict(json.loads(CONFIG.read_text())),
                   significant_min_intersection_vertices=1, significant_min_gt_fraction=.05)


def qualified(raw_ids, semantic=None, counts=None, review=()):
    raw = np.asarray(raw_ids, dtype=np.int64)
    semantic = np.ones(len(raw), dtype=np.int32) if semantic is None else np.asarray(semantic, dtype=np.int32)
    xyz = np.column_stack((np.arange(len(raw)) * .1, np.zeros(len(raw)), np.ones(len(raw))))
    source = CanonicalGT("toy", xyz, raw.copy(), semantic, np.ones(len(raw), dtype=bool),
        np.zeros(len(raw), dtype=bool), raw_instance_id=raw.copy(), raw_semantic_id=semantic.copy())
    scope = build_gt_scope(source, ["wall-plug", "wall"], ["wall-plug"], review_required_ids=review)
    gt, scope = apply_observed_scope(source, scope, np.ones(len(raw), dtype=np.uint32) if counts is None else np.asarray(counts, dtype=np.uint32),
        observation_metadata={"status": "TEST_ONLY"})
    return gt, scope, source


def mapped(gt, labels, xyz=None, native_ids=None, prediction_types=None, correspondence=None):
    xyz = gt.xyz_ref.copy() if xyz is None else xyz
    labels = np.asarray(labels, dtype=np.int64)
    correspondence = correspondence or build_fixed_surface_correspondence(xyz, gt.xyz_ref, .01)
    result = map_fixed_surface_labels(xyz, labels, gt.xyz_ref, correspondence,
        scene_id="toy", method_name="test", method_commit="test",
        native_instance_ids=native_ids, diagnostic_distance_m=.02,
        metadata={"profile_config_sha256": "test_profile"})
    support = native_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, .01)
    attach_reference_support(result, labels, support, gt, correspondence)
    for prediction in (result.prediction, result.diagnostic_prediction):
        for instance in prediction.instances:
            instance.metadata["prediction_type"] = (prediction_types or {}).get(int(instance.instance_uid), "unknown")
    return result, correspondence


def score(gt, result):
    return evaluate_scenes([(gt, result.prediction)], protocol(),
        diagnostic_predictions=[result.diagnostic_prediction])[0]


def test_small_objects_are_qualified_by_identity_and_observation_not_100_vertices():
    gt, scope, _ = qualified([1] * 54 + [2] * 99)
    assert [x["evaluable"] for x in scope["objects"]] == [True, True]
    result, _ = mapped(gt, [11] * 54 + [12] * 99)
    summary = score(gt, result)
    assert summary["CA_PRF1_0_5"]["TP"] == 2
    assert summary["CA_PQ"]["PQ"] == 1


def test_unknown_and_non_target_raw_identities_remain_distinct():
    gt, scope, _ = qualified([1, 2, 3], semantic=[1, 2, 0])
    assert [x["object_status"] for x in scope["objects"]] == ["TARGET", "NON_TARGET", "UNKNOWN"]
    assert gt.evaluation_region.tolist() == [1, 2, 0]
    assert gt.raw_instance_id.tolist() == [1, 2, 3]
    result, _ = mapped(gt, [11, -1, 13])
    summary = score(gt, result)
    assert summary["CA_PRF1_0_5"]["FN"] == 0
    assert summary["unknown_or_unobserved_prediction_count"] == 1


def test_tiny_gt_is_pending_review_instead_of_automatically_invalid():
    _, scope, _ = qualified([76] * 4, review=(76,))
    row = scope["objects"][0]
    assert row["object_status"] == "TARGET" and not row["evaluable"]
    assert row["review_required"] and row["reason"] == "PENDING_GEOMETRY_REVIEW"


def test_unobserved_target_is_not_a_normal_false_negative():
    gt, scope, _ = qualified([1, 2], counts=[1, 0])
    assert scope["objects"][1]["reason"] == "NO_TRUSTED_INPUT_OBSERVATION"
    result, _ = mapped(gt, [11, -1])
    assert score(gt, result)["CA_PRF1_0_5"]["FN"] == 0


def test_nearest_unassigned_geometry_blocks_label_hopping():
    gt, _, _ = qualified([1])
    xyz = np.vstack((gt.xyz_ref, gt.xyz_ref + [0, .003, 0]))
    result, cache = mapped(gt, [-1, 11], xyz=xyz)
    assert cache.nearest_native_index.tolist() == [0]
    assert result.statistics["unassigned_with_geometry_ref_vertices"] == 1
    assert len(result.prediction.instances[0].vertex_indices) == 0
    assert score(gt, result)["CA_PRF1_0_5"]["FP"] == 1


def test_label_switch_reuses_exact_indices_and_cache(tmp_path):
    gt, _, _ = qualified([1, 1, 2, 2])
    before, cache = mapped(gt, [11, -1, 12, 12])
    path = tmp_path / "correspondence.npz"
    save_fixed_surface_correspondence(path, cache)
    loaded = load_fixed_surface_correspondence(path, gt.xyz_ref, gt.xyz_ref)
    after, _ = mapped(gt, [11, 11, 12, 12], correspondence=loaded)
    assert before.prediction.metadata["correspondence_sha256"] == after.prediction.metadata["correspondence_sha256"]
    assert np.array_equal(cache.nearest_native_index, loaded.nearest_native_index)
    assert paired_revision_metrics(gt, before.prediction, after.prediction, protocol())["correct_owner_vertex_gain"] == 1


def test_geometry_change_or_point_deletion_rejects_pairing():
    gt, _, _ = qualified([1, 1])
    before, cache = mapped(gt, [11, 11])
    xyz = gt.xyz_ref + [0, .001, 0]
    with pytest.raises(EvaluationError, match="hash mismatch"):
        map_fixed_surface_labels(xyz, np.array([11,11]), gt.xyz_ref, cache,
            scene_id="toy", method_name="test", method_commit="test")
    after, _ = mapped(gt, [11, 11], xyz=xyz)
    with pytest.raises(EvaluationError, match="physical surface"):
        paired_revision_metrics(gt, before.prediction, after.prediction, protocol())
    with pytest.raises(EvaluationError, match="hash mismatch"):
        cache.validate(gt.xyz_ref[:1], gt.xyz_ref)


def test_zero_and_negative_states_are_never_native_instances():
    gt, _, _ = qualified([1, 1, 1, 1])
    result, _ = mapped(gt, [0, -1, -2, -3])
    assert result.prediction.instances == []
    assert result.statistics["unassigned_with_geometry_ref_vertices"] == 4
    assert score(gt, result)["CA_PRF1_0_5"]["FN"] == 1


def test_empty_projection_duplicate_and_empty_native_export_remain_fp():
    gt, _, _ = qualified([1] * 4)
    xyz = np.vstack((gt.xyz_ref, gt.xyz_ref))
    result, _ = mapped(gt, [11] * 4 + [12] * 4, xyz=xyz, native_ids=np.array([11,12,13]))
    assert sorted(len(x.vertex_indices) for x in result.prediction.instances) == [0,0,4]
    summary = score(gt, result)
    assert summary["CA_PRF1_0_5"]["TP"] == 1
    assert summary["CA_PRF1_0_5"]["FP"] == 2
    assert summary["empty_native_prediction_count"] == 1
    assert summary["structure"]["duplicate_prediction_count"] == 1


def test_20_percent_target_80_percent_unknown_cannot_escape_fp():
    gt, _, _ = qualified([1]*10 + [2]*8, semantic=[1]*10 + [0]*8)
    result, _ = mapped(gt, [11]*2 + [-1]*8 + [11]*8)
    summary = score(gt, result)
    assert summary["CA_PRF1_0_5"]["FP"] == 1
    assert summary["CA_PRF1_0_5"]["ignored_prediction_count"] == 0
    overlap = build_overlap(gt, result.prediction, protocol())
    assert overlap.pred_size.tolist() == [2]
    assert overlap.iou[0,0] == .2
    assert overlap.prediction_ignore_fraction[0] == .8


def test_known_background_leak_reduces_iou_and_is_reported_separately():
    gt, _, _ = qualified([1]*4 + [2]*4, semantic=[1]*4 + [2]*4)
    result, _ = mapped(gt, [11]*8)
    assert build_overlap(gt, result.prediction, protocol()).iou[0,0] == .5
    assert score(gt, result)["CA_PRF1_0_5"]["FP"] == 1
    background, _ = mapped(gt, [-1]*4 + [12]*4)
    assert score(gt, background)["background_only_prediction_count"] == 1
    declared_object, _ = mapped(gt, [-1]*4 + [12]*4, prediction_types={12:"object"})
    assert score(gt, declared_object)["CA_PRF1_0_5"]["FP"] == 1


def test_background_projection_does_not_become_fp_just_by_being_near_a_socket():
    gt, _, _ = qualified([1, 2], semantic=[1, 2])
    # A confirmed wall surface is adjacent to the socket, within the geometry gate.
    gt.xyz_ref[1] = gt.xyz_ref[0] + [0, .003, 0]
    from unified_eval.io import sha256_array
    gt.metadata['reference_xyz_sha256'] = sha256_array(gt.xyz_ref)
    result, _ = mapped(gt, [11, 12])
    assert result.prediction.instances[1].metadata['native_target_support_count'] == 1
    summary = score(gt, result)
    assert summary['background_only_prediction_count'] == 1
    assert summary['CA_PRF1_0_5']['FP'] == 0


def test_outside_reference_is_unverifiable_and_not_a_correct_prediction():
    gt, _, _ = qualified([1])
    result, _ = mapped(gt, [11], xyz=np.array([[99.,99.,99.]]))
    summary = score(gt, result)
    assert summary["unverifiable_prediction_count"] == 1
    assert summary["CA_PRF1_0_5"]["TP"] == 0
    assert summary["owner_surface"]["No_geometry_Coverage"] == 1


def test_deleting_wrong_labels_cannot_increase_correct_owner_surface():
    gt, _, _ = qualified([1]*4 + [2]*4)
    before, cache = mapped(gt, [11]*8)
    after, _ = mapped(gt, [11]*4 + [-1]*4, correspondence=cache)
    pair = paired_revision_metrics(gt, before.prediction, after.prediction, protocol())
    assert pair["correct_owner_vertex_gain"] == 0
    assert score(gt, after)["owner_surface"]["Unassigned_Coverage"] == .5


def test_artificial_merge_and_split_degrade_quality_and_structure():
    gt, _, _ = qualified([1]*4 + [2]*4)
    perfect, cache = mapped(gt, [11]*4 + [12]*4)
    merged, _ = mapped(gt, [11]*8, correspondence=cache)
    assert score(gt, perfect)["CA_PQ"]["PQ"] == 1
    assert score(gt, merged)["CA_PQ"]["PQ"] == 0
    assert score(gt, merged)["structure"]["merge_prediction_count"] == 1
    one, _, _ = qualified([1]*8)
    split, _ = mapped(one, [11]*4 + [12]*4)
    assert score(one, split)["structure"]["split_gt_count"] == 1


def test_instance_id_renaming_preserves_segmentation_metrics():
    gt, _, _ = qualified([1]*4 + [2]*4)
    before, _ = mapped(gt, [11]*4 + [12]*4)
    renamed, _ = mapped(gt, [801]*4 + [3]*4)
    a,b = score(gt, before), score(gt, renamed)
    for key in ("CA_AP_uniform","CA_AP50_uniform","CA_PQ","CA_PRF1_0_5","CA_mCov","owner_surface","structure"):
        assert a[key] == b[key]


def test_discrete_perfect_gt_oracle_has_ideal_scores():
    gt, _, _ = qualified([1]*4 + [2]*4)
    result, _ = mapped(gt, [1]*4 + [2]*4)
    summary = score(gt, result)
    assert summary["CA_AP50_uniform"] == summary["CA_PQ"]["PQ"] == summary["CA_PRF1_0_5"]["F1"] == summary["CA_mCov"] == 1
    assert summary["owner_surface"]["Correct_owner_Coverage"] == 1


def test_camera_depth_and_gt_id_prevent_wall_support_becoming_socket_support():
    xyz = np.array([[0.,0.,1.],[.001,0.,1.]])
    k=np.array([[10.,0.,0.],[0.,10.,0.],[0.,0.,1.]])
    frame=ObservationFrame(0,np.array([[1.]]),k,np.eye(4),np.array([[2]],dtype=np.int64))
    observed=observed_reference_vertices_from_frame(xyz,np.array([1,2]),frame,.02)
    assert observed.tolist() == [False,True]
    invalid=replace(frame,depth_m=np.array([[0.]]))
    assert not observed_reference_vertices_from_frame(xyz,np.array([1,2]),invalid,.02).any()
    occluded=replace(frame,depth_m=np.array([[.9]]))
    assert not observed_reference_vertices_from_frame(xyz,np.array([1,2]),occluded,.02).any()


def test_observation_counts_use_unique_frames_and_do_not_impose_pixel_size_filter():
    xyz=np.array([[0.,0.,1.]])
    frame=ObservationFrame(0,np.array([[1.]]),np.eye(3),np.eye(4),np.array([[1]],dtype=np.int64))
    counts,meta=accumulate_observed_support(xyz,np.array([1]),[frame],.02)
    assert counts.tolist() == [1] and meta["pixel_size_filter"] is None
    assert not meta["mesh_visibility_verified"]
    with pytest.raises(EvaluationError,match="unique"):
        accumulate_observed_support(xyz,np.array([1]),[frame,frame],.02)


def test_exact_gt_label_geometry_ambiguity_is_explicitly_ignored():
    xyz=np.array([[0.,0.,1.],[0.,0.,1.],[.1,0.,1.]])
    assert coincident_label_ambiguity(xyz,np.array([1,2,1])).tolist() == [True,True,False]


def test_scope_cache_roundtrip_and_profile_mixup_fail_closed(tmp_path):
    gt, _, _ = qualified([1,1,2],semantic=[1,1,0])
    result,_=mapped(gt,[11,11,12])
    path=tmp_path/'gt.npz';save_gt(path,gt); loaded=load_gt(path)
    assert np.array_equal(loaded.raw_instance_id,gt.raw_instance_id)
    assert np.array_equal(loaded.evaluation_region,gt.evaluation_region)
    predpath=tmp_path/'pred.npz';save_prediction(predpath,result.prediction)
    assert load_prediction(predpath).metadata == result.prediction.metadata
    altered=replace(result.prediction,metadata={**result.prediction.metadata,'gt_scope_sha256':'changed'})
    with pytest.raises(EvaluationError,match="mismatch"):
        build_overlap(gt,altered,protocol())
    with pytest.raises(EvaluationError,match="historical"):
        build_overlap(gt,result.prediction,replace(protocol(),evaluation_profile='legacy'))


def test_qualification_source_role_is_replaced_before_scoring():
    _, scope, source = qualified([1, 1])
    source.metadata['role'] = 'raw_gt_qualification_source_only'
    gt, _ = apply_observed_scope(source, build_gt_scope(source, ['wall-plug', 'wall'], ['wall-plug']),
        np.ones(2, dtype=np.uint32), observation_metadata={'status': 'TEST_ONLY'})
    result, _ = mapped(gt, [11, 11])
    assert score(gt, result)['CA_PQ']['PQ'] == 1


def test_modified_gt_arrays_and_ad_hoc_observation_masks_are_rejected():
    gt, _, _ = qualified([1, 1])
    result, _ = mapped(gt, [11, 11])
    with pytest.raises(EvaluationError, match='separate GT scope'):
        build_overlap(gt, result.prediction, protocol(), observed_mask=np.array([True, False]))
    for name in ('observation_count', 'evaluation_region', 'instance_id', 'xyz_ref', 'raw_instance_id'):
        changed = getattr(gt, name).copy()
        changed.flat[0] += 1
        if name == 'evaluation_region':
            # Preserve consistent masks so the stored checksum is the failing invariant.
            altered = replace(gt, evaluation_region=changed)
        else:
            altered = replace(gt, **{name: changed})
        with pytest.raises(EvaluationError, match='hash mismatch'):
            build_overlap(altered, result.prediction, protocol())


def test_repair_profile_cannot_smuggle_100_vertex_gate_or_legacy_mapping():
    raw=json.loads(CONFIG.read_text())
    raw['instance_filter']['min_valid_instance_vertices']=100
    with pytest.raises(EvaluationError,match="scope"):
        Protocol.from_dict(raw)


def test_ovimap_normalization_preserves_id_zero_full_geometry_and_empty_inventory(tmp_path):
    gt, _, _ = qualified([1, 1])
    xyz = np.vstack((gt.xyz_ref, gt.xyz_ref[0]))
    path = tmp_path / 'ovi.npz'
    np.savez_compressed(path, xyz=xyz, instance=np.array([0, -1, 1]),
        native_instance_ids=np.array([0, 12, 13]), classes=np.array([95, -1, -1]))
    loaded = load_surface_export(path)
    assert np.array_equal(loaded[0], xyz) and loaded[1].tolist() == [1, -1, 2]
    assert loaded[2].tolist() == [1, 2, 3] and loaded[3].tolist() == [0, 12, 13]
    gt_path = tmp_path / 'gt.npz'; save_gt(gt_path, gt)
    output = tmp_path / 'adapter'
    report = adapt_surface(path, gt_path, CONFIG, tmp_path / 'fixed.npz', output,
        method_name='OVI-MAP', method_commit='test')
    pred = load_prediction(output / 'canonical_prediction.npz')
    assert [x.instance_uid for x in pred.instances] == ['ovi-id:0', 'ovi-id:12', 'ovi-id:13']
    assert pred.instances[0].metadata['native_instance_id'] == 0
    assert pred.instances[0].metadata['normalized_positive_instance_id'] == 1
    assert pred.instances[0].metadata['prediction_type'] == 'unknown'
    assert report['summary']['CA_PRF1_0_5']['FP'] == 3
    assert report['manifest']['adapter_statistics']['geometry_point_count'] == 3


def test_ovimap_and_tsdf_cannot_replace_their_exported_inventory(tmp_path):
    xyz = np.array([[0., 0., 1.]])
    path = tmp_path / 'ovi.npz'
    np.savez_compressed(path, xyz=xyz, instance=np.array([1]), native_instance_ids=np.array([0]))
    with pytest.raises(EvaluationError, match='outside'):
        load_surface_export(path)
    path = tmp_path / 'tsdf.npz'
    np.savez_compressed(path, xyz_m=xyz, instance_id=np.array([11]), native_instance_ids=np.array([11]))
    with pytest.raises(EvaluationError, match='disagrees'):
        load_surface_export(path, np.array([11, 12]))


def test_development_profile_cannot_be_frozen_by_flipping_one_flag():
    raw = json.loads(CONFIG.read_text())
    raw['frozen'] = True
    with pytest.raises(EvaluationError, match='development'):
        Protocol.from_dict(raw)


def test_method_input_frame_mismatch_is_rejected_before_scoring(tmp_path):
    gt, _, _ = qualified([1, 1])
    path = tmp_path / 'surface.npz'
    np.savez_compressed(path, xyz_m=gt.xyz_ref, instance_id=np.array([11, 11]))
    gt_path = tmp_path / 'gt.npz'; save_gt(gt_path, gt)
    with pytest.raises(EvaluationError, match='input frame list'):
        adapt_surface(path, gt_path, CONFIG, tmp_path / 'fixed.npz', tmp_path / 'output',
            method_name='test', method_commit='test', source_provenance={'frame_list_sha256': 'changed'})


def test_old_adapted_distances_cannot_be_relabelled_as_a_different_profile():
    gt, _, _ = qualified([1, 1])
    result, _ = mapped(gt, [11, 11])
    with pytest.raises(EvaluationError, match='main geometry distance'):
        build_overlap(gt, result.prediction, replace(protocol(), geometry_mapping_max_distance_m=.02))
    with pytest.raises(EvaluationError, match='diagnostic geometry distance'):
        build_overlap(gt, result.diagnostic_prediction, replace(protocol(), diagnostic_max_distance_m=.03))
    raw=json.loads(CONFIG.read_text());raw['evaluation_profile']='legacy'
    raw['ignore_policy']['unmatched_prediction_void_fraction_gt']=.5
    with pytest.raises(EvaluationError,match="competitive"):
        Protocol.from_dict(raw)
