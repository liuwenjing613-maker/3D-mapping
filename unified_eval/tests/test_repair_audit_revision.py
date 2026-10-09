"""Freeze-review counterexamples: point order, adjacent boundaries and source locks."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from unified_eval.audit import background_only_fp_sensitivity, native_nearest_reference_region_support
from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import (attach_reference_support, build_fixed_surface_correspondence,
    map_fixed_surface_labels, native_reference_region_support, save_fixed_surface_correspondence,
    load_fixed_surface_correspondence)
from unified_eval.io import repair_evaluator_code_hashes, sha256_array, sha256_file
from unified_eval.metrics import build_overlap, significant_structure_diagnostics
from unified_eval.schema import EvaluationError, Protocol
from unified_eval.tests.test_object_observed_repair import qualified, mapped, protocol as baseline_protocol

CONFIG = Path(__file__).parents[1] / 'configs/replica_ca_v3.object_observed_repair.audit_r2.json'


def protocol():
    return Protocol.from_dict(json.loads(CONFIG.read_text()))


def mapped2(gt, labels, xyz=None, inventory=None, cache=None):
    p = protocol()
    xyz = gt.xyz_ref.copy() if xyz is None else xyz
    labels = np.asarray(labels, dtype=np.int64)
    cache = cache or build_fixed_surface_correspondence(xyz, gt.xyz_ref, p.geometry_mapping_max_distance_m)
    result = map_fixed_surface_labels(xyz, labels, gt.xyz_ref, cache, scene_id=gt.scene_id,
        method_name='audit-test', method_commit='test', native_instance_ids=inventory,
        diagnostic_distance_m=p.diagnostic_max_distance_m,
        duplicate_coordinate_policy=p.duplicate_coordinate_policy,
        metadata={'profile_revision': 2, 'profile_semantics_sha256': p.profile_semantics_sha256,
                  'profile_config_sha256': sha256_file(CONFIG), 'evaluator_code_sha256': repair_evaluator_code_hashes()})
    support = native_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, p.geometry_mapping_max_distance_m)
    attach_reference_support(result, labels, support, gt, cache)
    return result, cache


def score(gt, result):
    return evaluate_scenes([(gt, result.prediction)], protocol(),
                           diagnostic_predictions=[result.diagnostic_prediction])[0]


def test_ten_native_point_permutations_with_conflicting_exact_coordinates_preserve_scores():
    gt, _, _ = qualified([1] * 20 + [2] * 20)
    xyz = np.concatenate([gt.xyz_ref, gt.xyz_ref[[0, 20]]])
    labels = np.array([11] * 20 + [12] * 20 + [12, 11])
    original, _ = mapped2(gt, labels, xyz)
    expected = score(gt, original)
    assert original.statistics['conflicting_exact_coordinate_group_count'] == 2
    assert expected['owner_surface']['duplicate_coordinate_conflict_target_vertices'] == 2
    rng = np.random.default_rng(20261009)
    for _ in range(10):
        order = rng.permutation(len(xyz))
        shuffled, _ = mapped2(gt, labels[order], xyz[order])
        actual = score(gt, shuffled)
        for key in ('CA_AP_uniform', 'CA_AP50_uniform', 'CA_PQ', 'CA_PRF1_0_5', 'CA_mCov', 'owner_surface', 'structure'):
            assert actual[key] == expected[key]
        assert shuffled.statistics['conflicting_exact_coordinate_group_count'] == 2


def test_duplicate_conflict_is_unassigned_with_geometry_and_all_hypotheses_remain_fp():
    gt, _, _ = qualified([1] * 20)
    xyz = np.concatenate([gt.xyz_ref, gt.xyz_ref])
    result, _ = mapped2(gt, [11] * 20 + [12] * 20, xyz, inventory=np.array([11, 12, 13]))
    summary = score(gt, result)
    assert summary['CA_PRF1_0_5']['TP'] == 0 and summary['CA_PRF1_0_5']['FP'] == 3
    assert summary['owner_surface']['Unassigned_Coverage'] == 1
    assert summary['owner_surface']['No_geometry_Coverage'] == 0
    assert summary['owner_surface']['duplicate_coordinate_conflict_target_vertices'] == 20


def test_unanimous_duplicates_and_all_nonpositive_states_are_order_independent():
    gt, _, _ = qualified([1, 1])
    xyz = np.repeat(gt.xyz_ref, 2, axis=0)
    result, cache = mapped2(gt, [11, 11, -2, 0], xyz)
    assert result.prediction.instances[0].vertex_indices.tolist() == [0]
    assert result.statistics['conflicting_exact_coordinate_group_count'] == 0
    conflict, _ = mapped2(gt, [11, 0, -2, -1], xyz, cache=cache)
    assert len(conflict.prediction.instances[0].vertex_indices) == 0
    assert conflict.statistics['conflicting_exact_coordinate_group_count'] == 1


def test_conflicting_duplicate_labels_can_be_resolved_on_the_same_fixed_geometry_cache():
    gt, _, _ = qualified([1] * 20)
    xyz = np.repeat(gt.xyz_ref, 2, axis=0)
    before, cache = mapped2(gt, np.tile([11, 12], 20), xyz)
    after, same = mapped2(gt, [11] * len(xyz), xyz, cache=cache)
    assert same is cache
    assert before.prediction.metadata['correspondence_sha256'] == after.prediction.metadata['correspondence_sha256']
    assert score(gt, after)['owner_surface']['Correct_owner_Coverage'] == 1


def test_duplicate_conflict_policy_is_invariant_to_instance_id_renaming():
    gt, _, _ = qualified([1] * 20 + [2] * 20)
    xyz = np.concatenate([gt.xyz_ref, gt.xyz_ref[[0, 20]]])
    a, _ = mapped2(gt, [11] * 20 + [12] * 20 + [12, 11], xyz)
    b, _ = mapped2(gt, [801] * 20 + [3] * 20 + [3, 801], xyz)
    for key in ('CA_AP50_uniform', 'CA_PQ', 'CA_PRF1_0_5', 'CA_mCov', 'owner_surface', 'structure'):
        assert score(gt, a)[key] == score(gt, b)[key]


def test_correctly_separated_objects_15mm_apart_do_not_become_primary_merge_or_split():
    gt, _, _ = qualified([1] * 20 + [2] * 20)
    x = np.arange(20) * .0002
    gt.xyz_ref = np.concatenate([np.column_stack([x, np.zeros(20), np.ones(20)]),
                                 np.column_stack([x, np.full(20, .015), np.ones(20)])])
    gt.metadata['reference_xyz_sha256'] = sha256_array(gt.xyz_ref)
    result, _ = mapped2(gt, [11] * 20 + [12] * 20)
    summary = score(gt, result)
    assert summary['CA_PQ']['PQ'] == 1
    assert summary['structure']['merge_prediction_count'] == summary['structure']['split_gt_count'] == 0
    assert summary['auxiliary_geometry_structure']['merge_prediction_count'] == 2
    assert summary['auxiliary_geometry_structure']['split_gt_count'] == 2


def test_true_merge_and_split_are_detected_by_main_partition_intersections():
    gt, _, _ = qualified([1] * 20 + [2] * 20)
    merged, _ = mapped2(gt, [11] * 40)
    assert score(gt, merged)['structure']['merge_prediction_count'] == 1
    one, _, _ = qualified([1] * 40)
    split, _ = mapped2(one, [11] * 20 + [12] * 20)
    assert score(one, split)['structure']['split_gt_count'] == 1


def test_small_object_structure_threshold_sensitivity_is_explicit_not_a_hidden_rule_change():
    gt, _, _ = qualified([1] * 54)
    result, _ = mapped2(gt, [11] * 45 + [12] * 9)
    overlap = build_overlap(gt, result.prediction, protocol())
    assert significant_structure_diagnostics(overlap, protocol())['split_gt_count'] == 0
    assert significant_structure_diagnostics(overlap, replace(protocol(), significant_min_intersection_vertices=1))['split_gt_count'] == 1


def test_background_only_sensitivity_counts_fp_without_mutating_predictions_or_primary_policy():
    gt, _, _ = qualified([1] * 20 + [2] * 20, semantic=[1] * 20 + [2] * 20)
    result, _ = mapped2(gt, [11] * 20 + [12] * 20)
    report = background_only_fp_sensitivity([(gt, result.prediction)], protocol())
    assert report['background_only_candidate_count'] == 1
    assert report['ignore_background_only']['FP'] == 0
    assert report['count_all_background_only_as_FP']['FP'] == 1
    assert report['count_all_background_only_as_FP']['CA_PQ'] < report['ignore_background_only']['CA_PQ']
    assert score(gt, result)['CA_PRF1_0_5']['FP'] == 0


def test_empty_wall_duplicate_near_socket_exposes_target_priority_boundary_sensitivity():
    gt, _, _ = qualified([1, 2], semantic=[1, 2])
    gt.xyz_ref[1] = gt.xyz_ref[0] + [0, .003, 0]
    gt.metadata['reference_xyz_sha256'] = sha256_array(gt.xyz_ref)
    xyz = np.concatenate([gt.xyz_ref, gt.xyz_ref[1:2] + [0, .0001, 0]])
    labels = np.array([11, 12, 13])
    result, _ = mapped2(gt, labels, xyz)
    assert len(result.prediction.instances[2].vertex_indices) == 0
    priority = native_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, .01)
    nearest = native_nearest_reference_region_support(xyz, gt.xyz_ref, gt.evaluation_region, .01)
    assert priority[2] == 1 and nearest[2] == 2
    assert score(gt, result)['CA_PRF1_0_5']['FP'] == 1


def test_old_prediction_cannot_be_scored_under_revision_two_and_reverse():
    gt, _, _ = qualified([1] * 20)
    old, _ = mapped(gt, [11] * 20)
    with pytest.raises(EvaluationError, match='revision/configuration'):
        build_overlap(gt, old.prediction, protocol())
    new, _ = mapped2(gt, [11] * 20)
    with pytest.raises(EvaluationError, match='locked revision-1'):
        build_overlap(gt, new.prediction, baseline_protocol())


def test_new_configuration_or_changed_evaluator_hash_rejects_old_adapted_output():
    gt, _, _ = qualified([1] * 20)
    result, _ = mapped2(gt, [11] * 20)
    config = json.loads(CONFIG.read_text())
    config['significant_overlap']['min_intersection_vertices'] = 3
    with pytest.raises(EvaluationError, match='revision/configuration'):
        build_overlap(gt, result.prediction, Protocol.from_dict(config))
    result.prediction.metadata['evaluator_code_sha256'] = {}
    with pytest.raises(EvaluationError, match='evaluator code hashes'):
        build_overlap(gt, result.prediction, protocol())


def test_old_cache_requires_regeneration_and_new_group_cache_survives_roundtrip(tmp_path):
    gt, _, _ = qualified([1, 1])
    cache = build_fixed_surface_correspondence(gt.xyz_ref, gt.xyz_ref, .01)
    old = replace(cache, native_unique_index=None, unique_first_index=None)
    with pytest.raises(EvaluationError, match='new geometry cache'):
        mapped2(gt, [11, 11], cache=old)
    path = tmp_path / 'cache.npz'
    save_fixed_surface_correspondence(path, cache)
    loaded = load_fixed_surface_correspondence(path, gt.xyz_ref, gt.xyz_ref)
    assert loaded.sha256 == cache.sha256
    assert np.array_equal(loaded.native_unique_index, cache.native_unique_index)


def test_revision_one_semantics_cannot_be_changed_in_place():
    config = json.loads(CONFIG.read_text())
    config['profile_revision'] = 1
    with pytest.raises(EvaluationError, match='semantics are locked'):
        Protocol.from_dict(config)
