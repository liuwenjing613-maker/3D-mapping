"""Geometry-calibration regressions, including the known 5 cm failure."""

from dataclasses import replace
import os

import numpy as np
import pytest

from unified_eval.calibration import (
    eligible_oracle_ids, evaluate_oracle, oracle_clouds, oracle_identity_pass,
)
from unified_eval.evaluate import evaluate_scenes
from unified_eval.schema import (
    CanonicalGT, CanonicalInstance, CanonicalPrediction, EvaluationError, Protocol,
)
from unified_eval.replica import load_existing_reference


def protocol():
    return Protocol.from_dict({
        "name": "Replica-CA-v2", "dataset": "Replica", "confidence_mode": "uniform",
        "geometry_mapping": {"method": "independent_nearest_neighbor_per_instance",
                             "max_distance_m": 0.01},
        "instance_filter": {"min_valid_instance_vertices": 4,
                            "min_prediction_vertices": 0},
        "ignore_policy": {"unmatched_prediction_void_fraction_gt": 0.5},
        "significant_overlap": {"min_intersection_vertices": 1,
                                "min_gt_fraction": 0.05},
    })


def adjacent_scene():
    # Four vertices per object, with 3 cm between parallel surfaces.
    x = np.arange(4, dtype=float) * 0.002
    xyz = np.vstack((np.column_stack((x, np.zeros(4), np.zeros(4))),
                     np.column_stack((x, np.full(4, 0.03), np.zeros(4)))))
    return CanonicalGT("adjacent", xyz, np.array([1] * 4 + [2] * 4),
        np.ones(8, dtype=np.int32), np.ones(8, dtype=bool), np.zeros(8, dtype=bool))


def test_perfect_oracle_passes_at_1cm_and_known_5cm_failure_is_detected():
    gt = adjacent_scene()
    ids = eligible_oracle_ids(gt, 4)
    clouds = oracle_clouds(gt, ids)
    passed = evaluate_oracle(gt, protocol(), ids, clouds, delta_m=0.01)
    failed = evaluate_oracle(gt, protocol(), ids, clouds, delta_m=0.05)
    assert oracle_identity_pass(passed)
    assert passed["foreign_instance_vertex_claims"] == 0
    assert not oracle_identity_pass(failed)
    assert failed["foreign_instance_vertex_claims"] == 8
    assert failed["TP"] == 0  # Each exact object expands to both objects: IoU = 0.5.


def test_oracle_subset_monotonicity_and_no_hidden_duplicates():
    gt = adjacent_scene()
    gt.xyz_ref = np.vstack((gt.xyz_ref, np.array([[2., 0, 0], [2.002, 0, 0],
                                                  [2.004, 0, 0], [2.006, 0, 0]])))
    gt.instance_id = np.append(gt.instance_id, [3] * 4)
    gt.semantic_id = np.append(gt.semantic_id, [1] * 4)
    gt.valid_vertex_mask = np.append(gt.valid_vertex_mask, [True] * 4)
    gt.ignore_vertex_mask = np.append(gt.ignore_vertex_mask, [False] * 4)
    ids = eligible_oracle_ids(gt, 4)
    clouds = oracle_clouds(gt, ids)
    rows = [evaluate_oracle(gt, protocol(), ids[:k], clouds[:k], delta_m=0.01)
            for k in (1, 2, 3)]
    assert [row["TP"] for row in rows] == [1, 2, 3]
    assert all(a["F1_0_5"] < b["F1_0_5"] and a["AP50"] < b["AP50"]
               for a, b in zip(rows, rows[1:]))
    assert oracle_identity_pass(rows[-1])
    duplicate = evaluate_oracle(gt, protocol(), np.append(ids, ids[0]),
                                clouds + [clouds[0].copy()], delta_m=0.01)
    assert duplicate["TP"] == 3 and duplicate["FP"] == 1
    assert duplicate["AP50"] < 1 and duplicate["F1_0_5"] < 1


def test_perturbed_oracle_exposes_too_small_radius():
    gt = adjacent_scene()
    ids = eligible_oracle_ids(gt, 4)
    clouds = oracle_clouds(gt, ids, sigma_m=0.004, seed=17)
    tiny = evaluate_oracle(gt, protocol(), ids, clouds, delta_m=0.0001)
    tolerant = evaluate_oracle(gt, protocol(), ids, clouds, delta_m=0.01)
    assert not oracle_identity_pass(tiny)
    assert tolerant["TP"] >= tiny["TP"]
    assert tolerant["foreign_instance_vertex_claims"] == 0


@pytest.mark.parametrize("sigma,keep", [(float("nan"), 1), (-0.01, 1), (0, 0), (0, 1.1)])
def test_oracle_perturbation_inputs_are_validated(sigma, keep):
    gt = adjacent_scene()
    with pytest.raises(EvaluationError):
        oracle_clouds(gt, eligible_oracle_ids(gt, 4), sigma_m=sigma,
                      keep_fraction=keep)


@pytest.mark.skipif(not os.environ.get("REPLICA_REFERENCE_ROOT"),
                    reason="requires existing server-side Replica reference")
def test_real_room0_oracle_gate_rejects_historical_5cm():
    gt = load_existing_reference(os.environ["REPLICA_REFERENCE_ROOT"], "room0")
    p = replace(protocol(), min_valid_instance_vertices=100)
    ids = eligible_oracle_ids(gt, 100)
    assert len(ids) == 68
    direct = CanonicalPrediction("room0", gt.vertex_count,
        [CanonicalInstance(str(gt_id), np.flatnonzero(gt.instance_id == gt_id).astype(np.int32))
         for gt_id in ids], "direct GT oracle", "diagnostic", "direct",
        "Replica-CA-v2", True)
    direct_summary, _, _ = evaluate_scenes([(gt, direct)], p)
    assert direct_summary["CA_AP50_uniform"] == 1
    assert direct_summary["CA_PQ"]["PQ"] == 1
    clouds = oracle_clouds(gt, ids)
    near = evaluate_oracle(gt, p, ids, clouds, delta_m=0.01)
    inflated = evaluate_oracle(gt, p, ids, clouds, delta_m=0.02)
    assert oracle_identity_pass(near) and near["self_iou_min"] > 0.9
    assert near["PQ"] is None  # Exact coordinates can overlap across GT labels.
    assert oracle_identity_pass(inflated) and inflated["self_iou_min"] < 0.6
    assert inflated["foreign_instance_vertex_claims"] > 20000
    sparse = oracle_clouds(gt, ids, keep_fraction=0.25, seed=20260927)
    sparse_tight = evaluate_oracle(gt, p, ids, sparse, delta_m=0.01)
    sparse_wider = evaluate_oracle(gt, p, ids, sparse, delta_m=0.015)
    assert sparse_tight["TP"] == 0 and sparse_wider["TP"] == 68
    bad = evaluate_oracle(gt, p, ids, clouds, delta_m=0.05)
    assert not oracle_identity_pass(bad)
    assert bad["TP"] == 54 and bad["FN"] == 14
