"""S0 acceptance cases against the real unified_eval implementation."""

import numpy as np

from unified_eval.evaluate import evaluate_scenes
from unified_eval.schema import CanonicalGT, CanonicalInstance, CanonicalPrediction, Protocol


def case(prediction_groups):
    n = 400
    gt = CanonicalGT(
        "s0_toy",
        np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(float),
        np.array([1] * 200 + [2] * 200),
        np.ones(n, dtype=int),
        np.ones(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )
    pred = CanonicalPrediction(
        "s0_toy", n,
        [CanonicalInstance(uid, np.array(list(indices), dtype=np.int32), 1.0)
         for uid, indices in prediction_groups.items()],
        "s0_toy_method", "s0_test", "s0_test", "Replica-CA-v1", True,
    )
    proto = Protocol.from_dict({
        "name": "Replica-CA-v1", "dataset": "Replica", "confidence_mode": "uniform",
        "instance_filter": {"min_valid_instance_vertices": 100},
        "geometry_mapping": {
            "method": "nearest_neighbor_reference_to_prediction", "max_distance_m": 0.05,
        },
        "ignore_policy": {"unmatched_prediction_void_fraction_gt": 0.5},
    })
    return evaluate_scenes([(gt, pred)], proto)


def test_a_perfect():
    summary, rows, _ = case({"a": range(200), "b": range(200, 400)})
    assert summary["CA_AP50_uniform"] == 1
    assert summary["CA_PQ"]["PQ"] == 1
    assert rows[0]["diagnostics"]["macro_best_gt_recall"] == 1
    assert rows[0]["diagnostics"]["macro_prediction_purity"] == 1


def test_b_empty():
    summary, _, _ = case({})
    assert summary["CA_AP50_uniform"] == 0
    assert summary["CA_PQ"]["PQ"] == 0


def test_c_merge():
    perfect, _, _ = case({"a": range(200), "b": range(200, 400)})
    merged, _, _ = case({"merged": range(400)})
    assert merged["CA_AP50_uniform"] < perfect["CA_AP50_uniform"]
    assert merged["CA_PQ"]["PQ"] < perfect["CA_PQ"]["PQ"]


def test_d_split():
    perfect, _, _ = case({"a": range(200), "b": range(200, 400)})
    split, _, _ = case({"a1": range(100), "a2": range(100, 200), "b": range(200, 400)})
    assert split["CA_AP50_uniform"] < perfect["CA_AP50_uniform"]
    assert split["CA_PQ"]["PQ"] < perfect["CA_PQ"]["PQ"]


def test_e_repeat():
    first, first_rows, first_overlap = case({"a": range(200), "b": range(200, 400)})
    second, second_rows, second_overlap = case({"a": range(200), "b": range(200, 400)})
    assert first == second
    assert first_rows == second_rows
    for field in ("intersection", "iou", "precision", "recall"):
        np.testing.assert_array_equal(getattr(first_overlap[0], field), getattr(second_overlap[0], field))
