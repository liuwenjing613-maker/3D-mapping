import numpy as np
import pytest
import json

from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import map_point_labels_to_reference
from unified_eval.io import load_gt, load_prediction, save_gt, save_prediction
from unified_eval.metrics import build_overlap
from unified_eval.online import observed_vertices_from_depth, update_observation_count
from unified_eval.official_native import run_official_native
from unified_eval.schema import (
    CanonicalGT, CanonicalInstance, CanonicalPrediction, EvaluationError, Protocol,
)


def protocol(min_vertices=1):
    return Protocol.from_dict({
        "name": "Replica-CA-v1", "dataset": "Replica", "confidence_mode": "uniform",
        "instance_filter": {"min_valid_instance_vertices": min_vertices},
        "geometry_mapping": {"method": "nearest_neighbor_reference_to_prediction", "max_distance_m": 0.05},
        "ignore_policy": {"unmatched_prediction_void_fraction_gt": 0.5},
    })


def toy(gt_labels=None, predictions=None, partition=True):
    labels = np.array(gt_labels if gt_labels is not None else [1] * 100 + [2] * 100)
    n = len(labels)
    gt = CanonicalGT("toy", np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(float),
        labels, np.ones(n, dtype=int), np.ones(n, dtype=bool), np.zeros(n, dtype=bool))
    masks = predictions if predictions is not None else {"a": range(100), "b": range(100, 200)}
    pred = CanonicalPrediction("toy", n,
        [CanonicalInstance(uid, np.array(list(indices), dtype=np.int32), 0.5) for uid, indices in masks.items()],
        "toy_method", "test", "test_v1", "Replica-CA-v1", partition)
    return gt, pred


def evaluate(gt, pred):
    summary, rows, overlap = evaluate_scenes([(gt, pred)], protocol())
    return summary, rows[0], overlap[0]


def test_perfect_and_id_permutation():
    gt, pred = toy()
    summary, row, _ = evaluate(gt, pred)
    assert summary["CA_AP_uniform"] == summary["CA_AP50_uniform"] == summary["CA_AP25_uniform"] == 1
    assert summary["CA_PQ"]["PQ"] == 1
    assert row["CA_PQ"]["PQ"] == row["CA_PQ"]["SQ"] == row["CA_PQ"]["RQ"] == 1
    _, permuted = toy(predictions={"p53": range(100), "p17": range(100, 200)})
    other, other_row, _ = evaluate(gt, permuted)
    assert other["CA_AP_uniform"] == summary["CA_AP_uniform"]
    assert other_row["CA_PQ"] == row["CA_PQ"]


def test_missing_and_duplicate():
    gt, pred = toy(predictions={"a": range(100)})
    summary, row, _ = evaluate(gt, pred)
    assert summary["CA_AP50_uniform"] < 1
    assert row["CA_PQ"]["FN"] == 1
    gt, duplicate = toy(predictions={"a": range(100), "a2": range(100), "b": range(100, 200)}, partition=False)
    summary, row, _ = evaluate(gt, duplicate)
    assert summary["AP_by_threshold"]["0.50"]["FP"] == 1
    assert summary["CA_AP50_uniform"] == pytest.approx(2 / 3)
    assert row["CA_PQ"]["PQ"] is None


def test_merge_split_contamination():
    gt, merge = toy(predictions={"merged": range(200)})
    summary, row, overlap = evaluate(gt, merge)
    assert summary["CA_AP50_uniform"] == 0
    assert row["CA_PQ"]["TP"] == 0
    assert overlap.intersection.tolist() == [[100, 100]]
    gt, split = toy(gt_labels=[1] * 100, predictions={"left": range(50), "right": range(50, 100)})
    summary, row, overlap = evaluate(gt, split)
    assert summary["CA_AP50_uniform"] == 0
    assert row["CA_PQ"]["TP"] == 0
    assert overlap.recall[:, 0].tolist() == [0.5, 0.5]
    gt, contaminated = toy(predictions={"a_plus": range(120)})
    _, _, overlap = evaluate(gt, contaminated)
    assert overlap.iou[0, 0] == pytest.approx(100 / 120)


def test_mapping_threshold_and_sampling_density():
    ref = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])
    def mapped(points):
        return map_point_labels_to_reference(points, np.zeros(len(points), dtype=int), ref, .05,
            scene_id="toy", method_name="x", method_commit="x", adapter_version="test",
            protocol_version="Replica-CA-v1")
    a = mapped(np.array([[.01, 0, 0], [1.01, 0, 0]]))
    b = mapped(np.repeat(np.array([[.01, 0, 0], [1.01, 0, 0]]), 100, axis=0))
    assert a.prediction.instances[0].vertex_indices.tolist() == [0, 1]
    assert b.prediction.instances[0].vertex_indices.tolist() == [0, 1]
    assert a.statistics["unmapped_ref_vertices"] == 1
    distant = mapped(np.array([[1., 1., 0.]]))
    assert distant.statistics["mapped_ref_vertices"] == 0


def test_ignore_void_and_minimum_size():
    gt, pred = toy(gt_labels=[1] * 100 + [-1] * 100,
        predictions={"a": range(100), "background": range(100, 200)})
    summary, row, _ = evaluate(gt, pred)
    assert summary["CA_AP50_uniform"] == 1
    assert summary["AP_by_threshold"]["0.50"]["FP"] == 0
    assert row["CA_PQ"]["ignored_prediction_count"] == 1
    _, void_only = toy(gt_labels=[1] * 100 + [-1] * 100,
        predictions={"background": range(100, 200)})
    void_summary, _, _ = evaluate(gt, void_only)
    assert void_summary["CA_AP50_uniform"] == 0
    assert void_summary["AP_by_threshold"]["0.50"]["FP"] == 0
    gt, pred = toy(gt_labels=[1] * 100 + [2] * 3,
        predictions={"a": range(100), "tiny": range(100, 103)})
    overlap = build_overlap(gt, pred, protocol(min_vertices=5))
    assert overlap.ignored_small_gt_ids == [2]
    assert overlap.dropped_prediction_uids == ["tiny"]


def test_reject_overlapping_partition_and_missing_config():
    _, pred = toy(predictions={"a": range(100), "b": range(50, 150)}, partition=True)
    with pytest.raises(EvaluationError):
        pred.validate()
    with pytest.raises(EvaluationError, match="max_distance_m"):
        Protocol.from_dict({"name": "x", "dataset": "x", "confidence_mode": "uniform",
            "geometry_mapping": {"method": "nearest_neighbor_reference_to_prediction", "max_distance_m": None},
            "instance_filter": {"min_valid_instance_vertices": None},
            "ignore_policy": {"unmatched_prediction_void_fraction_gt": .5}})


def test_roundtrip(tmp_path):
    gt, pred = toy()
    gt.metadata["mesh"] = "test"
    gt_file, pred_file = tmp_path / "gt.npz", tmp_path / "pred.npz"
    save_gt(gt_file, gt)
    save_prediction(pred_file, pred)
    loaded_gt, loaded_pred = load_gt(gt_file), load_prediction(pred_file)
    assert loaded_gt.metadata == gt.metadata
    assert loaded_pred.instances[0].vertex_indices.tolist() == list(range(100))
    assert evaluate(loaded_gt, loaded_pred)[0]["CA_AP50_uniform"] == 1


def test_observed_surface_uses_current_depth_only():
    ref = np.array([[0., 0, 1], [5., 0, 1]])
    observed = observed_vertices_from_depth(np.array([[1.]]), np.eye(3), np.eye(4), ref, .05)
    assert observed.tolist() == [True, False]
    count = update_observation_count(np.zeros(2, dtype=np.uint16), observed)
    assert count.tolist() == [1, 0]


def test_uniform_ap_is_invariant_to_prediction_ids_order_and_file_order(tmp_path):
    base = []
    for scene_id in ("scene_a", "scene_b"):
        gt, pred = toy(predictions={"one": range(100), "duplicate": range(100),
            "two_partial": range(100, 180)}, partition=False)
        gt.scene_id = pred.scene_id = scene_id
        base.append((gt, pred))
    expected = evaluate_scenes(base, protocol())[0]
    for seed in range(12):
        rng = np.random.default_rng(seed)
        shuffled_scenes = []
        for gt, pred in base:
            order = rng.permutation(len(pred.instances))
            renamed = CanonicalPrediction(gt.scene_id, pred.reference_vertex_count,
                [CanonicalInstance(f"uid_{seed}_{i}", pred.instances[int(j)].vertex_indices.copy(), .5)
                 for i, j in enumerate(order)], pred.method_name, pred.method_commit,
                pred.adapter_version, pred.protocol_version, False)
            path = tmp_path / f"random_file_{seed}_{gt.scene_id}.npz"
            save_prediction(path, renamed)
            shuffled_scenes.append((gt, load_prediction(path)))
        rng.shuffle(shuffled_scenes)
        actual = evaluate_scenes(shuffled_scenes, protocol())[0]
        for key in ("CA_AP_uniform", "CA_AP50_uniform", "CA_AP25_uniform"):
            assert actual[key] == expected[key]
        assert actual["AP_by_threshold"] == expected["AP_by_threshold"]


def test_official_native_is_separate_from_replica_uniform(tmp_path):
    raw = {
        "name": "Replica-CA-v1", "dataset": "Replica", "confidence_mode": "native",
        "instance_filter": {"min_valid_instance_vertices": 1},
        "geometry_mapping": {"method": "nearest_neighbor_reference_to_prediction", "max_distance_m": .05},
        "ignore_policy": {"unmatched_prediction_void_fraction_gt": .5},
    }
    with pytest.raises(EvaluationError, match="separate official-native"):
        Protocol.from_dict(raw)
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"dataset":"Replica"}', encoding="utf-8")
    with pytest.raises(EvaluationError, match="Replica has no official"):
        run_official_native(recipe, tmp_path / "out")


def test_official_runner_preserves_external_metrics_without_uniform_matcher(tmp_path):
    # This tests the runner contract, not compatibility with an official dataset script.
    script = tmp_path / "fixture_evaluator.py"
    script.write_text("import json,sys\njson.dump({'ap':.37,'ap50':.41,'ap25':.58},open(sys.argv[1],'w'))\n", encoding="utf-8")
    source = tmp_path / "input.txt"
    source.write_text("fixture", encoding="utf-8")
    metric_file = tmp_path / "official_result.json"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "dataset": "ScanNet200", "official_evaluator_script": str(script),
        "official_source_commit": "fixture-only", "working_directory": str(tmp_path),
        "argv": [str(metric_file)], "input_files": [str(source)],
        "metric_file": str(metric_file),
        "metric_spec": {"format": "json", "metric_columns": {
            "AP_official_native": "ap", "AP50_official_native": "ap50", "AP25_official_native": "ap25"}},
    }), encoding="utf-8")
    metrics = run_official_native(recipe, tmp_path / "run")
    assert metrics == {"AP_official_native": .37, "AP50_official_native": .41, "AP25_official_native": .58}
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["compatibility_status"].startswith("UNVERIFIED")
