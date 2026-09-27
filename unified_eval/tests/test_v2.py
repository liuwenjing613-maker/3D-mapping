"""Replica-CA-v2 regression cases. V1 tests remain independent and unchanged."""

import gzip
import json
import pickle
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from unified_eval.cli import cmd_eval_online_prefix, cmd_eval_scene, load_protocol
from unified_eval.conceptgraphs import adapt_map
from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import map_instances_to_reference_v2
from unified_eval.io import save_gt, save_prediction
from unified_eval.online import evaluate_online_prefixes
from unified_eval.schema import CanonicalGT, CanonicalInstance, CanonicalPrediction, EvaluationError, Protocol


def v2(*, gt_min=1, significant_vertices=1, significant_fraction=0.05):
    return Protocol.from_dict({
        "name": "Replica-CA-v2", "dataset": "Replica", "confidence_mode": "uniform",
        "ap_policy": "replica_ca_101",
        "geometry_mapping": {"method": "independent_nearest_neighbor_per_instance", "max_distance_m": 0.05},
        "instance_filter": {"min_valid_instance_vertices": gt_min, "min_prediction_vertices": 0},
        "ignore_policy": {"unmatched_prediction_void_fraction_gt": 0.5},
        "significant_overlap": {"min_intersection_vertices": significant_vertices,
                                "min_gt_fraction": significant_fraction},
    })


def scene(labels, masks, *, scene_id="toy", partition=True):
    ids = np.asarray(labels, dtype=np.int64)
    n = len(ids)
    gt = CanonicalGT(scene_id, np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))),
        ids, np.ones(n, dtype=np.int32), np.ones(n, dtype=bool), np.zeros(n, dtype=bool))
    pred = CanonicalPrediction(scene_id, n,
        [CanonicalInstance(name, np.asarray(list(vertices), dtype=np.int32)) for name, vertices in masks.items()],
        "toy", "commit", "test_v2", "Replica-CA-v2", partition)
    return gt, pred


def test_independent_projection_preserves_duplicate_and_empty_prediction():
    ref = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    points = [ref[:1].copy(), ref[:1].copy(), np.empty((0, 3))]
    mapped = map_instances_to_reference_v2(points, ref, 0.05, scene_id="toy",
        method_name="test", method_commit="commit", adapter_version="v2",
        protocol_version="Replica-CA-v2")
    assert [x.vertex_indices.tolist() for x in mapped.prediction.instances] == [[0], [0], []]
    assert mapped.prediction.is_partition is False
    assert mapped.statistics["overlapped_ref_vertices"] == 1
    gt, _ = scene([1, -1], {})
    summary, rows, _ = evaluate_scenes([(gt, mapped.prediction)], v2())
    assert summary["CA_PRF1_0_5"]["TP"] == 1
    assert summary["CA_PRF1_0_5"]["FP"] == 2
    assert rows[0]["CA_PQ"]["PQ"] is None
    assert rows[0]["structure"]["duplicate_prediction_count"] == 1
    assert rows[0]["structure"]["split_gt_count"] == 0


def test_conceptgraphs_adapter_keeps_colliding_native_ids_and_duplicate(tmp_path):
    ref = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    native = {"objects": [
        {"id": 7, "pcd_np": ref[:1]},
        {"id": 7, "pcd_np": ref[:1]},
        {"id": 9, "pcd_np": np.empty((0, 3))},
    ]}
    path = tmp_path / "map.pkl.gz"
    with gzip.open(path, "wb") as handle:
        pickle.dump(native, handle)
    mapped = adapt_map(path, ref, 0.05, scene_id="toy", method_name="test",
        method_commit="commit", protocol_version="Replica-CA-v2",
        mapping_method="independent_nearest_neighbor_per_instance")
    assert [x.instance_uid for x in mapped.prediction.instances] == ["cg-index:0", "cg-index:1", "cg-index:2"]
    assert [x.metadata["native_object_id"] for x in mapped.prediction.instances] == ["7", "7", "9"]
    assert mapped.prediction.is_partition is False


def test_small_prediction_remains_false_positive_and_significant_split():
    gt, pred = scene([1] * 1000, {"main": range(930), "fragment": range(930, 1000)})
    summary, rows, overlap = evaluate_scenes([(gt, pred)],
        v2(gt_min=100, significant_vertices=10, significant_fraction=0.05))
    assert overlap[0].dropped_prediction_uids == []
    assert summary["CA_PRF1_0_5"]["TP"] == 1
    assert summary["CA_PRF1_0_5"]["FP"] == 1
    assert summary["CA_PRF1_0_5"]["F1"] == pytest.approx(2 / 3)
    assert rows[0]["structure"]["split_gt_rate"] == 1
    assert rows[0]["structure"]["duplicate_prediction_rate"] == 0


def test_small_gt_is_ignored_in_matching_area_and_unmatched_prediction():
    gt, pred = scene([1] * 100 + [2] * 30,
                     {"large": range(100), "small_only": range(100, 130)})
    summary, rows, overlap = evaluate_scenes([(gt, pred)], v2(gt_min=100))
    assert overlap[0].ignored_small_gt_ids == [2]
    assert overlap[0].pred_void_fraction.tolist() == [0.0, 1.0]
    assert summary["CA_PRF1_0_5"]["FP"] == 0
    assert summary["CA_AP50_uniform"] == 1
    assert rows[0]["CA_PQ"]["PQ"] == 1
    gt, pred = scene([1] * 100 + [2] * 30, {"large_plus_small": range(130)})
    _, _, overlap = evaluate_scenes([(gt, pred)], v2(gt_min=100))
    assert overlap[0].pred_size.tolist() == [100]
    assert overlap[0].iou[0, 0] == 1


def test_significant_merge_excludes_single_vertex_boundary_noise():
    gt, noisy = scene([1] * 100 + [2] * 100,
                      {"a_with_noise": list(range(100)) + [100], "b": range(101, 200)})
    protocol = v2(gt_min=50, significant_vertices=10, significant_fraction=0.05)
    _, rows, _ = evaluate_scenes([(gt, noisy)], protocol)
    assert rows[0]["structure"]["merge_prediction_count"] == 0
    gt, merged = scene([1] * 100 + [2] * 100, {"ab": range(200)})
    _, rows, _ = evaluate_scenes([(gt, merged)], protocol)
    assert rows[0]["structure"]["merge_prediction_rate"] == 1


def test_pooled_and_macro_are_distinct_and_overlapping_f1_is_defined():
    one = scene([1] * 10, {"a": range(10)}, scene_id="one")
    many = scene(sum(([i] * 10 for i in range(1, 10)), []), {}, scene_id="many")
    summary, _, _ = evaluate_scenes([one, many], v2())
    assert summary["CA_PRF1_0_5"]["F1"] == pytest.approx(2 / 11)
    assert summary["macro_per_scene"]["CA_F1_0_5"] == pytest.approx(0.5)
    assert summary["CA_AP50_uniform"] != summary["macro_per_scene"]["CA_AP50_uniform"]


def test_v2_uniform_metrics_ignore_prediction_order_and_uids():
    gt, a = scene([1] * 100, {"first": range(100), "second": range(100)}, partition=False)
    _, b = scene([1] * 100, {"other": range(100), "renamed": range(100)}, partition=False)
    expected, _, _ = evaluate_scenes([(gt, a)], v2())
    actual, _, _ = evaluate_scenes([(gt, b)], v2())
    for key in ("CA_AP_uniform", "CA_AP50_uniform", "CA_PRF1_0_5", "structure"):
        assert actual[key] == expected[key]


def test_pending_v2_refuses_to_score_without_calibrated_thresholds():
    config = Path(__file__).parents[1] / "configs" / "replica_ca_v2.pending.json"
    args = type("Args", (), {"debug_max_distance_m": None,
        "debug_min_valid_instance_vertices": None,
        "debug_significant_min_vertices": None,
        "debug_significant_min_gt_fraction": None})()
    with pytest.raises(EvaluationError, match="significant_overlap thresholds"):
        load_protocol(config, args)
    args.debug_significant_min_vertices = 10
    args.debug_significant_min_gt_fraction = 0.05
    protocol, _, debug = load_protocol(config, args)
    assert protocol.is_v2 and debug
    assert protocol.prediction_min_valid_vertices == 0


def test_online_prefix_uses_only_past_depth_and_rejects_future_input(tmp_path):
    gt, _ = scene([1, 2], {})
    gt.xyz_ref = np.array([[0.0, 0, 1], [1.0, 0, 1]])
    for frame_id, depth in enumerate(([[1.0, 0.0]], [[0.0, 1.0]])):
        np.save(tmp_path / f"depth{frame_id}.npy", np.array(depth))
    np.save(tmp_path / "k.npy", np.eye(3))
    np.save(tmp_path / "pose.npy", np.eye(4))
    source = {"entries": [{"scene": "toy", "method": "toy", "cost": {"frames": 2},
                           "input": {"start": 0, "end": 2, "stride": 1}}]}
    (tmp_path / "source.json").write_text(json.dumps(source))
    for frame_id, masks in enumerate(({}, {"a": [0], "b": [1]})):
        _, pred = scene([1, 2], masks)
        pred.metadata.update({"committed_frame_id": frame_id,
                              "max_input_frame_id": frame_id})
        save_prediction(tmp_path / f"pred{frame_id}.npz", pred)
    spec = {"scene_id": "toy", "source_experiment_manifest": "source.json", "source_method": "toy",
            "frames": [{"frame_id": i, "depth_m": f"depth{i}.npy", "intrinsics": "k.npy",
                        "world_from_camera": "pose.npy"} for i in range(2)],
            "checkpoints": [{"frame_id": i, "prediction": f"pred{i}.npz"} for i in range(2)]}
    path = tmp_path / "online.json"
    path.write_text(json.dumps(spec))
    rows, provenance = evaluate_online_prefixes(gt, v2(), path)
    assert [row["observed_reference_vertices"] for row in rows] == [1, 2]
    assert [row["CA_PRF1_0_5"]["F1"] for row in rows] == [0, 1]
    assert provenance["frame_count"] == 2
    gt_file = tmp_path / "gt.npz"
    save_gt(gt_file, gt)
    config = Path(__file__).parents[1] / "configs" / "replica_ca_v2.pending.json"
    args = Namespace(config=config, gt=gt_file, online_manifest=path,
        out=tmp_path / "online_output", debug_max_distance_m=None,
        debug_min_valid_instance_vertices=1, debug_significant_min_vertices=1,
        debug_significant_min_gt_fraction=0.05)
    cmd_eval_online_prefix(args)
    output = json.loads((args.out / "prefix_metrics.json").read_text())
    assert output["status"] == "DEBUG_ONLY / NON_OFFICIAL"
    assert len(output["checkpoints"]) == 2
    assert (args.out / "prefix_curve.csv").is_file()
    _, unsafe = scene([1, 2], {})
    unsafe.metadata.update({"committed_frame_id": 0, "max_input_frame_id": 1})
    save_prediction(tmp_path / "pred0.npz", unsafe)
    with pytest.raises(EvaluationError, match="latest input frame"):
        evaluate_online_prefixes(gt, v2(), path)


def test_v2_eval_scene_writes_explicit_ignore_fraction(tmp_path):
    gt, pred = scene([1] * 100 + [2] * 30,
                     {"large": range(100), "small": range(100, 130)})
    gt_file, pred_file = tmp_path / "gt.npz", tmp_path / "pred.npz"
    save_gt(gt_file, gt)
    save_prediction(pred_file, pred)
    args = Namespace(config=Path(__file__).parents[1] / "configs" / "replica_ca_v2.pending.json",
        gt=gt_file, pred=pred_file, out=tmp_path / "evaluation",
        debug_max_distance_m=None, debug_min_valid_instance_vertices=None,
        debug_significant_min_vertices=10, debug_significant_min_gt_fraction=0.05)
    cmd_eval_scene(args)
    output = json.loads((args.out / "metrics.json").read_text())
    assert output["status"] == "DEBUG_ONLY / NON_OFFICIAL"
    assert output["CA_PRF1_0_5"]["F1"] == 1
    with np.load(args.out / "overlap_matrix.npz", allow_pickle=False) as matrix:
        assert matrix["pred_ignore_fraction"].tolist() == [0.0, 1.0]
