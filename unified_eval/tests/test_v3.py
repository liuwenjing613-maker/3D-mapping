"""Replica-CA-v3: competitive main metric and independent structure support."""

from argparse import Namespace
from dataclasses import replace
import gzip
import json
import pickle

import numpy as np
import pytest

from unified_eval.calibration import eligible_oracle_ids, evaluate_oracle, oracle_clouds
from unified_eval.cli import cmd_eval_scene, load_protocol
from unified_eval.conceptgraphs import adapt_map
from unified_eval.evaluate import evaluate_scenes
from unified_eval.geometry import map_instances_to_reference_v3
from unified_eval.io import save_gt, save_prediction
from unified_eval.online import evaluate_online_prefixes
from unified_eval.ovimap import adapt_export
from unified_eval.schema import CanonicalGT, EvaluationError, Protocol


def protocol():
    return Protocol.from_dict({
        "name": "Replica-CA-v3", "dataset": "Replica", "confidence_mode": "uniform",
        "geometry_mapping": {"method": "competitive_nearest_instance", "max_distance_m": .02},
        "diagnostic_mapping": {"method": "pairwise_geometry_support", "max_distance_m": .02},
        "instance_filter": {"min_valid_instance_vertices": 2, "min_prediction_vertices": 0},
        "ignore_policy": {"unmatched_prediction_void_fraction_gt": .5},
        "significant_overlap": {"min_intersection_vertices": 1, "min_gt_fraction": .1},
    })


def gt_for(labels, *, spacing=.01):
    xyz = np.column_stack((np.arange(len(labels)) * spacing,
                           np.zeros(len(labels)), np.zeros(len(labels))))
    n = len(labels)
    return CanonicalGT("toy", xyz, np.asarray(labels, dtype=np.int64),
        np.ones(n, dtype=np.int32), np.ones(n, dtype=bool), np.zeros(n, dtype=bool))


def mapped(gt, clouds, *, delta=.02, diag=.02):
    return map_instances_to_reference_v3(clouds, gt.xyz_ref, delta,
        scene_id=gt.scene_id, method_name="oracle", method_commit="test",
        adapter_version="test_v3", protocol_version="Replica-CA-v3",
        diagnostic_distance_m=diag)


def score(gt, value):
    summary, rows, _ = evaluate_scenes([(gt, value.prediction)], protocol(),
        diagnostic_predictions=[value.diagnostic_prediction])
    return summary, rows[0]


def test_adjacent_exact_oracle_exposes_missing_neighbor_leakage():
    xyz = np.array([[0., 0, 0], [.002, 0, 0], [.004, 0, 0], [.006, 0, 0],
                    [0., .012, 0], [.002, .012, 0], [.004, .012, 0], [.006, .012, 0]])
    gt = gt_for([1] * 4 + [2] * 4)
    gt.xyz_ref = xyz
    ids = eligible_oracle_ids(gt, 2)
    clouds = oracle_clouds(gt, ids)
    partial = evaluate_oracle(gt, protocol(), ids[:1], clouds[:1], delta_m=.02)
    full = evaluate_oracle(gt, protocol(), ids, clouds, delta_m=.02)
    # Unique ownership stops cross-claims only when the adjacent native object
    # is present. With it absent, the radius gate still leaks into that GT.
    assert partial["TP"] == 0 and full["TP"] == 2
    assert partial["foreign_instance_vertex_claims"] == 4
    assert full["AP50"] == full["F1_0_5"] == full["PQ"] == 1
    assert full["foreign_instance_vertex_claims"] == 0
    assert full["overlapped_ref_vertices"] == 0


def test_empty_and_shifted_duplicate_are_fp_but_diagnostic_recognizes_duplicate():
    gt = gt_for([1] * 4, spacing=.002)
    duplicate = gt.xyz_ref + np.array([0., .006, 0.])
    value = mapped(gt, [gt.xyz_ref, duplicate, np.empty((0, 3))])
    assert [len(x.vertex_indices) for x in value.prediction.instances] == [4, 0, 0]
    summary, row = score(gt, value)
    assert summary["CA_PRF1_0_5"]["TP"] == 1
    assert summary["CA_PRF1_0_5"]["FP"] == 2
    assert row["structure"]["duplicate_prediction_count"] == 1
    assert row["CA_PQ"]["PQ"] == pytest.approx(0.5)


def test_exact_coordinate_duplicate_is_not_split_by_kdtree_ties():
    gt = gt_for([1] * 4, spacing=.002)
    value = mapped(gt, [gt.xyz_ref.copy(), gt.xyz_ref.copy()])
    assert sorted(len(x.vertex_indices) for x in value.prediction.instances) == [0, 4]
    summary, row = score(gt, value)
    assert summary["CA_PRF1_0_5"]["TP"] == 1
    assert summary["CA_PRF1_0_5"]["FP"] == 1
    assert row["structure"]["duplicate_prediction_count"] == 1


def test_merge_and_split_diagnostics_use_pairwise_support_not_competitive_overlap():
    gt = gt_for([1] * 4 + [2] * 4, spacing=.1)
    a, b = gt.xyz_ref[:4], gt.xyz_ref[4:]
    merged = mapped(gt, [np.vstack((a, b))], delta=.02, diag=.02)
    merge_summary, merge_row = score(gt, merged)
    assert merge_summary["CA_PRF1_0_5"]["TP"] == 0
    assert merge_row["structure"]["merge_prediction_count"] == 1
    one = gt_for([1] * 8, spacing=.1)
    split = mapped(one, [one.xyz_ref[:4], one.xyz_ref[4:]], delta=.02, diag=.02)
    split_summary, split_row = score(one, split)
    assert split_summary["CA_PRF1_0_5"]["FP"] == 2
    assert split_row["structure"]["split_gt_count"] == 1


def test_prediction_and_point_permutation_do_not_change_uniform_metrics():
    gt = gt_for([1] * 4 + [2] * 4, spacing=.1)
    clouds = [gt.xyz_ref[:4], gt.xyz_ref[4:], gt.xyz_ref[:4] + [0, .006, 0]]
    base = score(gt, mapped(gt, clouds))[0]
    rng = np.random.default_rng(20260927)
    for _ in range(12):
        order = rng.permutation(len(clouds))
        shuffled = [clouds[i][rng.permutation(len(clouds[i]))] for i in order]
        got = score(gt, mapped(gt, shuffled))[0]
        for key in ("CA_AP50_uniform", "CA_AP_uniform", "CA_PRF1_0_5", "CA_PQ", "structure"):
            assert got[key] == base[key]


def test_native_adapters_keep_duplicate_ids_empty_instances_and_pairwise_support(tmp_path):
    gt = gt_for([1] * 4)
    objects = {"objects": [{"id": 7, "pcd_np": gt.xyz_ref},
                           {"id": 7, "pcd_np": gt.xyz_ref + [0, .006, 0]},
                           {"id": 8, "pcd_np": np.empty((0, 3))}]}
    map_path = tmp_path / "map.pkl.gz"
    with gzip.open(map_path, "wb") as handle:
        pickle.dump(objects, handle)
    cg = adapt_map(map_path, gt.xyz_ref, .02, scene_id="toy", method_name="cg",
        method_commit="test", protocol_version="Replica-CA-v3",
        mapping_method="competitive_nearest_instance", diagnostic_distance_m=.02)
    assert [x.instance_uid for x in cg.prediction.instances] == ["cg-index:0", "cg-index:1", "cg-index:2"]
    assert cg.prediction.is_partition
    assert score(gt, cg)[0]["CA_PRF1_0_5"]["FP"] == 2
    assert cg.diagnostic_prediction.metadata["role"] == "structure_diagnostics_only"
    export_path = tmp_path / "ovi.npz"
    np.savez(export_path, xyz=np.vstack((objects["objects"][0]["pcd_np"],
        objects["objects"][1]["pcd_np"])), instance=np.array([0] * 4 + [1] * 4),
        native_instance_ids=np.array([7, 9, 11]))
    ovi = adapt_export(export_path, gt.xyz_ref, .02, scene_id="toy", method_name="ovi",
        method_commit="test", protocol_version="Replica-CA-v3",
        mapping_method="competitive_nearest_instance", diagnostic_distance_m=.02)
    assert [x.instance_uid for x in ovi.prediction.instances] == ["ovi-id:7", "ovi-id:9", "ovi-id:11"]
    assert len(ovi.prediction.instances[2].vertex_indices) == 0
    assert score(gt, ovi)[0]["CA_PRF1_0_5"]["FP"] == 2


def test_diagnostic_radius_cannot_change_main_ap_f1_or_pq():
    gt = gt_for([1] * 4 + [2] * 4, spacing=.1)
    clouds = [gt.xyz_ref[:4], gt.xyz_ref[4:]]
    narrow = mapped(gt, clouds, diag=.01)
    wide = mapped(gt, clouds, diag=.2)
    a, _ = score(gt, narrow)
    b, _ = score(gt, wide)
    for key in ("CA_AP_uniform", "CA_AP50_uniform", "CA_PRF1_0_5", "CA_PQ"):
        assert a[key] == b[key]
    with pytest.raises(ValueError, match="main prediction"):
        evaluate_scenes([(gt, narrow.diagnostic_prediction)], protocol())


def test_v3_pending_config_fails_closed_and_debug_cli_keeps_diagnostic_separate(tmp_path):
    from pathlib import Path
    config = Path(__file__).parents[1] / "configs" / "replica_ca_v3.pending.json"
    args = Namespace(config=config, debug_max_distance_m=None,
        debug_diagnostic_max_distance_m=None, debug_min_valid_instance_vertices=None,
        debug_significant_min_vertices=None, debug_significant_min_gt_fraction=None)
    with pytest.raises(EvaluationError, match="geometry_mapping.max_distance_m"):
        load_protocol(config, args)
    args.debug_max_distance_m = .02
    with pytest.raises(EvaluationError, match="min_valid_instance_vertices"):
        load_protocol(config, args)
    args.debug_min_valid_instance_vertices = 2
    with pytest.raises(EvaluationError, match="diagnostic_mapping.max_distance_m"):
        load_protocol(config, args)
    args.debug_diagnostic_max_distance_m = .02
    args.debug_significant_min_vertices = 1
    args.debug_significant_min_gt_fraction = .1
    loaded, _, debug = load_protocol(config, args)
    assert loaded.is_v3 and debug
    gt = gt_for([1] * 4)
    value = mapped(gt, [gt.xyz_ref])
    gt_file = tmp_path / "gt.npz"
    pred_file = tmp_path / "canonical_prediction.npz"
    diag_file = tmp_path / "diagnostic_support_prediction.npz"
    save_gt(gt_file, gt)
    save_prediction(pred_file, value.prediction)
    save_prediction(diag_file, value.diagnostic_prediction)
    args.gt, args.pred, args.out, args.diagnostic_pred = gt_file, pred_file, tmp_path / "out", diag_file
    cmd_eval_scene(args)
    metrics = json.loads((args.out / "metrics.json").read_text())
    assert metrics["status"] == "DEBUG_ONLY / NON_OFFICIAL"
    assert metrics["CA_PQ"]["PQ"] == 1
    with np.load(args.out / "pairwise_support_matrix.npz", allow_pickle=False) as support:
        assert support["supported_gt_fraction"].tolist() == [[1.0]]
    assert not (args.out / "summary.csv").exists()
    args.diagnostic_pred = tmp_path / "missing.npz"
    with pytest.raises(EvaluationError, match="diagnostic support"):
        cmd_eval_scene(args)


def test_v3_online_prefix_requires_checkpoint_support_and_observation_gate(tmp_path):
    gt = gt_for([1, 2], spacing=1)
    gt.xyz_ref = np.array([[0., 0, 1], [1., 0, 1]])
    for frame_id, depth in enumerate(([[1., 0.]], [[0., 1.]])):
        np.save(tmp_path / f"depth{frame_id}.npy", np.array(depth))
    np.save(tmp_path / "k.npy", np.eye(3))
    np.save(tmp_path / "pose.npy", np.eye(4))
    source = {"entries": [{"scene": "toy", "method": "toy", "cost": {"frames": 2},
                           "input": {"start": 0, "end": 2, "stride": 1}}]}
    (tmp_path / "source.json").write_text(json.dumps(source))
    for frame_id, clouds in enumerate(([], [gt.xyz_ref[:1], gt.xyz_ref[1:]])):
        value = mapped(gt, clouds)
        value.prediction.metadata.update({"committed_frame_id": frame_id,
                                          "max_input_frame_id": frame_id})
        save_prediction(tmp_path / f"pred{frame_id}.npz", value.prediction)
        save_prediction(tmp_path / f"diag{frame_id}.npz", value.diagnostic_prediction)
    spec = {"scene_id": "toy", "source_experiment_manifest": "source.json",
            "source_method": "toy", "observation_max_distance_m": .02,
            "frames": [{"frame_id": i, "depth_m": f"depth{i}.npy", "intrinsics": "k.npy",
                        "world_from_camera": "pose.npy"} for i in range(2)],
            "checkpoints": [{"frame_id": i, "prediction": f"pred{i}.npz",
                             "diagnostic_prediction": f"diag{i}.npz"} for i in range(2)]}
    manifest = tmp_path / "online.json"
    manifest.write_text(json.dumps(spec))
    rows, provenance = evaluate_online_prefixes(gt, replace(protocol(), min_valid_instance_vertices=1), manifest)
    assert [row["observed_reference_vertices"] for row in rows] == [1, 2]
    assert [row["CA_PRF1_0_5"]["F1"] for row in rows] == [0, 1]
    assert provenance["observation_max_distance_m"] == .02
    del spec["observation_max_distance_m"]
    manifest.write_text(json.dumps(spec))
    with pytest.raises(EvaluationError, match="observation_max_distance_m"):
        evaluate_online_prefixes(gt, replace(protocol(), min_valid_instance_vertices=1), manifest)
