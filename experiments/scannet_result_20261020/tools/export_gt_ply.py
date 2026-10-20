"""Export frozen ScanNet GT surfaces with colors inherited from existing prediction PLYs.

Visualization only: this script does not change predictions, GT qualification,
correspondence, or evaluation. Run on the result server with numpy/scipy.
"""
from pathlib import Path
import colorsys
import hashlib
import json
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment

R = Path("/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020")
OLD = Path("/data/chenkejun/CVPR/evaluation_results/scannet_v3_r2_reeval_20261009")
METHODS = ["baseline", "E1", "swap_partial"]
PRED_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("instance_id", "<i4"), ("state", "u1"), ("surface_index", "<i4"),
])
GT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("instance_id", "<i4"), ("official_object_id", "<i4"),
    ("semantic_id", "<i4"), ("color_prediction_id", "<i4"),
    ("reference_vertex_index", "<i4"), ("evaluation_target", "u1"),
    ("observation_count", "<u4"),
])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_ply(path, dtype):
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            assert line, f"Incomplete header: {path}"
            header.append(line)
            if line == b"end_header\n":
                break
        assert b"format binary_little_endian 1.0\n" in header
        count = int(next(x for x in header if x.startswith(b"element vertex ")).split()[-1])
        arr = np.fromfile(f, dtype=dtype)
        assert len(arr) == count
        return arr


def prediction_rgb(pid):
    return (np.asarray(colorsys.hsv_to_rgb((int(pid) * .61803398875) % 1, .67, .94)) * 255).astype(np.uint8)


def write_gt(path, array, view):
    ply_types = {"f": "float", "i": "int", "u": "uint"}
    header = ["ply", "format binary_little_endian 1.0",
              "comment Ground-truth mesh vertices; no resampling or prediction label transfer",
              "comment Identity coordinates; no axisAlignment; units meters",
              f"comment {view}; color mapping is visualization only",
              f"element vertex {len(array)}"]
    for name in GT_DTYPE.names:
        dtype = GT_DTYPE.fields[name][0]
        header.append(f"property {'uchar' if dtype.itemsize == 1 else ply_types[dtype.kind]} {name}")
    header.append("end_header")
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        array.tofile(f)
    np.testing.assert_array_equal(read_ply(path, GT_DTYPE), array)


def export_scene(scene, expected_count, expected_observed):
    out = R / scene / "gt_ply"
    out.mkdir(parents=True, exist_ok=True)
    input_hashes = {}

    def record(path):
        input_hashes[str(path)] = sha(path)
        return path

    gt_path = record(OLD / scene / "ground_truth/gt.npz")
    with np.load(gt_path, allow_pickle=False) as d:
        gt = {k: d[k].copy() for k in d.files if k != "metadata_json"}
        meta = json.loads(d["metadata_json"].item())
    ids = np.unique(gt["instance_id"][gt["evaluation_region"] == 1])
    assert len(ids) == expected_count and int((gt["evaluation_region"] == 1).sum()) == expected_observed
    assert np.all(gt["instance_id"][gt["evaluation_region"] == 1] == gt["raw_instance_id"][gt["evaluation_region"] == 1])
    aggreg = json.loads(record(Path(meta["source_aggregation"])).read_text())
    categories = {int(x["objectId"]) + 1: x["label"] for x in aggreg["segGroups"]}
    overlap, palettes = {}, {}
    for method in METHODS:
        folder = R / scene / "evaluation_r2" / f"{method}_final"
        source = record(folder / "overlap_matrix.npz")
        with np.load(source, allow_pickle=False) as d:
            overlap[method] = {k: d[k].copy() for k in d.files}
        np.testing.assert_array_equal(overlap[method]["gt_ids"], ids)
        np.testing.assert_array_equal(overlap[method]["gt_size"],
                                      [np.count_nonzero(gt["instance_id"] == gid) for gid in ids])
        # Include original small correspondence evidence for portable inspection.
        shutil.copyfile(source, out / f"{method}_overlap_matrix.npz")
        adapter = json.loads(record(folder / "adapter_manifest.json").read_text())
        assert adapter["gt_scope_sha256"] == meta["gt_scope_sha256"]
        assert adapter["gt_observed_support_sha256"] == meta["gt_observed_support_sha256"]
        pred = read_ply(record(R / scene / "ply" / f"{scene}_{method}_instances.ply"), PRED_DTYPE)
        palette = {}
        for pid in np.unique(pred["instance_id"]):
            selected = pred[pred["instance_id"] == pid]
            rgb = np.column_stack([selected[n] for n in ["red", "green", "blue"]])
            color = prediction_rgb(int(pid))
            assert np.all(rgb == color)
            palette[int(pid)] = color.tolist()
        palettes[method] = palette

    # Color association only. One prediction color can identify at most one GT,
    # so a merged prediction does not make distinct GT instances indistinguishable.
    # Prefer strict baseline; use repair colors only for remaining GT instances.
    chosen, used = {}, set()
    for method in METHODS:
        d = overlap[method]
        pids = d["pred_uids"].astype(int)
        recall = d["intersection"] / d["gt_size"][None, :]
        valid = (d["iou"] >= .05) & (d["intersection"] >= 10) & (recall >= .05)
        valid[:, np.isin(ids, list(chosen))] = False
        valid[np.isin(pids, list(used)), :] = False
        weights = np.where(valid, d["iou"], 0)
        rows, cols = linear_sum_assignment(-weights)
        for r, c in zip(rows, cols):
            if weights[r, c] <= 0:
                continue
            gid, pid = int(ids[c]), int(pids[r])
            assert pid in palettes[method]
            chosen[gid] = dict(color_prediction_id=pid, color_source=method,
                               matching_iou=float(d["iou"][r, c]),
                               matching_intersection=int(d["intersection"][r, c]),
                               matching_gt_recall=float(recall[r, c]),
                               rgb=palettes[method][pid])
            used.add(pid)

    all_previous_colors = {tuple(rgb) for palette in palettes.values() for rgb in palette.values()}
    rows = []
    for c, raw_gid in enumerate(ids):
        gid = int(raw_gid)
        row = chosen.get(gid)
        if row is None:
            # Use a distinct darker palette for unmatched GT. Never pretend this
            # color represents an existing prediction instance.
            hue = (gid * .61803398875) % 1
            rgb = (np.asarray(colorsys.hsv_to_rgb(hue, .9, .7)) * 255).astype(np.uint8).tolist()
            while tuple(rgb) in all_previous_colors:
                hue = (hue + .137) % 1
                rgb = (np.asarray(colorsys.hsv_to_rgb(hue, .9, .7)) * 255).astype(np.uint8).tolist()
            row = dict(color_prediction_id=-1, color_source="new_distinct_color",
                       matching_iou=None, matching_intersection=0, matching_gt_recall=0, rgb=rgb)
        all_previous_colors.add(tuple(row["rgb"]))
        row = dict(row, raw_gt_instance_id=gid, official_object_id=gid - 1,
                   category=categories.get(gid, "unknown"),
                   full_reference_vertices=int((gt["raw_instance_id"] == gid).sum()),
                   evaluated_observed_vertices=int((gt["instance_id"] == gid).sum()))
        best = {}
        for method in METHODS:
            d = overlap[method]
            r = int(d["iou"][:, c].argmax())
            best[method] = dict(prediction_id=int(d["pred_uids"][r]),
                                iou=float(d["iou"][r, c]),
                                intersection=int(d["intersection"][r, c]))
        row["best_overlap_per_method"] = best
        row["low_overlap_visual_match"] = row["matching_iou"] is not None and row["matching_iou"] < .25
        row["color_differs_from_best_id_for_methods"] = [
            m for m in METHODS if best[m]["prediction_id"] != row["color_prediction_id"]]
        row["baseline_best_color_competing_gt_ids"] = [
            int(ids[k]) for k in range(len(ids)) if k != c and
            int(overlap["baseline"]["pred_uids"][overlap["baseline"]["iou"][:, k].argmax()])
            == best["baseline"]["prediction_id"]]
        rows.append(row)
    assert len({tuple(x["rgb"]) for x in rows}) == len(rows)
    matched_pids = [x["color_prediction_id"] for x in rows if x["color_prediction_id"] > 0]
    assert len(set(matched_pids)) == len(matched_pids)

    idx = np.flatnonzero(np.isin(gt["raw_instance_id"], ids))
    array = np.zeros(len(idx), dtype=GT_DTYPE)
    xyz = gt["xyz_ref"][idx].astype(np.float32)
    np.testing.assert_array_equal(xyz.astype(np.float64), gt["xyz_ref"][idx])
    for k, name in enumerate(["x", "y", "z"]):
        array[name] = xyz[:, k]
    array["instance_id"] = gt["raw_instance_id"][idx]
    array["official_object_id"] = array["instance_id"] - 1
    array["semantic_id"] = gt["raw_semantic_id"][idx]
    array["reference_vertex_index"] = idx
    array["evaluation_target"] = (gt["evaluation_region"][idx] == 1)
    array["observation_count"] = gt["observation_count"][idx]
    for row in rows:
        selected = array["instance_id"] == row["raw_gt_instance_id"]
        for name, value in zip(["red", "green", "blue"], row["rgb"]):
            array[name][selected] = value
        array["color_prediction_id"][selected] = row["color_prediction_id"]
    observed = array[array["evaluation_target"] == 1]
    assert len(observed) == expected_observed
    np.testing.assert_array_equal(observed["reference_vertex_index"], np.flatnonzero(gt["evaluation_region"] == 1))
    write_gt(out / f"{scene}_gt_instances.ply", array, "Full mesh surface of evaluation-qualified GT objects")
    write_gt(out / f"{scene}_gt_instances_observed.ply", observed, "Exact R2 evaluated observed target vertices")

    summary = dict(scene=scene, gt_instances=len(ids), full_reference_vertices=len(array),
                   evaluated_observed_vertices=len(observed),
                   colors_from={m: sum(x["color_source"] == m for x in rows)
                                for m in METHODS + ["new_distinct_color"]},
                   low_overlap_visual_matches=sum(x["low_overlap_visual_match"] for x in rows),
                   unmatched_gt_ids=[x["raw_gt_instance_id"] for x in rows if x["color_prediction_id"] < 0])
    mapping = dict(
        summary=summary, purpose="Visualization only; no evaluation or algorithm changes",
        coordinate_transform=meta["coordinate_transform"],
        gt_scope_sha256=meta["gt_scope_sha256"],
        gt_observed_support_sha256=meta["gt_observed_support_sha256"],
        instance_id_encoding="raw_gt_instance_id = official_objectId + 1; NOT prediction ID",
        scope="Only the same 27/34 qualified objects used in this run; full means their complete reference mesh surface, not all annotated scene objects.",
        color_policy=dict(
            priority=METHODS, one_to_one=True, objective="maximum sum of IoU per priority stage",
            min_iou=.05, min_intersection_vertices=10, min_gt_recall=.05,
            low_overlap_warning_below_iou=.25,
            note="Permissive visualization associations, not semantic identity verification or TP matches. Splits/merges can prevent exact color agreement across all methods.",
            reused_colors="Exact RGB values checked against each existing prediction PLY",
            new_colors="Distinct darker palette when all acceptable candidates are already allocated or absent"),
        source_sha256=input_hashes, instances=rows,
    )
    save_json(out / "color_mapping.json", mapping)
    lines = [
        f"# {scene} GT 实例 PLY",
        "",
        f"两个文件均包含本轮评价的 {len(ids)} 个 GT 实例，保持原始 ScanNet 坐标，单位为米。",
        f"- `{scene}_gt_instances.ply`：这些实例的完整 GT 网格表面顶点，共 {len(array):,} 点。",
        f"- `{scene}_gt_instances_observed.ply`：严格对应 R2 评价中的可见目标区域，共 {len(observed):,} 点。",
        "",
        "完整表面版包含这些目标实例的未观测区域，用于查看形状，不是扩大评价范围；不包含墙、地面等本轮非目标实例。两个文件都是顶点点云，不含三角面。",
        "",
        "配色优先匹配严格一票无修复基线，再为剩余 GT 匹配 E1、partial。沿用已有预测 PLY 的精确 RGB；同一个预测颜色只分配给一个 GT，避免合并错误掩盖两个不同物体。匹配以已冻结评价重叠矩阵的 IoU 作一对一优化，门槛为 IoU≥0.05、交集≥10 点、GT 覆盖率≥0.05。这里只建立可视化对应，不等同于正确预测；IoU<0.25 的对应标记为弱重叠。",
        "",
        f"颜色来源：基线 {summary['colors_from']['baseline']} 个，E1 {summary['colors_from']['E1']} 个，partial {summary['colors_from']['swap_partial']} 个，独立新颜色 {summary['colors_from']['new_distinct_color']} 个。弱重叠对应 {summary['low_overlap_visual_matches']} 个。",
        "",
        "PLY 的 `instance_id` 是 GT 原始 ID（官方 objectId+1），`color_prediction_id` 是借用颜色的预测 ID（-1 表示新颜色）；二者不能混用。`evaluation_target` 标记是否参与本轮评价，`reference_vertex_index` 可回查原始 GT。",
        "",
        "完整对应、三个方法各自最佳重叠实例、颜色竞争和输入哈希见 `color_mapping.json`。原预测、GT、评估结果与旧 PLY 不变。",
        "",
        "| GT ID | 类别 | 颜色来源 | 预测 ID | RGB | IoU | 注记 |",
        "|---:|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        notes = []
        if row["color_prediction_id"] < 0:
            notes.append("没有可用的独立匹配颜色")
        if row["low_overlap_visual_match"]:
            notes.append("弱重叠")
        if row["baseline_best_color_competing_gt_ids"]:
            notes.append("基线最佳颜色有合并竞争")
        score = "—" if row["matching_iou"] is None else f"{row['matching_iou']:.4f}"
        lines.append(f"| {row['raw_gt_instance_id']} | {row['category']} | {row['color_source']} | {row['color_prediction_id']} | {','.join(map(str,row['rgb']))} | {score} | {'；'.join(notes)} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path, value in input_hashes.items():
        assert sha(path) == value, f"Input modified: {path}"
    check = dict(status="PASS", summary=summary,
                 input_files_unchanged=len(input_hashes), exact_ply_binary_readback=True,
                 original_reference_coordinates_and_indices=True, distinct_gt_colors=True,
                 reused_rgb_verified_against_original_ply=True, exact_observed_target_membership=True,
                 exporter_sha256=sha(__file__))
    save_json(out / "validation.json", check)
    print(json.dumps(check, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    summaries = [export_scene("scene0011_00", 27, 78514), export_scene("scene0608_00", 34, 54224)]
    files = {}
    for summary in summaries:
        for p in sorted((R / summary["scene"] / "gt_ply").iterdir()):
            if p.is_file():
                files[p.relative_to(R).as_posix()] = dict(bytes=p.stat().st_size, sha256=sha(p))
    save_json(R / "gt_ply_delivery_manifest.json",
              dict(status="PASS", summaries=summaries, exporter_sha256=sha(__file__), files=files))
