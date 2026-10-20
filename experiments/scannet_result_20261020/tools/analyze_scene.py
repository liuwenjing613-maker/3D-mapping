"""Read frozen ScanNet outputs only; write derived tables and point differences."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root, scene, out = args.results.resolve(), args.scene, args.output.resolve()
source = root / scene
out.mkdir(parents=True, exist_ok=True)
input_hashes = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = source / relative
    input_hashes[f"{scene}/{relative}"] = digest(path)
    return json.loads(path.read_text(encoding="utf-8"))


comparison = read("evaluation_r2/comparison.json")
assert comparison["status"] == "PASS"
assert comparison["paired_geometry_scope_correspondence_equal"]
rows = {(r["method"], r["stage"]): r for r in comparison["rows"]}
methods = ["baseline", "E1", "swap_partial"]
names = {"baseline": "严格一票无修复", "E1": "E1（swap 种子）",
         "swap_partial": "auto-v3-swap partial"}
metrics = {m: read(f"evaluation_r2/{m}_final/metrics.json")["summary"] for m in methods}
pairs = {m: read(f"evaluation_r2/paired/{m}_final.json") for m in methods[1:]}
manifests = {m: read(f"evaluation_r2/{m}_final/adapter_manifest.json") for m in methods}
freeze = read("partial/predictions_freeze.json")
policy = read("partial/policy.json")
selection = read("e1_repair/confirmed_selection.json")
automatic = read("v2/selection.json")
read("evaluation_r2/protocol_freeze.json")
lock_keys = ["geometry_xyz_sha256", "correspondence_sha256", "gt_scope_sha256",
             "gt_observed_support_sha256", "profile_config_sha256",
             "profile_semantics_sha256", "evaluator_code_sha256"]
locks = {k: manifests["baseline"][k] for k in lock_keys}
for m in methods:
    assert all(manifests[m][k] == locks[k] for k in lock_keys)
    s = metrics[m]
    r = rows[m, "final"]
    assert r["TP"] + r["FN"] == comparison["GT"]
    assert r["F1"] == s["CA_PRF1_0_5"]["F1"]
    assert math.isclose(r["PQ"], s["CA_PQ"]["RQ"] * s["CA_PQ"]["SQ"], abs_tol=1e-14)

summary = {}
for m in methods:
    s = metrics[m]
    r = rows[m, "final"]
    summary[m] = dict(r, precision=s["CA_PRF1_0_5"]["P"],
                     recall=s["CA_PRF1_0_5"]["R"], SQ=s["CA_PQ"]["SQ"],
                     RQ=s["CA_PQ"]["RQ"], AP25=s["CA_AP25_uniform"],
                     scored_prediction_count=r["TP"] + r["FP"],
                     ignored_prediction_count=s["CA_PRF1_0_5"]["ignored_prediction_count"],
                     main_structure=s["structure"],
                     auxiliary_structure=s["auxiliary_geometry_structure"])

metric_keys = ["AP", "AP50", "AP25", "F1", "PQ", "mCov", "precision", "recall", "SQ"]
count_keys = ["TP", "FP", "FN", "gray_total", "scored_prediction_count",
              "ignored_prediction_count"]


def deltas(before, after):
    return {
        "metric_delta_percentage_points": {k: 100 * (after[k] - before[k]) for k in metric_keys},
        "count_delta": {k: after[k] - before[k] for k in count_keys},
        "owner_delta_vertices": {
            k: after["owner_surface"][k] - before["owner_surface"][k]
            for k in ["correct_owner_vertices", "wrong_owner_vertices",
                      "unassigned_with_geometry_target_vertices", "no_geometry_target_vertices"]
        },
    }


vs_baseline = {m: deltas(summary["baseline"], summary[m]) for m in methods[1:]}
partial_vs_e1 = deltas(summary["E1"], summary["swap_partial"])
by_gt = {}
for m, pair in pairs.items():
    assert len(pair["per_gt"]) == comparison["GT"]
    assert len({r["raw_gt_id"] for r in pair["per_gt"]}) == comparison["GT"]
    assert math.isclose(np.mean([r["delta_iou"] for r in pair["per_gt"]]),
                        summary[m]["mCov"] - summary["baseline"]["mCov"], abs_tol=1e-14)
    improved = sorted([r for r in pair["per_gt"] if r["delta_iou"] > 0],
                      key=lambda r: (-r["delta_iou"], r["raw_gt_id"]))
    degraded = sorted([r for r in pair["per_gt"] if r["delta_iou"] < 0],
                      key=lambda r: (r["delta_iou"], r["raw_gt_id"]))
    by_gt[m] = {
        "definition": pair["instance_change_definition"],
        "improved": len(improved), "degraded": len(degraded),
        "unchanged": sum(r["delta_iou"] == 0 for r in pair["per_gt"]),
        "sum_positive_delta_iou": sum(r["delta_iou"] for r in improved),
        "sum_negative_delta_iou": sum(r["delta_iou"] for r in degraded),
        "mean_delta_iou": float(np.mean([r["delta_iou"] for r in pair["per_gt"]])),
        "top_improvements": improved[:5], "top_degradations": degraded[:5],
        "all_per_gt": pair["per_gt"],
        "best_iou_threshold_crossings": {
            str(t): {
                "up_raw_gt_ids": [r["raw_gt_id"] for r in pair["per_gt"]
                                 if r["before_best_iou"] < t <= r["after_best_iou"]],
                "down_raw_gt_ids": [r["raw_gt_id"] for r in pair["per_gt"]
                                   if r["after_best_iou"] < t <= r["before_best_iou"]]
            } for t in [.25, .5]
        },
    }
    assert by_gt[m]["improved"] == pair["corrected_instances"]
    assert by_gt[m]["degraded"] == pair["degraded_instances"]
e1_gt = {r["raw_gt_id"]: r for r in pairs["E1"]["per_gt"]}
partial_gt = {r["raw_gt_id"]: r for r in pairs["swap_partial"]["per_gt"]}
partial_per_gt = []
for gid in sorted(e1_gt):
    a, b = e1_gt[gid], partial_gt[gid]
    assert a["before_best_iou"] == b["before_best_iou"]
    partial_per_gt.append(dict(raw_gt_id=gid, E1_best_iou=a["after_best_iou"],
                               partial_best_iou=b["after_best_iou"],
                               delta_iou=b["after_best_iou"] - a["after_best_iou"]))

dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                  ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                  ("instance_id", "<i4"), ("state", "u1"), ("surface_index", "<i4")])


def load_ply(method):
    chunks = []
    for kind in ["instances", "three_states_black_yellow_red"]:
        rel = f"ply/{scene}_{method}_{kind}.ply"
        path = source / rel
        input_hashes[f"{scene}/{rel}"] = digest(path)
        with path.open("rb") as f:
            header = []
            while True:
                line = f.readline()
                assert line, "Incomplete PLY header"
                header.append(line)
                if line == b"end_header\n":
                    break
            assert b"format binary_little_endian 1.0\n" in header
            size = int(next(s for s in header if s.startswith(b"element vertex ")).split()[-1])
            array = np.fromfile(f, dtype=dtype)
            assert len(array) == size
            chunks.append(array)
    all_points = np.concatenate(chunks)
    all_points = all_points[np.argsort(all_points["surface_index"])]
    np.testing.assert_array_equal(all_points["surface_index"], np.arange(len(all_points)))
    return all_points


point_arrays = {m: load_ply(m) for m in methods}
for m in methods[1:]:
    for k in ["x", "y", "z"]:
        np.testing.assert_array_equal(point_arrays["baseline"][k], point_arrays[m][k])
surface_summary = {}
for m, a in point_arrays.items():
    assert int((a["instance_id"] <= 0).sum()) == summary[m]["gray_total"]
    states, counts = np.unique(a["state"], return_counts=True)
    surface_summary[m] = {
        "surface_vertices": len(a),
        "assigned_vertices": int((a["instance_id"] > 0).sum()),
        "unassigned_vertices": int((a["instance_id"] <= 0).sum()),
        "positive_instance_ids": len(np.unique(a["instance_id"][a["instance_id"] > 0])),
        "native_state_counts": {str(s): int(c) for s, c in zip(states, counts)},
    }
surface_delta = {}
partial_changed = []
for before, after in [("baseline", "E1"), ("baseline", "swap_partial"), ("E1", "swap_partial")]:
    a, b = point_arrays[before], point_arrays[after]
    d = a["instance_id"] != b["instance_id"]
    sd = a["state"] != b["state"]
    surface_delta[f"{before}_to_{after}"] = {
        "label_changed_vertices": int(d.sum()), "native_state_changed_vertices": int(sd.sum()),
        "label_or_native_state_changed_vertices": int((d | sd).sum()),
        "unassigned_to_assigned_vertices": int(((a["instance_id"] <= 0) & (b["instance_id"] > 0)).sum()),
        "assigned_to_unassigned_vertices": int(((a["instance_id"] > 0) & (b["instance_id"] <= 0)).sum()),
        "different_positive_id_vertices": int(((a["instance_id"] > 0) & (b["instance_id"] > 0) & d).sum()),
        "label_changed_fraction_of_full_surface": float(d.mean()),
        "scope": "complete fixed TSDF; not limited to observed evaluable GT vertices",
        "state_semantics": "native evidence state stored alongside final postprocessed labels",
    }
    if before == "E1":
        for i in np.flatnonzero(d | sd):
            partial_changed.append(dict(surface_index=int(i),
                                        E1_instance_id=int(a["instance_id"][i]),
                                        partial_instance_id=int(b["instance_id"][i]),
                                        E1_native_state=int(a["state"][i]),
                                        partial_native_state=int(b["state"][i])))

delivery = json.loads((root / "delivery_manifest.json").read_text(encoding="utf-8"))
for rel, h in input_hashes.items():
    assert delivery["sha256"][rel] == h, f"Delivery hash mismatch: {rel}"
result = {
    "status": "PASS",
    "scene": scene, "primary_baseline": "strict_one_vote_no_repair",
    "scope": dict(GT=comparison["GT"],
                  frame_count=manifests["baseline"]["source_provenance"]["frame_count"],
                  fixed_locks=locks, observed_evaluable_gt_vertices=summary["baseline"]["owner_surface"]["target_vertices"]),
    "protocol": {"profile": comparison["profile"], "frozen": comparison["frozen"],
                 "scoring_commit": comparison["scoring_commit"], "entry": comparison["entry"]},
    "final_metrics": summary,
    "vs_strict_no_repair": vs_baseline,
    "partial_vs_E1": partial_vs_e1,
    "threshold_metrics": {m: metrics[m]["AP_by_threshold"] for m in methods},
    "paired_best_gt_iou": by_gt,
    "partial_vs_E1_per_gt": partial_per_gt,
    "surface_summary": surface_summary,
    "surface_changes": surface_delta,
    "partial_vs_E1_changed_surface_rows": partial_changed,
    "native_to_final": {m: {
        "metric_delta_percentage_points": {k: 100 * (rows[m, "final"][k] - rows[m, "native"][k])
                                            for k in ["AP", "AP50", "F1", "PQ", "mCov"]},
        "gray_delta": rows[m, "final"]["gray_total"] - rows[m, "native"]["gray_total"]
    } for m in methods},
    "audit": {
        "automatic_seed_count": len(automatic["retained_group_ids"]),
        "executed_seed_count": len(selection["included"]),
        "zero_projection_skipped": selection.get("execution_skipped_seeds", []),
        "GT_read_by_mapper": freeze["GT_read_by_mapper"],
        "all_recorded_replay_checks_true": all(freeze["checks"].values()),
        "replay_checks": freeze["checks"],
        "partial_policy": policy,
    },
    "historical_reference_only": comparison["historical_rows"],
    "limitations": [
        "Single-scene conclusions; no inference of general ScanNet performance.",
        "GT IDs are raw source IDs; no semantic names inferred.",
        "Pair improvement means per-GT best IoU improvement, not a new TP.",
        "Full-TSDF unassigned vertices differ from unassigned observed-evaluable GT vertices.",
        "Main partition structure metrics and auxiliary geometry metrics are reported separately.",
        "Fragmentation is a descriptive diagnosis; tracking or seed-selection causality has not been isolated.",
        "Historical methods do not form a controlled repair comparison against this strict baseline.",
        "Original audit key all_400_frames is retained; this run uses the locked actual frame count.",
    ],
    "source_sha256": input_hashes,
    "analysis_script_sha256": digest(Path(__file__)),
}
assert result["audit"]["all_recorded_replay_checks_true"]
(out / f"analysis_{scene}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2.15]})
colors = ["#445c78", "#e38945", "#3a9d91"]
x = np.arange(2)
for k, m in enumerate(methods):
    values = [summary[m][key] * 100 for key in ["F1", "PQ"]]
    bars = axes[0].bar(x + (k - 1) * .24, values, width=.24, color=colors[k],
                      label=["Strict no repair", "E1", "Swap partial"][k])
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
axes[0].set_xticks(x, ["F1 @ IoU 0.5", "PQ"])
axes[0].set_ylabel("Score (%)")
axes[0].set_title("Final instance metrics")
axes[0].set_ylim(0, max(summary[m]["F1"] * 100 for m in methods) * 1.2)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
gids = sorted(e1_gt, key=lambda gid: e1_gt[gid]["delta_iou"])
gx = np.arange(len(gids))
for k, m in enumerate(["E1", "swap_partial"]):
    paired = {r["raw_gt_id"]: r for r in pairs[m]["per_gt"]}
    axes[1].bar(gx + (k - .5) * .38, [paired[gid]["delta_iou"] for gid in gids],
                width=.38, color=colors[k + 1], label=["E1", "Swap partial"][k])
axes[1].axhline(0, color="#455", linewidth=.8)
axes[1].set_xticks(gx, [str(gid) for gid in gids], fontsize=8)
axes[1].set_xlabel("Raw GT ID (sorted by E1 change)")
axes[1].set_ylabel("Best IoU change vs strict no repair")
axes[1].set_title("Paired GT changes; not TP changes")
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle(f"ScanNet {scene} | {comparison['GT']} fixed GT | R2 development protocol", fontsize=13)
fig.text(.5, .012, "Fixed geometry / GT scope / correspondence. Partial and E1 may overlap at this scale.",
         ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=[0, .035, 1, .96])
figure = out / f"analysis_{scene}.png"
fig.savefig(figure, dpi=160, facecolor="white")
plt.close(fig)

b, e, p = [summary[m] for m in methods]
pd = surface_delta["E1_to_swap_partial"]
if all(summary[m]["F1"] < b["F1"] and summary[m]["PQ"] < b["PQ"] for m in methods[1:]):
    outcome = ("本场景 E1 和 partial 的 F1、PQ 均低于严格一票无修复基线，"
               "局部最佳 IoU 改善没有转化成整体收益。")
else:
    outcome = "本场景按严格一票无修复基线计算收益，具体指标及变化如下。"
same_structure = all(e["main_structure"][k] == p["main_structure"][k]
                     for k in ["split_gt_count", "merge_prediction_count"])
thresholds = [f"{t/100:.2f}" for t in range(50, 100, 5)]
same_tp = all(metrics["baseline"]["AP_by_threshold"][t]["TP"] ==
              metrics["E1"]["AP_by_threshold"][t]["TP"] for t in thresholds)
same_threshold_text = (
    "在 IoU 0.50–0.95 的全部 AP 阈值上，E1 与基线的 TP 数相同，额外 FP 是 AP 下降的直接计分因素。"
    "这说明本次 AP 下降主要体现为精度损失，不等于每个匹配实例的 IoU 都下降。"
    if same_tp else "各 AP 阈值的 TP/FP/FN 全量数据保存在结构化 JSON，可区分召回变化与额外预测。"
)
lines = [
    f"# ScanNet {scene}：严格一票无修复、E1 与 auto-v3-swap partial",
    "",
    "本报告只分析已冻结输出，不修改实现、筛选或参数。唯一主基线为严格一票无修复；历史结果单独列作参考。",
    "",
    outcome,
    "",
    f"![最终指标和逐 GT 最佳 IoU 变化](analysis_{scene}.png)",
    "",
    f"[结构化分析与全部差异](analysis_{scene}.json) · "
    "[原始总体指标](evaluation_r2/comparison.json) · "
    "[E1 逐 GT 原始结果](evaluation_r2/paired/E1_final.json) · "
    "[partial 逐 GT 原始结果](evaluation_r2/paired/swap_partial_final.json)",
    "",
    f"本场景采用 {result['scope']['frame_count']} 个已有步长 10 帧、{comparison['GT']} 个固定可评价 GT、"
    f"{result['scope']['observed_evaluable_gt_vertices']:,} 个可评价已观测 GT 表面顶点。"
    f"三组共用 {surface_summary['baseline']['surface_vertices']:,} 点的固定 TSDF、GT 范围与对应索引；各项哈希已核对。"
    "评分为 main 锁定的 object_observed_repair / revision 2，通过已批准的 ScanNet 数据适配入口调用；"
    "开发协议 frozen=false，不是官方 ScanNet 排行榜结果，也不把 Replica 350 GT 范围套用于本场景。",
    "",
    "## 整体结果",
    "",
    "指标以百分数展示，变化为百分点；TP/FP/FN 使用 IoU 0.5。AP/AP50 采用协议的统一置信度计算。",
    "",
    "| 方法 | AP | AP50 | F1 | PQ | mCov | TP / FP / FN |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for m in methods:
    r = summary[m]
    lines.append(f"| {names[m]} | {r['AP']*100:.4f} | {r['AP50']*100:.4f} | {r['F1']*100:.4f} | "
                 f"{r['PQ']*100:.4f} | {r['mCov']*100:.4f} | {r['TP']} / {r['FP']} / {r['FN']} |")
lines += ["", "| 相对严格基线 | ΔAP | ΔAP50 | ΔF1 | ΔPQ | ΔmCov |",
          "|---|---:|---:|---:|---:|---:|"]
for m in methods[1:]:
    d = vs_baseline[m]["metric_delta_percentage_points"]
    lines.append(f"| {names[m]} | " + " | ".join(f"{d[k]:+.4f}" for k in ["AP", "AP50", "F1", "PQ", "mCov"]) + " |")
lines += [
    "",
    f"E1 的 TP/FN 为 {e['TP']}/{e['FN']}，基线为 {b['TP']}/{b['FN']}；FP 从 {b['FP']} 增至 {e['FP']}。"
    f"精度从 {b['precision']*100:.4f}% 变为 {e['precision']*100:.4f}%，"
    f"召回率从 {b['recall']*100:.4f}% 变为 {e['recall']*100:.4f}%。"
    f"PQ 中的 SQ 从 {b['SQ']*100:.4f}% 变为 {e['SQ']*100:.4f}%，"
    f"RQ/F1 从 {b['RQ']*100:.4f}% 变为 {e['RQ']*100:.4f}%。"
    "因此应同时检查匹配质量和额外预测数量，不能只看局部修复数。",
    "",
    same_threshold_text,
    "",
    "## 逐 GT：改善个数不能替代改善幅度",
    "",
    f"E1 相对基线：{by_gt['E1']['improved']} 个 GT 最佳 IoU 上升、"
    f"{by_gt['E1']['degraded']} 个下降、{by_gt['E1']['unchanged']} 个不变。"
    f"全部正向 ΔIoU 之和为 {by_gt['E1']['sum_positive_delta_iou']:.6f}，"
    f"全部负向之和为 {by_gt['E1']['sum_negative_delta_iou']:.6f}；"
    f"平均 ΔIoU 为 {by_gt['E1']['mean_delta_iou']:.6f}，与 mCov 的变化一致。"
    "这里的“改善”仅指同一 GT 的最佳 IoU 增大，不能解读为新增 TP。",
    "",
    "| E1 变化最大的 GT | 基线最佳 IoU | E1 最佳 IoU | ΔIoU |",
    "|---|---:|---:|---:|",
]
for kind, entries in [("改善", by_gt["E1"]["top_improvements"]), ("退化", by_gt["E1"]["top_degradations"])]:
    for r in entries:
        lines.append(f"| {kind} / raw GT {r['raw_gt_id']} | {r['before_best_iou']:.6f} | "
                     f"{r['after_best_iou']:.6f} | {r['delta_iou']:+.6f} |")
lines += [
    "",
    "## 归属与结构诊断",
    "",
    "| 已观测可评价 GT 表面 | 严格基线 | E1 | partial |",
    "|---|---:|---:|---:|",
]
for label, k in [("正确归属顶点", "correct_owner_vertices"), ("错误归属顶点", "wrong_owner_vertices"),
                 ("有几何但未分配顶点", "unassigned_with_geometry_target_vertices"),
                 ("无几何顶点", "no_geometry_target_vertices")]:
    lines.append(f"| {label} | " + " | ".join(str(summary[m]["owner_surface"][k]) for m in methods) + " |")
lines += [
    "",
    f"主分区结构诊断（main_partition_intersection）：Split GT 从 {b['main_structure']['split_gt_count']} "
    f"变为 {e['main_structure']['split_gt_count']}，Merge 预测从 {b['main_structure']['merge_prediction_count']} "
    f"变为 {e['main_structure']['merge_prediction_count']}；partial 的对应计数为 "
    f"{p['main_structure']['split_gt_count']}、{p['main_structure']['merge_prediction_count']}。"
    f"辅助几何诊断另报告 Split GT {b['auxiliary_structure']['split_gt_count']}→"
    f"{e['auxiliary_structure']['split_gt_count']}，Merge 预测 {b['auxiliary_structure']['merge_prediction_count']}→"
    f"{e['auxiliary_structure']['merge_prediction_count']}。两套诊断范围不同，不能混用。",
    "",
    "这些计数支持“预测碎片增多、整体归属收益不足”的现象描述；尚未按种子、追踪段或撤票阶段做因果消融，"
    "不能据此认定某一具体模块是唯一原因。正确归属的对齐是评分专用的一对一最优匹配，不反馈到算法。",
    "",
    "## partial 与 E1 的实际差异",
    "",
    f"AP、AP50、F1 的差值分别为 "
    f"{partial_vs_e1['metric_delta_percentage_points']['AP']:+.8f}、"
    f"{partial_vs_e1['metric_delta_percentage_points']['AP50']:+.8f}、"
    f"{partial_vs_e1['metric_delta_percentage_points']['F1']:+.8f} 个百分点；"
    f"PQ 差值 {partial_vs_e1['metric_delta_percentage_points']['PQ']:+.8f}，"
    f"mCov 差值 {partial_vs_e1['metric_delta_percentage_points']['mCov']:+.8f} 个百分点。"
    f"可评价 GT 上正确归属顶点增加 {partial_vs_e1['owner_delta_vertices']['correct_owner_vertices']}，"
    f"错误归属变化 {partial_vs_e1['owner_delta_vertices']['wrong_owner_vertices']}。",
    "",
    f"按 PLY 原始 surface_index 逐点核对整个固定 TSDF，partial 相对 E1 有 "
    f"{pd['label_changed_vertices']} 点最终实例标签变化"
    f"（占 {pd['label_changed_fraction_of_full_surface']*100:.6f}%），"
    f"其中 {pd['unassigned_to_assigned_vertices']} 点由未分配变为已分配、"
    f"{pd['assigned_to_unassigned_vertices']} 点反向变化、"
    f"{pd['different_positive_id_vertices']} 点在不同正实例 ID 间切换；"
    f"原生证据状态变化 {pd['native_state_changed_vertices']} 点。"
    f"全 TSDF 未分配点净减少 {-partial_vs_e1['count_delta']['gray_total']}，不等于 GT 上纠正了同样多的点。"
    "PLY 的 state 来自原生证据，instance_id 来自最终后处理；不能把两者当作同一阶段。",
    "",
    "| raw GT（仅列 partial 与 E1 不同者） | E1 最佳 IoU | partial 最佳 IoU | ΔIoU |",
    "|---|---:|---:|---:|",
]
for r in partial_per_gt:
    if r["delta_iou"]:
        lines.append(f"| {r['raw_gt_id']} | {r['E1_best_iou']:.8f} | {r['partial_best_iou']:.8f} | {r['delta_iou']:+.8f} |")
lines += [
    "",
    "partial 的冻结策略只把旧观测计划的 whole 撤票改为现有 partial 撤票，复用同批种子、SAM2 追踪、ID、门控、"
    "可信区域、TSDF 和后处理。原 E1 重放、严格一票、全量/增量一致、精确回滚、范围外不变等已记录检查全部为 true。"
    "这些检查证明执行约束一致，不代表指标必然改善。",
    "",
    "## 执行与复核",
    "",
    f"自动选中 {result['audit']['automatic_seed_count']} 个种子，实际执行 {result['audit']['executed_seed_count']} 个。"
    "零有效正向投影按用户确认同步跳过并留痕，E1 与 partial 共用执行种子。",
]
for r in result["audit"]["zero_projection_skipped"]:
    lines.append(f"跳过 {r['group_id']} / {r['mask_id']} / 原始帧 {r['frame']}：{r['reason']}；"
                 f"有效深度像素 {r['valid_depth_pixels']}。")
lines += [
    "",
    "最终指标与原生指标均保留在原始 comparison.json；本报告主表统一使用 final。"
    "历史 P1-A1 使用反向投影投票，本轮严格基线使用 E1 正向投影及同帧冲突弃票，"
    "旧新差值包含基线实现变化，不能作为本轮修复收益。历史 MaskClustering 也不属于这组三方法的固定几何修复对照。",
    "",
    f"所有分析输入与本地 delivery_manifest.json 的 SHA-256 一致。结构化 JSON 包含完整 {comparison['GT']} GT 数据、"
    "各阈值 TP/FP/FN、指标差值、点级差异索引、原始输入哈希及脚本哈希；"
    "表中的 raw GT ID 不臆造语义名称。原审计键名 all_400_frames 沿用旧名称，实际帧数以本报告固定帧列表为准。",
    "",
    "建议同时归档：evaluation_r2 中的原始 metrics/paired/adapter_manifest、protocol_freeze、"
    "confirmed_selection、partial/policy 与 predictions_freeze、三个方法的实例/未分配 PLY、"
    "本分析 JSON 及复核脚本。实现差异应与已有 E1/auto-v3 源码逐项对照，只上传实际不同的适配、配置和来源记录。",
    "",
    "本报告不推断其他场景表现，不把“局部最佳 IoU 改善”解释为整体算法有效，也不据此修改参数。",
]
(out / f"analysis_{scene}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(dict(status="PASS", scene=scene, verified_inputs=len(input_hashes),
                      output_files=[str(out / f"analysis_{scene}.{ext}") for ext in ["md", "json"]],
                      partial_label_changes=pd["label_changed_vertices"],
                      partial_state_changes=pd["native_state_changed_vertices"]), ensure_ascii=False))
