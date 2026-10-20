"""Diagnostic window apertures, including holes absent from cleaned GT mesh."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from pathlib import Path
from collections import defaultdict
import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

C = Path("/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020")
R = Path("/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020")
OUT = Path("/data/chenkejun/CVPR/results/scannet_geometry_gt_diagnosis_20261020")


def make_plane(pts):
    center = np.median(pts, axis=0)
    _, _, V = np.linalg.svd(pts - center, full_matrices=False)
    # Fit all reference window vertices; disclose thickness/residual.
    uv = (pts - center) @ V[:2].T
    hull = ConvexHull(uv)
    return dict(center=center, basis=V, equations=hull.equations,
                residual_mm=(1000 * np.quantile(np.abs((pts - center) @ V[2]), [.5, .9, .99])).tolist(),
                vertices=len(pts))


def plane_intersection(origin, dr, p):
    denom = dr @ p["basis"][2]
    t = ((p["center"] - origin) @ p["basis"][2]) / np.where(np.abs(denom) > 1e-8, denom, np.nan)
    xyz = origin + dr * t[..., None]
    local = (xyz - p["center"]) @ p["basis"][:2].T
    margin = -(local @ p["equations"][:, :2].T + p["equations"][:, 2]).max(axis=-1)
    return t, (t > .2) & (t < 4) & (margin > .025)


for scene in ["scene0011_00", "scene0608_00"]:
    out = OUT / scene
    config = json.loads((C / "configs" / (scene + ".json")).read_text())
    cam = config["camera"]
    raw = Path("/data/chenkejun/datasets/ScanNet/scans") / scene
    processed = Path("/data/chenkejun/CVPR/datasets/MaskClustering/scannet/processed") / scene
    mesh = o3d.io.read_triangle_mesh(str(raw / (scene + "_vh_clean_2.ply")))
    caster = o3d.t.geometry.RaycastingScene(nthreads=4)
    caster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    with np.load(Path("/data/chenkejun/CVPR/evaluation_results/scannet_v3_r2_reeval_20261009") / scene / "ground_truth/gt.npz") as z:
        ref, ids = z["xyz_ref"], z["raw_instance_id"]
    objects = json.loads((raw / (scene + ".aggregation.json")).read_text())["segGroups"]
    planes = {x["objectId"] + 1: make_plane(ref[ids == x["objectId"] + 1]) for x in objects if x["label"] == "window"}
    with np.load(out / "ghost_source_trace.npz") as z:
        ghost = z["xyz"]
        original_support = z["support"]
    gs = np.zeros(len(ghost), np.uint16)
    gw = gs.copy()
    gb = gs.copy()
    behindplane_examples = []
    poses = np.loadtxt(R / "inputs" / scene / "traj.txt").reshape(-1, 4, 4)
    v, u = np.mgrid[0:480:4, 0:640:4]
    rays = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], axis=-1)
    rows, accum = [], defaultdict(list)
    best = None
    for fid in config["frame_ids"]:
        pose = poses[fid]
        d_full = cv2.imread(str(processed / "depth" / (str(fid) + ".png")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        d = d_full[::4, ::4]
        dr = rays @ pose[:3, :3].T
        ro = np.broadcast_to(pose[:3, 3], dr.shape)
        rd = caster.cast_rays(o3d.core.Tensor(np.concatenate([ro, dr], axis=-1).astype(np.float32)), nthreads=4)["t_hit"].numpy()
        total_window_mask = np.zeros(d.shape, bool)
        near_plane = np.full(d.shape, np.inf, np.float32)
        for pid, p in planes.items():
            t, in_hull = plane_intersection(pose[:3, 3], dr, p)
            visible = in_hull & (rd >= t - .05)
            mask = cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if not mask.any():
                continue
            total_window_mask |= mask
            near_plane[mask] = np.minimum(near_plane[mask], t[mask])
            valid = mask & (d > 0)
            accepted = valid & (d < 4)
            r = dict(frame=fid, raw_instance_id=int(pid), pixels=int(mask.sum()),
                     missing=int((mask & (d == 0)).sum()), over4=int((mask & (d >= 4)).sum()),
                     accepted=int(accepted.sum()), accepted_behind_10cm=int((accepted & (d > t + .1)).sum()),
                     accepted_front_10cm=int((accepted & (d < t - .1)).sum()),
                     raw_behind_10cm=int((valid & (d > t + .1)).sum()),
                     reference_has_hole=int((mask & ~np.isfinite(rd)).sum()))
            rows.append(r)
            accum["accepted_residual"].append((d - t)[accepted])
            if r["pixels"] >= 100 and (best is None or r["raw_behind_10cm"] + r["missing"] > best[0]):
                best = (r["raw_behind_10cm"] + r["missing"], fid)
        # Fixed-map points have to be supported by a valid measured depth first.
        xyz_c = (ghost - pose[:3, 3]) @ pose[:3, :3]
        z = xyz_c[:, 2]
        uv = np.column_stack([cam["fx"] * xyz_c[:, 0] / np.maximum(z, 1e-9) + cam["cx"],
                              cam["fy"] * xyz_c[:, 1] / np.maximum(z, 1e-9) + cam["cy"]])
        uv = np.rint(uv).astype(np.int64)
        valid = (z > .1) & (z < 4) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        gi = np.flatnonzero(valid)
        dep = d_full[uv[gi, 1], uv[gi, 0]]
        obs = np.column_stack([(uv[gi, 0] - cam["cx"]) * dep / cam["fx"],
                               (uv[gi, 1] - cam["cy"]) * dep / cam["fy"], dep])
        support = (dep > 0) & (dep < 4) & (np.linalg.norm(obs - xyz_c[gi], axis=1) < .02)
        gi = gi[support]
        if len(gi):
            gs[gi] += 1
            drg = (xyz_c[gi] / z[gi, None]) @ pose[:3, :3].T
            rg = np.column_stack([np.broadcast_to(pose[:3, 3], drg.shape), drg])
            dg = caster.cast_rays(o3d.core.Tensor(rg.astype(np.float32)), nthreads=4)["t_hit"].numpy()
            window = np.zeros(len(gi), bool)
            behind = window.copy()
            for p in planes.values():
                t, inside = plane_intersection(pose[:3, 3], drg, p)
                is_win = inside & (dg >= t - .05)
                window |= is_win
                behind |= is_win & (z[gi] > t + .1)
            gw[gi] += window
            gb[gi] += behind
        if fid % 500 == 0:
            print(scene, "WINDOW", fid, flush=True)
    # Record diagnostic classifications only; no prediction or masks are edited.
    totals = {k: int(sum(r[k] for r in rows)) for k in ["pixels", "missing", "over4", "accepted", "accepted_behind_10cm", "accepted_front_10cm", "raw_behind_10cm", "reference_has_hole"]}
    residual = np.concatenate(accum["accepted_residual"]) if accum["accepted_residual"] else np.array([])
    totals["accepted_residual_mm_p10_p50_p90_p99"] = (np.quantile(residual, [.1, .5, .9, .99]) * 1000).tolist() if len(residual) else []
    result = dict(scene=scene, purpose="diagnostic full window aperture including cleaned-mesh holes",
                  method="PCA plane through GT window vertices, convex hull inset 2.5cm; discard GT occlusion in front by >5cm; image eroded 3x3 at stride4",
                  planes={str(k): {n: v.tolist() if isinstance(v, np.ndarray) else v for n, v in p.items()} for k, p in planes.items()},
                  totals=totals, frame_records=rows, best_frame=best,
                  ghost_trace=dict(over10cm_from_GT=len(ghost), measured_depth_supported=int((gs > 0).sum()),
                                   window_aperture_any=int((gw > 0).sum()), behind_window_any=int((gb > 0).sum()),
                                   window_aperture_majority=int(((gs > 0) & (gw > gs / 2)).sum())),
                  limitations="Window ray returns can reflect glass, transmit to real surfaces beyond glass, or be noisy. These tests do not uniquely identify an optical mechanism.")
    (out / "window_apertures.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / "window_ghost_trace.npz", xyz=ghost, support=gs, window=gw, behind_window=gb)
    if best:
        fid = best[1]
        pose = poses[fid]
        vv, uu = np.mgrid[:480, :640]
        rr = np.stack([(uu - cam["cx"]) / cam["fx"], (vv - cam["cy"]) / cam["fy"], np.ones_like(uu)], axis=-1)
        dr = rr @ pose[:3, :3].T
        ray = np.concatenate([np.broadcast_to(pose[:3, 3], dr.shape), dr], axis=-1).astype(np.float32)
        gt_depth = caster.cast_rays(o3d.core.Tensor(ray), nthreads=4)["t_hit"].numpy()
        roi = np.zeros(gt_depth.shape, bool)
        plane_depth = np.full(gt_depth.shape, np.inf, np.float32)
        for p in planes.values():
            t, inside = plane_intersection(pose[:3, 3], dr, p)
            m = inside & (gt_depth >= t - .05)
            m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            roi |= m
            plane_depth[m] = np.minimum(plane_depth[m], t[m])
        depth = cv2.imread(str(processed / "depth" / f"{fid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        cv2.imwrite(str(out / f"window_aperture_frame{fid}_rgb.jpg"),
                    cv2.imread(str(R / "inputs" / scene / "results" / f"frame{fid:06d}.png")))
        np.savez_compressed(out / "window_aperture_example.npz", frame=fid, roi=roi, depth=depth,
                            window_plane_depth=plane_depth, GT_depth=gt_depth)
    print(scene, "WINDOW_RESULT", json.dumps({k: result[k] for k in ["totals", "ghost_trace", "best_frame"]}), flush=True)
