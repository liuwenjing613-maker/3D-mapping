"""GT-free repeat-view consistency diagnostic; does not write any map input."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from pathlib import Path
import json
import cv2
import numpy as np

C = Path("/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020")
R = Path("/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020")
OUT = Path("/data/chenkejun/CVPR/results/scannet_geometry_gt_diagnosis_20261020")

for scene in ["scene0011_00", "scene0608_00"]:
    config = json.loads((C / "configs" / (scene + ".json")).read_text())
    cam = config["camera"]
    processed = Path("/data/chenkejun/CVPR/datasets/MaskClustering/scannet/processed") / scene
    poses = np.loadtxt(R / "inputs" / scene / "traj.txt").reshape(-1, 4, 4)
    fids = np.asarray(config["frame_ids"])
    translations, directions = poses[fids, :3, 3], poses[fids, :3, 2]
    v, u = np.mgrid[:480, :640]
    rays = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], axis=-1).astype(np.float32)
    cache = {}
    def frame(fid):
        if fid not in cache:
            if len(cache) >= 10:
                cache.pop(next(iter(cache)))
            d = cv2.imread(str(processed / "depth" / f"{fid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            pts = rays * d[..., None]
            n = np.cross(pts[1:-1, 2:] - pts[1:-1, :-2], pts[2:, 1:-1] - pts[:-2, 1:-1])
            norm = np.linalg.norm(n, axis=-1)
            normals = np.zeros_like(pts)
            normals[1:-1, 1:-1] = n / np.maximum(norm[..., None], 1e-12)
            lo = cv2.erode(d, np.ones((5, 5), np.uint8))
            hi = cv2.dilate(d, np.ones((5, 5), np.uint8))
            stable = (d > .3) & (d < 3.5) & (lo > 0) & (hi - lo < .05)
            cache[fid] = (d, pts, normals, stable)
        return cache[fid]
    pairs = set()
    for fid in list(fids[::5]) + ([1190, 1200, 1250, 2010] if scene == "scene0011_00" else [300, 1420]):
        if fid + 10 in fids:
            pairs.add((int(fid), int(fid + 10), "adjacent_10"))
        delta = np.linalg.norm(translations - poses[fid, :3, 3], axis=1)
        cosine = directions @ poses[fid, :3, 2]
        choices = np.flatnonzero((np.abs(fids - fid) >= 300) & (delta < 1.0) & (cosine > .8))
        if len(choices):
            chosen = choices[np.argmin(delta[choices] + .5 * (1 - cosine[choices]))]
            pairs.add((int(fid), int(fids[chosen]), "revisit_ge300"))
    records = []
    pooled = {"adjacent_10": [], "revisit_ge300": []}
    for a, b, kind in sorted(pairs):
        da, pa, na, va = frame(a)
        db, pb, nb, vb = frame(b)
        stride_mask = np.zeros(da.shape, bool)
        stride_mask[::4, ::4] = True
        ai = np.flatnonzero(va & stride_mask)
        p = pa.reshape(-1, 3)[ai]
        n = na.reshape(-1, 3)[ai]
        world = p @ poses[a, :3, :3].T + poses[a, :3, 3]
        xyz_b = (world - poses[b, :3, 3]) @ poses[b, :3, :3]
        n_b = (n @ poses[a, :3, :3].T) @ poses[b, :3, :3]
        z = xyz_b[:, 2]
        uv = np.rint(np.column_stack([cam["fx"] * xyz_b[:, 0] / np.maximum(z, 1e-9) + cam["cx"],
                                      cam["fy"] * xyz_b[:, 1] / np.maximum(z, 1e-9) + cam["cy"]])).astype(int)
        inside = (z > .2) & (z < 4) & (uv[:, 0] >= 1) & (uv[:, 0] < 639) & (uv[:, 1] >= 1) & (uv[:, 1] < 479)
        ii = np.flatnonzero(inside)
        xx, yy = uv[ii, 0], uv[ii, 1]
        target = pb[yy, xx]
        normals = nb[yy, xx]
        # Exclude discontinuities, normal disagreement, and >10cm occlusion/depth differences.
        good = vb[yy, xx] & (np.abs(np.sum(n_b[ii] * normals, axis=-1)) > .94) & (np.abs(z[ii] - db[yy, xx]) < .1)
        delta = xyz_b[ii][good] - target[good]
        signed = np.sum(delta * normals[good], axis=1)
        residual = np.abs(signed)
        if len(residual) < 200:
            continue
        pooled[kind].append(residual)
        records.append(dict(source=a, target=b, kind=kind, n=len(residual),
                            abs_point_plane_mm_p50_p90_p99=(1000 * np.quantile(residual, [.5, .9, .99])).tolist(),
                            over15mm=float(np.mean(residual > .015)),
                            over30mm=float(np.mean(residual > .03))))
    summary = {}
    for kind, parts in pooled.items():
        x = np.concatenate(parts) if parts else np.array([])
        summary[kind] = dict(pairs=sum(r["kind"] == kind for r in records), n=len(x),
                             abs_point_plane_mm_p50_p90_p99=(1000 * np.quantile(x, [.5, .9, .99])).tolist() if len(x) else [],
                             over15mm=float(np.mean(x > .015)) if len(x) else None,
                             over30mm=float(np.mean(x > .03)) if len(x) else None)
    result = dict(scene=scene, GT_used=False, summary=summary, records=records,
                  method="4px source sample; both frames valid 0.3-3.5m and 5x5 depth range<5cm; normal cosine>.94; projected z difference<10cm; point-to-plane residual",
                  limitations="Excludes large occlusions/discontinuities; repeated pixel-frame samples are not independent. Residual reflects combined sensor depth and supplied pose consistency.")
    (OUT / scene / "frame_pair_consistency.json").write_text(json.dumps(result, indent=2) + "\n")
    print(scene, "PAIR_SUMMARY", json.dumps(summary), flush=True)
    print(scene, "PAIR_WORST", json.dumps(sorted(records, key=lambda x: -x["over15mm"])[:4]), flush=True)
