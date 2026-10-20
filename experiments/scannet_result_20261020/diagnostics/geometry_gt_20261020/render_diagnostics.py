"""Scientific diagnostic plots, identical axes; no map/label edits."""
from pathlib import Path
import json
import colorsys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
for scene in ["scene0011_00", "scene0608_00"]:
    folder = ROOT / scene
    with np.load(folder / "geometry_samples.npz") as z:
        xyz = z["xyz"]
        gt = z["gt_xyz"]
        dist = z["distance_m"]
        target = z["gt_target"]
        labels = z["swap_partial"]
        gt_raw = z["gt_raw_ids"]
        color = z["gt_rgb"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    # A common oblique orthographic projection in meter coordinates.
    basis = np.array([[.75, -.66, 0], [.66 * .7, .75 * .7, .714]])
    pp = xyz @ basis.T
    gp = gt @ basis.T
    positive = labels > 0
    colors = np.array([colorsys.hsv_to_rgb((int(i) * .61803398875) % 1, .67, .94) for i in labels[positive]])
    ax[0].scatter(gp[target, 0], gp[target, 1], c="#346cab", s=.12, rasterized=True)
    ax[0].set_title("Delivered GT: evaluation objects only")
    ax[1].scatter(pp[positive, 0], pp[positive, 1], c=colors, s=.16, rasterized=True)
    ax[1].set_title("Delivered partial: all published instances")
    ax[2].scatter(gp[:, 0], gp[:, 1], c="#c5cbd2", s=.12, rasterized=True)
    near = positive & (dist <= .03)
    far = positive & (dist > .1)
    ax[2].scatter(pp[near, 0], pp[near, 1], c="#0f968f", s=.15, label="Map <=3 cm from full GT")
    ax[2].scatter(pp[far, 0], pp[far, 1], c="#df5a45", s=.4, label="Map >10 cm from full GT")
    ax[2].set_title("Same coordinates, no alignment applied")
    ax[2].legend(loc="lower left", markerscale=5, fontsize=8)
    bounds = np.vstack([gp, pp[positive]])
    lo, hi = np.quantile(bounds, [.001, .999], axis=0)
    for a in ax:
        a.set_aspect("equal")
        a.set_xlim(lo[0] - .2, hi[0] + .2)
        a.set_ylim(lo[1] - .2, hi[1] + .2)
        a.set_xlabel("Common projection axis (m)")
        a.set_ylabel("Common projection axis (m)")
        a.grid(alpha=.15)
    fig.suptitle(scene + " | Same camera, scale and world coordinates", fontsize=15)
    fig.savefig(folder / "same_coordinates_comparison.png", dpi=170)
    plt.close(fig)
    frames = [520, 2010, 1190] if scene == "scene0011_00" else [0, 1420, 3380]
    fig, ax = plt.subplots(len(frames), 3, figsize=(14, 10), constrained_layout=True)
    for row, fid in enumerate(frames):
        with np.load(folder / f"frame{fid}_depth.npz") as z:
            d, rd, cls, names = (z[k] for k in ["depth", "reference_depth", "reference_class", "class_names"])
        rgb = np.asarray(Image.open(folder / f"frame{fid}_rgb.jpg"))
        ax[row, 0].imshow(rgb)
        ax[row, 0].set_title(f"Frame {fid}: calibrated RGB")
        dm = np.ma.masked_where(d == 0, d)
        cm = plt.get_cmap("viridis").copy()
        cm.set_bad("#dcdcdc")
        im = ax[row, 1].imshow(dm, vmin=0, vmax=4, cmap=cm)
        ax[row, 1].set_title("Raw depth (m); gray = missing")
        residual = d - rd
        good = np.isfinite(rd) & (d > 0) & (d < 4) & (rd < 4)
        res = np.ma.masked_where(~good, residual * 1000)
        cm2 = plt.get_cmap("coolwarm").copy()
        cm2.set_bad("#dcdcdc")
        im2 = ax[row, 2].imshow(res, vmin=-100, vmax=100, cmap=cm2)
        ax[row, 2].set_title("Depth minus full GT ray depth (mm)")
        for a in ax[row]:
            a.axis("off")
    fig.colorbar(im, ax=ax[:, 1], shrink=.6)
    fig.colorbar(im2, ax=ax[:, 2], shrink=.6)
    fig.suptitle(scene + " | Measured depth before instance repair; GT only for diagnosis", fontsize=14)
    fig.savefig(folder / "raw_depth_examples.png", dpi=150)
    plt.close(fig)
print("PLOTS_READY")
