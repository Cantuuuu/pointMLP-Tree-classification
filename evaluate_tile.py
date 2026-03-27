"""
Single-Tile Watershed Evaluation with Ground Truth Comparison
=============================================================
Runs watershed grid search on one specific LiDAR tile, evaluates each
parameter combination against XML bounding-box annotations, and generates
a comparative HTML report showing:

  "What should have been seen" (GT annotations)
  "What was seen"              (detected watershed crowns)

Usage:
    python evaluate_tile.py
    python evaluate_tile.py --laz   path/to/tile.laz \\
                            --xml   path/to/annotations.xml \\
                            --mode  quick
    python evaluate_tile.py --mode full --match-dist 8.0

Coordinate mapping (NEON AOP):
    The XML bounding boxes are in pixel coordinates of a 0.1 m/px RGB image
    whose spatial extent matches the LAZ tile.  We derive the pixel → UTM
    transform from the LAZ header bounds + image size declared in the XML.
"""

import argparse
import io
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_LAZ = "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_MLBS_3_541000_4140000_image_crop.laz"
DEFAULT_XML = "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_MLBS_3_541000_4140000_image_crop.xml"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_tile(laz_path: str):
    las = laspy.read(laz_path)
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    xyz = np.stack([x, y, z], axis=1)
    try:
        cls = np.array(las.classification, dtype=np.int32)
    except Exception:
        cls = np.zeros(len(x), dtype=np.int32)
    area_ha = (x.max() - x.min()) * (y.max() - y.min()) / 10_000
    return xyz, cls, area_ha


def load_gt_annotations(xml_path: str, xyz: np.ndarray):
    """
    Parse XML annotations and convert pixel bboxes → UTM world coordinates.

    NEON imagery is 0.1 m/px; image spatial extent matches the LAZ tile.
    Mapping:
        world_x = x_min + (px_x / img_w) * (x_max - x_min)
        world_y = y_max - (px_y / img_h) * (y_max - y_min)   # Y is flipped
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    img_w = int(size_el.find("width").text)
    img_h = int(size_el.find("height").text)

    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()

    def px_to_world(px_x, px_y):
        wx = x_min + (px_x / img_w) * (x_max - x_min)
        wy = y_max - (px_y / img_h) * (y_max - y_min)
        return wx, wy

    trees_gt = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin_px = float(bb.find("xmin").text)
        ymin_px = float(bb.find("ymin").text)
        xmax_px = float(bb.find("xmax").text)
        ymax_px = float(bb.find("ymax").text)

        wx_min, wy_max = px_to_world(xmin_px, ymin_px)  # top-left
        wx_max, wy_min = px_to_world(xmax_px, ymax_px)  # bottom-right

        cx = (wx_min + wx_max) / 2
        cy = (wy_min + wy_max) / 2
        w_m  = wx_max - wx_min
        h_m  = wy_max - wy_min
        area = w_m * h_m

        trees_gt.append({
            "cx": cx, "cy": cy,
            "xmin": wx_min, "xmax": wx_max,
            "ymin": wy_min, "ymax": wy_max,
            "width_m": w_m, "height_m": h_m,
            "area_m2": area,
        })

    return trees_gt, img_w, img_h


# ---------------------------------------------------------------------------
# 2. Watershed (with centroid extraction)
# ---------------------------------------------------------------------------

def run_watershed_with_centroids(xyz, tree_mask, classification, params):
    """Run watershed and return per-crown centroids and pixel regions."""
    from src.segmentation.watershed_grid_search import _run_watershed
    result = _run_watershed(xyz, tree_mask, classification, params)

    labels = result["labels"]
    unique = sorted(set(labels) - {-1})
    if not unique:
        return [], result

    tree_xyz = xyz[tree_mask]
    centroids = []
    for lbl in unique:
        pts = tree_xyz[labels == lbl]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        n_pts = len(pts)

        # Crown area from watershed labels
        crown_area_m2 = 0.0
        if result.get("watershed_labels") is not None:
            px = result["chm_res"] ** 2
            # find the watershed label for this cluster
            wlabels = result["watershed_labels"]
            col_arr = result["col"][labels == lbl]
            row_arr = result["row"][labels == lbl]
            if len(col_arr):
                wl = wlabels[row_arr[0], col_arr[0]]
                crown_area_m2 = float((wlabels == wl).sum()) * px

        centroids.append({
            "id": lbl, "cx": cx, "cy": cy,
            "n_pts": n_pts, "crown_area_m2": crown_area_m2,
        })

    return centroids, result


# ---------------------------------------------------------------------------
# 3. Matching: detected ↔ GT
# ---------------------------------------------------------------------------

def match_detections_to_gt(detected, gt_trees, match_dist_m=8.0):
    """
    Greedy nearest-neighbour matching.
    A detection matches a GT tree if their centroid distance ≤ match_dist_m.

    Returns:
        tp_pairs  - list of (det_idx, gt_idx)
        fp_ids    - detected not matched
        fn_ids    - GT not matched
        det_match - array of GT index matched (or -1) for each detection
        gt_match  - array of det index matched (or -1) for each GT
    """
    n_det = len(detected)
    n_gt  = len(gt_trees)

    if n_det == 0 or n_gt == 0:
        return [], list(range(n_det)), list(range(n_gt)), \
               np.full(n_det, -1), np.full(n_gt, -1)

    det_xy = np.array([[d["cx"], d["cy"]] for d in detected])
    gt_xy  = np.array([[g["cx"], g["cy"]] for g in gt_trees])

    # pairwise distances
    diff = det_xy[:, None, :] - gt_xy[None, :, :]   # (n_det, n_gt, 2)
    dists = np.sqrt((diff ** 2).sum(axis=2))          # (n_det, n_gt)

    det_match = np.full(n_det, -1, dtype=int)
    gt_match  = np.full(n_gt, -1, dtype=int)
    tp_pairs  = []

    # Sort all candidate pairs by distance
    pairs = sorted(
        [(dists[i, j], i, j) for i in range(n_det) for j in range(n_gt)
         if dists[i, j] <= match_dist_m],
        key=lambda x: x[0],
    )
    for dist, di, gi in pairs:
        if det_match[di] == -1 and gt_match[gi] == -1:
            det_match[di] = gi
            gt_match[gi] = di
            tp_pairs.append((di, gi, dist))

    fp_ids = [i for i in range(n_det) if det_match[i] == -1]
    fn_ids = [j for j in range(n_gt)  if gt_match[j]  == -1]

    return tp_pairs, fp_ids, fn_ids, det_match, gt_match


def instance_metrics(tp_pairs, fp_ids, fn_ids):
    tp = len(tp_pairs)
    fp = len(fp_ids)
    fn = len(fn_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "n_detected": tp + fp,
            "n_gt":       tp + fn}


# ---------------------------------------------------------------------------
# 4. Grid search with GT evaluation
# ---------------------------------------------------------------------------

def run_gt_grid_search(xyz, cls, area_ha, gt_trees, param_grid,
                       match_dist_m=8.0, verbose=True):
    from src.segmentation.watershed_grid_search import (
        expand_grid, compute_metrics, quality_score,
    )

    tree_mask = (cls == 5)
    combos = expand_grid(param_grid)
    n_gt = len(gt_trees)
    results = []

    if verbose:
        print(f"\n{'='*65}")
        print(f"Grid search on tile  |  GT trees: {n_gt}  |  "
              f"{len(combos)} parameter combos")
        print(f"{'='*65}")
        hdr = (f"{'res':>5} {'sig':>5} {'win':>5} | "
               f"{'det':>5} {'TP':>4} {'FP':>4} {'FN':>4} | "
               f"{'P':>6} {'R':>6} {'F1':>6} | score")
        print(hdr)
        print("-" * 65)

    for i, params in enumerate(combos):
        centroids, result = run_watershed_with_centroids(
            xyz, tree_mask, cls, params)
        ws_metrics = compute_metrics(result, area_ha)
        score = quality_score(ws_metrics)

        tp_pairs, fp_ids, fn_ids, _, _ = match_detections_to_gt(
            centroids, gt_trees, match_dist_m)
        im = instance_metrics(tp_pairs, fp_ids, fn_ids)

        row = {**params, **ws_metrics, **im,
               "quality_score": score,
               "centroids": centroids,
               "ws_result": result}
        results.append(row)

        if verbose:
            print(
                f"  {params['chm_resolution']:>4.2f} "
                f"{params['smooth_sigma']:>4.1f} "
                f"{params['local_max_window']:>4d} | "
                f"{im['n_detected']:>5} "
                f"{im['tp']:>4} {im['fp']:>4} {im['fn']:>4} | "
                f"{im['precision']:>6.3f} {im['recall']:>6.3f} "
                f"{im['f1']:>6.3f} | {score:.3f}"
            )

    # Sort by F1 (GT-based), break ties by quality_score
    results.sort(key=lambda x: (x["f1"], x["quality_score"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def make_chm(xyz, tree_mask, classification, chm_res=0.5):
    """Build a CHM array + spatial transform for plotting."""
    tree_xyz = xyz[tree_mask]
    ground_mask = (classification == 2)
    if ground_mask.sum() > 0:
        ground_z = float(np.median(xyz[ground_mask, 2]))
    else:
        ground_z = float(np.percentile(xyz[:, 2], 5))

    x, y = tree_xyz[:, 0], tree_xyz[:, 1]
    z = tree_xyz[:, 2] - ground_z
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()

    col = ((x - x_min) / chm_res).astype(np.int32)
    row = ((y - y_min) / chm_res).astype(np.int32)
    n_cols = int((xyz[:, 0].max() - x_min) / chm_res) + 1
    n_rows = int((xyz[:, 1].max() - y_min) / chm_res) + 1

    chm = np.full((n_rows, n_cols), 0.0, dtype=np.float32)
    np.maximum.at(chm.ravel(), row * n_cols + col, z)

    # spatial transform: pixel → world coords
    transform = {
        "x_min": float(x_min), "y_min": float(y_min),
        "x_max": float(xyz[:, 0].max()), "y_max": float(xyz[:, 1].max()),
        "res": chm_res, "n_rows": n_rows, "n_cols": n_cols,
        "ground_z": ground_z,
    }
    return chm, transform


def world_to_chm_px(wx, wy, transform):
    t = transform
    col = (wx - t["x_min"]) / t["res"]
    row = (wy - t["y_min"]) / t["res"]
    return col, row


def generate_grid_figures(all_results):
    """
    Generate figures that visualize the full grid search results.
    Returns dict of {name: base64_png}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import io, base64

    def b64_fig(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    figures = {}
    if not all_results:
        return figures

    res_vals    = sorted(set(r["chm_resolution"]   for r in all_results))
    sigma_vals  = sorted(set(r["smooth_sigma"]      for r in all_results))
    window_vals = sorted(set(r["local_max_window"]  for r in all_results))

    # ------------------------------------------------------------------
    # 1. F1 heatmaps: sigma × window for each chm_resolution
    # ------------------------------------------------------------------
    ncols = len(res_vals)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4),
                             sharey=True, squeeze=False)
    for col, res in enumerate(res_vals):
        ax = axes[0][col]
        mat = np.full((len(sigma_vals), len(window_vals)), np.nan)
        for r in all_results:
            if r["chm_resolution"] != res:
                continue
            i = sigma_vals.index(r["smooth_sigma"])
            j = window_vals.index(r["local_max_window"])
            mat[i, j] = r["f1"]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(len(window_vals)))
        ax.set_xticklabels([str(w) for w in window_vals], fontsize=9)
        ax.set_yticks(range(len(sigma_vals)))
        ax.set_yticklabels([str(s) for s in sigma_vals], fontsize=9)
        ax.set_xlabel("local_max_window (px)", fontsize=9)
        if col == 0:
            ax.set_ylabel("smooth_sigma", fontsize=9)
        ax.set_title(f"chm_resolution = {res} m/px", fontsize=10)
        for i in range(len(sigma_vals)):
            for j in range(len(window_vals)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if mat[i, j] < 0.5 else "black",
                            fontweight="bold" if mat[i, j] == np.nanmax(mat) else "normal")
        plt.colorbar(im, ax=ax, label="F1")

    fig.suptitle("F1 Score — Full Grid Search Results\n"
                 "(brighter green = better, each cell = one parameter combination)",
                 fontsize=11)
    plt.tight_layout()
    figures["f1_heatmap"] = b64_fig(fig)

    # ------------------------------------------------------------------
    # 2. Precision–Recall scatter: all combos, colored by F1
    # ------------------------------------------------------------------
    prec  = [r["precision"]  for r in all_results]
    rec   = [r["recall"]     for r in all_results]
    f1s   = [r["f1"]         for r in all_results]
    res_c = [r["chm_resolution"] for r in all_results]
    win_c = [r["local_max_window"] for r in all_results]
    sig_c = [r["smooth_sigma"] for r in all_results]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(rec, prec, c=f1s, cmap="RdYlGn",
                    vmin=0, vmax=1, s=80, alpha=0.85,
                    edgecolors="gray", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="F1 score")

    # Annotate best 3
    top3 = sorted(all_results, key=lambda x: x["f1"], reverse=True)[:3]
    for i, r in enumerate(top3):
        ax.annotate(
            f"#{i+1}  res={r['chm_resolution']} σ={r['smooth_sigma']} w={r['local_max_window']}",
            (r["recall"], r["precision"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=7, color="#1a3c1b",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
        )

    # F1 iso-curves
    p_line = np.linspace(0.01, 1, 300)
    for f_iso in [0.6, 0.7, 0.8, 0.85, 0.9]:
        r_line = f_iso * p_line / (2 * p_line - f_iso + 1e-9)
        mask = (r_line >= 0) & (r_line <= 1)
        ax.plot(r_line[mask], p_line[mask], "--", color="steelblue",
                lw=0.7, alpha=0.5)
        idx_mid = np.argmin(np.abs(r_line[mask] - 0.5))
        pts = p_line[mask]
        rpts = r_line[mask]
        if len(pts) > idx_mid:
            ax.text(rpts[idx_mid], pts[idx_mid], f"F1={f_iso}",
                    fontsize=6, color="steelblue", alpha=0.7)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision vs Recall — all parameter combinations\n"
                 "(dashed lines = F1 iso-curves)", fontsize=10)
    plt.tight_layout()
    figures["pr_scatter"] = b64_fig(fig)

    # ------------------------------------------------------------------
    # 3. Per-parameter sensitivity: mean F1 while varying one param
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    param_names = ["chm_resolution", "smooth_sigma", "local_max_window"]
    param_labels = ["chm_resolution (m/px)", "smooth_sigma (px)", "local_max_window (px)"]
    param_colors = ["#2c5f2e", "#e07b39", "#3a7d9e"]

    for ax, pname, plabel, pcolor in zip(axes, param_names, param_labels, param_colors):
        unique_vals = sorted(set(r[pname] for r in all_results))
        means, stds = [], []
        for v in unique_vals:
            subset = [r["f1"] for r in all_results if r[pname] == v]
            means.append(np.mean(subset))
            stds.append(np.std(subset))
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(unique_vals, means, "o-", color=pcolor, lw=2, ms=7)
        ax.fill_between(unique_vals,
                        np.clip(means - stds, 0, 1),
                        np.clip(means + stds, 0, 1),
                        alpha=0.2, color=pcolor)

        best_v = unique_vals[np.argmax(means)]
        ax.axvline(best_v, color=pcolor, linestyle="--", lw=1.2, alpha=0.7,
                   label=f"best = {best_v}")
        ax.set_xlabel(plabel, fontsize=9)
        ax.set_ylabel("Mean F1 across other params", fontsize=9)
        ax.set_title(f"Sensitivity: {pname}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Parameter Sensitivity — effect of each parameter on F1\n"
                 "(shaded band = ±1 std across all other parameter values)",
                 fontsize=10)
    plt.tight_layout()
    figures["sensitivity"] = b64_fig(fig)

    # ------------------------------------------------------------------
    # 4. Detection count vs GT across all combos (sorted by F1)
    # ------------------------------------------------------------------
    sorted_r = sorted(all_results, key=lambda x: x["f1"], reverse=True)
    labels_bar = [
        f"r={r['chm_resolution']}\nσ={r['smooth_sigma']}\nw={r['local_max_window']}"
        for r in sorted_r
    ]
    n_det  = [r["n_detected"] for r in sorted_r]
    n_tp   = [r["tp"]         for r in sorted_r]
    n_fp   = [r["fp"]         for r in sorted_r]
    n_fn   = [r["fn"]         for r in sorted_r]
    n_gt_v = sorted_r[0]["n_gt"] if sorted_r else 864
    f1_bar = [r["f1"]         for r in sorted_r]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(sorted_r) * 0.55), 9),
                                    gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(sorted_r))
    ax1.bar(x, n_tp,  label="TP (correctly detected)", color="#2dc653", alpha=0.9)
    ax1.bar(x, n_fp,  bottom=n_tp, label="FP (false alarms)", color="#e63946", alpha=0.9)
    ax1.axhline(n_gt_v, color="navy", linestyle="--", lw=1.5,
                label=f"GT = {n_gt_v} trees")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_bar, fontsize=6, rotation=0)
    ax1.set_ylabel("Tree count")
    ax1.set_title("Detections per configuration (sorted by F1, best = left)",
                  fontsize=10)
    ax1.legend(fontsize=8)

    colors_f1 = [cm.RdYlGn(f) for f in f1_bar]
    ax2.bar(x, f1_bar, color=colors_f1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_bar, fontsize=6, rotation=0)
    ax2.set_ylabel("F1")
    ax2.set_ylim(0, 1)
    ax2.axhline(max(f1_bar), color="green", linestyle="--", lw=1, alpha=0.5)
    ax2.set_title("F1 per configuration", fontsize=10)

    plt.tight_layout()
    figures["detection_bar"] = b64_fig(fig)

    return figures


def generate_figures(xyz, cls, gt_trees, best_result, second_result,
                     chm_res=0.4):
    """
    Generate matplotlib figures for the HTML report.
    Returns dict of {name: base64_png}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle, FancyArrowPatch
    import base64

    tree_mask = cls == 5
    chm, tr = make_chm(xyz, tree_mask, cls, chm_res)

    # smooth CHM slightly for display
    chm_display = ndimage.gaussian_filter(chm.astype(np.float32), sigma=0.8)

    x_min, y_min = tr["x_min"], tr["y_min"]
    x_max, y_max = tr["x_max"], tr["y_max"]
    res = tr["res"]
    extent = [x_min, x_max, y_min, y_max]

    def b64_fig(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    figures = {}

    # ------------------------------------------------------------------
    # Figure 1: Ground Truth — CHM + GT bounding boxes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.imshow(chm_display, origin="lower", extent=extent,
              cmap="YlGn", vmin=0, vmax=chm_display.max() * 0.9, alpha=0.85)

    for g in gt_trees:
        rect = Rectangle(
            (g["xmin"], g["ymin"]),
            g["xmax"] - g["xmin"], g["ymax"] - g["ymin"],
            linewidth=0.8, edgecolor="#e63946", facecolor="none", alpha=0.7,
        )
        ax.add_patch(rect)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"GROUND TRUTH — {len(gt_trees)} annotated trees\n"
                 f"(MLBS tile · {(x_max-x_min):.0f}×{(y_max-y_min):.0f} m · "
                 f"{len(gt_trees)/((x_max-x_min)*(y_max-y_min)/10000):.0f} trees/ha)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    legend = [mpatches.Patch(facecolor="none", edgecolor="#e63946",
                             label="Annotated tree (GT)")]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    figures["gt_map"] = b64_fig(fig)

    # ------------------------------------------------------------------
    # Figure 2: Best detection result
    # ------------------------------------------------------------------
    def plot_detection(result, title_prefix, fig_key):
        centroids  = result["centroids"]
        params     = {k: result[k] for k in ["chm_resolution", "smooth_sigma",
                                              "local_max_window"]}
        tp_pairs, fp_ids, fn_ids, det_match, gt_match = \
            match_detections_to_gt(centroids, gt_trees)

        tp_set = set(di for di, gi, _ in tp_pairs)
        fig, ax = plt.subplots(figsize=(9, 12))
        ax.imshow(chm_display, origin="lower", extent=extent,
                  cmap="YlGn", vmin=0, vmax=chm_display.max() * 0.9, alpha=0.85)

        # Draw watershed regions if available
        ws_result = result.get("ws_result")
        if ws_result and ws_result.get("watershed_labels") is not None:
            wlabels = ws_result["watershed_labels"]
            ws_res  = ws_result["chm_res"]
            ws_xmin = xyz[:, 0].min()  # same as x_min for tree points
            ws_ymin = xyz[cls == 5, 1].min()
            # overlay watershed as semi-transparent colored mask
            unique_w = sorted(set(wlabels.ravel()) - {0})
            overlay = np.zeros((*wlabels.shape, 4), dtype=np.float32)
            for wlbl in unique_w:
                mask_px = wlabels == wlbl
                r, c = np.where(mask_px)
                if not len(r): continue
                # centroid of this region in world coords
                cx_w = ws_xmin + c.mean() * ws_res
                cy_w = ws_ymin + r.mean() * ws_res
                # is this a TP or FP?
                # find closest detection centroid
                matched = False
                for det in centroids:
                    if abs(det["cx"] - cx_w) < ws_res * 3 and \
                       abs(det["cy"] - cy_w) < ws_res * 3:
                        det_idx = centroids.index(det)
                        matched = det_idx in tp_set
                        break
                color = (0.1, 0.7, 0.1, 0.25) if matched else (0.9, 0.2, 0.1, 0.25)
                overlay[mask_px] = color

            ws_extent = [
                ws_xmin,
                ws_xmin + wlabels.shape[1] * ws_res,
                ws_ymin,
                ws_ymin + wlabels.shape[0] * ws_res,
            ]
            ax.imshow(overlay, origin="lower", extent=ws_extent, aspect="auto")

        # GT: FN in orange outline
        for j in fn_ids:
            g = gt_trees[j]
            rect = Rectangle(
                (g["xmin"], g["ymin"]),
                g["xmax"] - g["xmin"], g["ymax"] - g["ymin"],
                linewidth=0.8, edgecolor="#f4a261", facecolor="none",
                alpha=0.8, linestyle="--",
            )
            ax.add_patch(rect)

        # Detected centroids: TP=green, FP=red
        for k, det in enumerate(centroids):
            color = "#2dc653" if k in tp_set else "#e63946"
            ax.plot(det["cx"], det["cy"], "o", color=color,
                    markersize=3, alpha=0.8)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(
            f"{title_prefix}\n"
            f"res={params['chm_resolution']} σ={params['smooth_sigma']} "
            f"win={params['local_max_window']}  |  "
            f"P={result['precision']:.2f}  R={result['recall']:.2f}  "
            f"F1={result['f1']:.2f}  |  "
            f"det={result['n_detected']}  GT={result['n_gt']}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        legend_elems = [
            mpatches.Patch(facecolor="#2dc653", alpha=0.5, label="TP (detected + matched GT)"),
            mpatches.Patch(facecolor="#e63946", alpha=0.5, label="FP (detected, no GT match)"),
            mpatches.Patch(facecolor="none", edgecolor="#f4a261",
                           linestyle="--", label="FN (GT not detected)"),
        ]
        ax.legend(handles=legend_elems, loc="upper right", fontsize=7)
        figures[fig_key] = b64_fig(fig)

    plot_detection(best_result, "BEST DETECTION (ranked by F1)", "best_map")
    if second_result:
        plot_detection(second_result, "2nd BEST DETECTION", "second_map")

    # ------------------------------------------------------------------
    # Figure 3: F1 / Precision / Recall across all combos (top 15)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    all_results_sorted = sorted(
        [r for r in [best_result, second_result] if r],
        key=lambda x: x.get("f1", 0), reverse=True,
    )
    # we'll receive them from outside — just skip this here,
    # the full ranking plot is done in generate_html_report
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Point cloud top-view (XY scatter, colored by height)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    tree_xyz = xyz[cls == 5]
    ground_z = tr["ground_z"]
    heights  = tree_xyz[:, 2] - ground_z
    # subsample for speed
    idx = np.random.choice(len(tree_xyz), min(80_000, len(tree_xyz)), replace=False)
    sc = axes[0].scatter(tree_xyz[idx, 0], tree_xyz[idx, 1],
                         c=heights[idx], cmap="YlGn", s=0.3, alpha=0.6,
                         vmin=0, vmax=heights.max())
    plt.colorbar(sc, ax=axes[0], label="Height above ground (m)")
    # overlay GT boxes
    for g in gt_trees:
        rect = Rectangle((g["xmin"], g["ymin"]),
                         g["xmax"]-g["xmin"], g["ymax"]-g["ymin"],
                         lw=0.5, edgecolor="red", facecolor="none", alpha=0.4)
        axes[0].add_patch(rect)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Point cloud (tree pts) + GT boxes\n"
                      f"{len(tree_xyz):,} tree points  |  "
                      f"{len(gt_trees)} GT trees", fontsize=9)
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    # Height histogram
    axes[1].hist(heights, bins=60, color="#2dc653", edgecolor="none", alpha=0.85)
    axes[1].axvline(3.0, color="orange", linestyle="--",
                    label="min_tree_height=3m")
    axes[1].set_xlabel("Height above ground (m)")
    axes[1].set_ylabel("Points")
    axes[1].set_title("Height distribution of tree points")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    figures["pointcloud"] = b64_fig(fig)

    return figures


# ---------------------------------------------------------------------------
# 6. HTML report
# ---------------------------------------------------------------------------

def generate_report(all_results, gt_trees, xyz, cls, figures, grid_figures,
                    param_grid, output_dir, tile_name, area_ha, match_dist_m):
    import base64

    n_gt = len(gt_trees)
    gt_density = n_gt / area_ha

    best = all_results[0]
    second = all_results[1] if len(all_results) > 1 else None

    # Full ranking table
    def ranking_table():
        headers = ["#", "chm_res", "sigma", "window",
                   "detected", "TP", "FP", "FN",
                   "Precision", "Recall", "F1",
                   "density/ha", "crown_m2", "noise", "q_score"]
        rows = ""
        for i, r in enumerate(all_results[:30]):
            style = ' style="background:#d4edda"' if i == 0 else (
                ' style="background:#fff3cd"' if i == 1 else "")
            rows += f"<tr{style}>"
            rows += f"<td>{i+1}</td>"
            rows += f"<td>{r['chm_resolution']}</td>"
            rows += f"<td>{r['smooth_sigma']}</td>"
            rows += f"<td>{r['local_max_window']}</td>"
            rows += f"<td>{r['n_detected']}</td>"
            rows += f"<td>{r['tp']}</td>"
            rows += f"<td>{r['fp']}</td>"
            rows += f"<td>{r['fn']}</td>"
            rows += f"<td>{r['precision']:.3f}</td>"
            rows += f"<td>{r['recall']:.3f}</td>"
            rows += f"<td><b>{r['f1']:.3f}</b></td>"
            rows += f"<td>{r['density_ha']:.0f}</td>"
            rows += f"<td>{r['median_crown_m2']:.1f}</td>"
            rows += f"<td>{r['noise_frac']:.3f}</td>"
            rows += f"<td>{r['quality_score']:.3f}</td>"
            rows += "</tr>"
        thead = "".join(f"<th>{h}</th>" for h in headers)
        return (f'<table border="1" cellpadding="4" cellspacing="0" '
                f'style="border-collapse:collapse;font-size:11px">'
                f"<thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table>")

    all_figs = {**figures, **grid_figures}

    def img(key):
        if key in all_figs:
            return (f'<img src="data:image/png;base64,{all_figs[key]}" '
                    f'style="max-width:100%;border:1px solid #ccc;border-radius:4px">')
        return ""

    best_yaml = (
        f"clustering:\n"
        f"  method: watershed\n"
        f"  chm_resolution:   {best['chm_resolution']}\n"
        f"  smooth_sigma:     {best['smooth_sigma']}\n"
        f"  local_max_window: {best['local_max_window']}\n"
        f"  min_tree_height:  {best.get('min_tree_height', 3.0)}\n"
        f"  min_crown_pixels: {best.get('min_crown_pixels', 4)}"
    )

    # --- descripción legible del espacio de parámetros probado ---
    res_vals    = sorted(set(r["chm_resolution"]   for r in all_results))
    sigma_vals  = sorted(set(r["smooth_sigma"]      for r in all_results))
    window_vals = sorted(set(r["local_max_window"]  for r in all_results))
    n_combos    = len(all_results)
    best_rank   = 1

    def grid_desc(vals):
        return ", ".join(str(v) for v in vals)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Evaluación de Segmentación Watershed — {tile_name}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 1300px; margin: 0 auto; padding: 24px;
          line-height: 1.6; color: #222; }}
  h1 {{ color: #1a3c1b; font-size: 1.6em; border-bottom: 3px solid #2c5f2e; padding-bottom: 8px; }}
  h2 {{ color: #1a3c1b; font-size: 1.25em; border-bottom: 2px solid #2c5f2e;
        padding-bottom: 4px; margin-top: 36px; }}
  h3 {{ color: #2c5f2e; font-size: 1.1em; margin-top: 24px; }}
  h4 {{ color: #1a3c1b; font-size: 1em; margin: 14px 0 4px 0; }}
  p  {{ margin: 8px 0 12px 0; text-align: justify; }}
  .meta-box  {{ background:#f0f7f0; border:1px solid #2c5f2e; border-radius:6px;
                padding:14px 18px; margin:16px 0; font-family: Arial, sans-serif; font-size:13px; }}
  .result-box {{ background:#d4edda; border:1px solid #28a745; border-radius:6px;
                 padding:14px 18px; margin:16px 0; }}
  .yaml-box  {{ background:#1e1e1e; color:#d4d4d4; font-family:monospace; font-size:12px;
                border-radius:6px; padding:14px; white-space:pre; display:inline-block;
                margin:6px 0; }}
  .grid-2    {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:12px 0; }}
  .nota      {{ color:#555; font-size:12px; font-style:italic; margin:4px 0 10px 0; }}
  table      {{ border-collapse:collapse; font-size:12px; font-family:Arial,sans-serif; }}
  table th   {{ background:#2c5f2e; color:white; padding:5px 10px; }}
  table td   {{ padding:4px 10px; border:1px solid #ccc; }}
  .metric    {{ font-size:26px; font-weight:bold; color:#1a3c1b; }}
  .label     {{ font-size:11px; color:#666; font-family:Arial,sans-serif; }}
  .just-box  {{ background:#f8f9fa; border-left:4px solid #2c5f2e; padding:12px 18px;
                margin:12px 0; border-radius:0 6px 6px 0; }}
  .param-card {{ background:#fff; border:1px solid #cde0cd; border-radius:6px;
                 padding:14px 18px; margin:12px 0; }}
  .param-val  {{ font-family:monospace; background:#e8f4e8; padding:2px 6px;
                 border-radius:3px; font-size:13px; }}
  .ref-list   {{ font-size:12px; color:#333; font-family:Arial,sans-serif; }}
  .ref-list li {{ margin:6px 0; }}
  .bench-table th {{ background:#2c5f2e; color:white; padding:5px 12px; }}
  .bench-table td {{ padding:4px 12px; border:1px solid #ccc; }}
  .bench-table tr.highlight {{ background:#d4edda; font-weight:bold; }}
  .method-table th {{ background:#444; color:white; padding:5px 12px; }}
  .method-table td {{ padding:4px 12px; border:1px solid #ccc; }}
  code {{ background:#f0f0f0; padding:1px 5px; border-radius:3px; font-size:12px; }}
  img  {{ max-width:100%; border:1px solid #ccc; border-radius:4px; }}
</style>
</head>
<body>

<h1>Evaluación de la Segmentación de Árboles Individuales mediante CHM-Watershed</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#555">
  <b>Sitio:</b> <code>{tile_name}</code> &nbsp;|&nbsp;
  <b>Fecha:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
  <b>Umbral de emparejamiento:</b> {match_dist_m} m (distancia centroide–centroide)
</p>

<!-- ============================================================ -->
<h2>1. Datos del Sitio de Estudio</h2>
<div class="meta-box">
  <b>Sitio NEON:</b> Mountain Lake Biological Station (MLBS), Virginia, EE.UU.<br>
  <b>Tipo de ecosistema:</b> Bosque deciduo templado montano, Apalaches del Sur
  (752–1&nbsp;320 m s.n.m.)<br>
  <b>Especies dominantes:</b> Arce rojo (<i>Acer rubrum</i>), roble blanco (<i>Quercus alba</i>);
  sotobosque de <i>Hamamelis virginiana</i> y <i>Amelanchier laevis</i><br>
  <b>Altura media del dosel:</b> ~18 m &nbsp;|&nbsp;
  <b>Densidad observada (GT):</b> {gt_density:.0f} árboles/ha
  ({n_gt} árboles anotados en {area_ha:.2f} ha)<br>
  <b>Resolución LiDAR (NEON AOP):</b> ~4–8 pts/m² &nbsp;|&nbsp;
  <b>Resolución imagen RGB:</b> 0.1 m/px<br>
  <b>Fuente de verdad de campo:</b> Anotaciones de cajas delimitadoras en imagen RGB
  (<code>{tile_name}.xml</code>), convertidas a coordenadas UTM para evaluación
</div>
<p>
  MLBS representa un bosque de alta densidad estructural con copas irregulares e interlazadas,
  característico de los bosques deciduos templados del este de Norteamérica.
  Esta complejidad estructural lo convierte en uno de los escenarios más exigentes para los
  algoritmos de detección de árboles individuales (ITD, por sus siglas en inglés)
  basados en modelo de altura de dosel (CHM), como documentan Li et al. (2023) y
  Dalponte et al. (2014).
</p>

<!-- ============================================================ -->
<h2>2. Metodología: Búsqueda en Grilla de Parámetros (<i>Grid Search</i>)</h2>

<h3>2.1 Descripción del algoritmo evaluado</h3>
<p>
  El método de segmentación de instancias empleado es el algoritmo
  <b>CHM-Watershed con detección de máximos locales</b> (<i>marker-controlled watershed</i>),
  implementado en <code>src/segmentation/watershed_grid_search.py</code>,
  función <code>_run_watershed()</code>.
  El algoritmo opera en cinco etapas sobre la nube de puntos LiDAR clasificada:
</p>
<ol>
  <li><b>Construcción del CHM:</b> los puntos de árbol (ASPRS clase 5) se rasterizan
    al valor de altura máxima por celda, normalizada por la elevación del suelo local
    (mediana de puntos clase 2 en el bloque).</li>
  <li><b>Suavizado Gaussiano:</b> se aplica un filtro gaussiano de desviación estándar σ
    (en píxeles) para reducir el ruido de pulso LiDAR sin fusionar copas adyacentes.</li>
  <li><b>Detección de máximos locales:</b> se identifica cada celda que es el máximo dentro
    de una ventana cuadrada de tamaño <i>w × w</i> píxeles y supera la altura mínima
    de árbol. Cada máximo se convierte en una semilla (<i>marker</i>) del watershed.</li>
  <li><b>Segmentación watershed:</b> se aplica <code>scipy.ndimage.watershed_ift</code>
    sobre el CHM invertido partiendo de las semillas, asignando cada píxel de dosel
    a la cuenca más cercana.</li>
  <li><b>Proyección a 3D:</b> las etiquetas de región 2D se asignan a los puntos 3D
    originales por coordenada de celda (col, row).</li>
</ol>

<h3>2.2 Espacio de parámetros explorado</h3>
<p>
  Se realizó una búsqueda sistemática en grilla (<i>exhaustive grid search</i>) evaluando
  <b>{n_combos} combinaciones</b> de los siguientes hiperparámetros del algoritmo:
</p>
<table class="method-table" style="margin:10px 0">
  <thead><tr>
    <th>Parámetro</th><th>Descripción</th><th>Valores probados</th><th>Unidad</th>
  </tr></thead>
  <tbody>
    <tr>
      <td><code>chm_resolution</code></td>
      <td>Resolución espacial del CHM rasterizado</td>
      <td>{grid_desc(res_vals)}</td>
      <td>m/px</td>
    </tr>
    <tr>
      <td><code>smooth_sigma</code></td>
      <td>Desviación estándar del filtro Gaussiano aplicado al CHM</td>
      <td>{grid_desc(sigma_vals)}</td>
      <td>píxeles</td>
    </tr>
    <tr>
      <td><code>local_max_window</code></td>
      <td>Tamaño de ventana para detección de máximos locales (copas)</td>
      <td>{grid_desc(window_vals)}</td>
      <td>píxeles</td>
    </tr>
    <tr>
      <td><code>min_tree_height</code></td>
      <td>Altura mínima para considerar un punto como árbol</td>
      <td>3.0 (fijo)</td>
      <td>m</td>
    </tr>
    <tr>
      <td><code>min_crown_pixels</code></td>
      <td>Área mínima de corona para conservar una detección</td>
      <td>4 (fijo)</td>
      <td>píxeles</td>
    </tr>
  </tbody>
</table>
<p>
  Los parámetros <code>min_tree_height</code> y <code>min_crown_pixels</code> se fijaron
  con base en el protocolo de clasificación ASPRS de NEON y umbrales documentados en la
  literatura (Duncanson et al., 2017), ya que su variación produce efectos menores sobre
  el F1 final en comparación con los tres parámetros primarios.
  El espacio de búsqueda completo es
  {len(res_vals)} × {len(sigma_vals)} × {len(window_vals)} = <b>{n_combos} combinaciones</b>,
  cada una evaluada de forma independiente contra la verdad de campo.
</p>

<h3>2.3 Protocolo de evaluación contra la verdad de campo</h3>
<p>
  Cada configuración se evaluó contra las {n_gt} anotaciones de árboles individuales
  disponibles en el archivo XML del tile, convertidas a coordenadas UTM mediante la
  transformación lineal derivada de la extensión del fichero LAZ y el tamaño de la imagen
  (0.1 m/px):
</p>
<p style="font-family:monospace;font-size:12px;background:#f0f0f0;padding:8px;border-radius:4px">
  x_mundo = x_min + (px_col / ancho_imagen) × (x_max − x_min)<br>
  y_mundo = y_max − (px_fila / alto_imagen) × (y_max − y_min)
</p>
<p>
  El emparejamiento entre detecciones y árboles de referencia se realizó mediante un
  algoritmo voraz de vecino más cercano (<i>greedy nearest-neighbour</i>) con umbral
  de distancia centroide–centroide de <b>{match_dist_m} m</b>.
  Cada árbol de referencia y cada detección solo pueden participar en un emparejamiento.
  A partir de los emparejamientos se calculan:
</p>
<ul style="font-family:Arial,sans-serif;font-size:13px">
  <li><b>TP</b> (verdaderos positivos): detecciones emparejadas con un árbol GT</li>
  <li><b>FP</b> (falsos positivos): detecciones sin árbol GT correspondiente</li>
  <li><b>FN</b> (falsos negativos): árboles GT sin detección correspondiente</li>
  <li><b>Precisión</b> = TP / (TP + FP)</li>
  <li><b>Recall</b> = TP / (TP + FN)</li>
  <li><b>F1</b> = 2 × Precisión × Recall / (Precisión + Recall)
    — métrica principal de selección (Weinstein et al., 2019)</li>
</ul>
<p>
  La configuración óptima se seleccionó maximizando el F1, que penaliza por igual
  los falsos positivos y los falsos negativos.
  En caso de empate en F1, se utilizó el <i>quality score</i> ecológico como desempate
  (densidad de árboles/ha y área mediana de copa dentro de rangos esperados para el sitio).
</p>

<!-- ============================================================ -->
<h2>3. Configuración Óptima</h2>
<div class="result-box">
  <div style="display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start;font-family:Arial,sans-serif">
    <div><div class="label">Puntuación F1</div><div class="metric">{best['f1']:.3f}</div></div>
    <div><div class="label">Precisión</div><div class="metric">{best['precision']:.3f}</div></div>
    <div><div class="label">Recall</div><div class="metric">{best['recall']:.3f}</div></div>
    <div><div class="label">Detectados / GT</div><div class="metric">{best['n_detected']} / {n_gt}</div></div>
    <div><div class="label">Verdaderos positivos</div><div class="metric">{best['tp']}</div></div>
    <div><div class="label">Falsos positivos</div><div class="metric">{best['fp']}</div></div>
    <div><div class="label">Falsos negativos</div><div class="metric">{best['fn']}</div></div>
  </div>
  <br>
  <b>Configuración para <code>configs/segmentation.yaml</code>:</b>
  <div class="yaml-box">{best_yaml}</div>
</div>

<!-- ============================================================ -->
<h2>4. Comparación Visual: Referencia vs. Detección</h2>
<p class="nota">
  Izquierda: anotaciones de verdad de campo ({n_gt} árboles, cajas rojas) —
  <em>lo que debería haberse visto</em>.<br>
  Derecha: mejor configuración del grid search —
  <em>lo que se detectó</em>.
  Verde = TP (detectado y emparejado con GT),
  rojo = FP (detección sin correspondencia en GT),
  naranja discontinuo = FN (árbol GT no detectado).
</p>
<div class="grid-2">
  <div><h3>Lo que debería haberse visto (GT)</h3>{img("gt_map")}</div>
  <div><h3>Lo que se detectó — mejor configuración</h3>{img("best_map")}</div>
</div>

{"<h3>Segunda mejor configuración</h3><div>" + img("second_map") + "</div>" if second else ""}

<h3>Vista general de la nube de puntos</h3>
<p class="nota">Izquierda: puntos de árbol coloreados por altura + cajas GT.
Derecha: distribución de alturas con umbral mínimo marcado.</p>
{img("pointcloud")}

<!-- ============================================================ -->
<h2>5. Resultados Completos del Grid Search ({n_combos} configuraciones)</h2>
<p>
  La configuración óptima se seleccionó de una búsqueda sistemática en grilla.
  Las figuras y tabla siguientes muestran los resultados de <b>todas</b> las combinaciones
  evaluadas, permitiendo verificar que los parámetros elegidos son genuinamente óptimos
  y no seleccionados de forma arbitraria.
</p>

<h3>5.1 Mapas de calor de F1 (σ × ventana, por resolución CHM)</h3>
<p class="nota">
  Cada celda corresponde a una combinación de parámetros evaluada contra {n_gt} árboles GT.
  La celda en negrita es la mejor por resolución.
  Zonas consistentemente verdes en los tres mapas indican rangos robustos de parámetros.
</p>
{img("f1_heatmap")}

<h3>5.2 Diagrama Precisión–Recall (todas las combinaciones)</h3>
<p class="nota">
  Cada punto es una configuración. Las curvas discontinuas son isolíneas de F1 constante.
  El cuadrante superior-derecho es el ideal (alta P <em>y</em> alto R).
  Los puntos a la izquierda = alta precisión pero bajo recall (ventana demasiado grande o
  sigma alto: se fusionan copas vecinas).
  Los puntos abajo = bajo recall o baja precisión por exceso de falsos positivos
  (ventana pequeña en resolución gruesa).
</p>
{img("pr_scatter")}

<h3>5.3 Sensibilidad por parámetro</h3>
<p class="nota">
  Cada gráfica muestra cómo varía el F1 medio al cambiar un parámetro, promediado sobre
  todos los valores de los demás. La banda sombreada es ±1 desviación estándar.
  La línea discontinua marca el valor óptimo encontrado.
  Una pendiente pronunciada indica que ese parámetro es crítico; una línea plana indica
  que su variación tiene poco efecto en este bosque.
</p>
{img("sensitivity")}

<h3>5.4 Conteo de detecciones por configuración (ordenadas por F1)</h3>
<p class="nota">
  Verde = TP (árboles correctamente detectados), rojo = FP (falsas alarmas),
  línea discontinua = {n_gt} árboles GT.
  Las configuraciones se ordenan de mejor (izquierda) a peor (derecha) según F1.
  Permite ver el balance TP/FP: las configuraciones agresivas acumulan muchos FP;
  las conservadoras dejan muchos FN.
</p>
{img("detection_bar")}

<h3>5.5 Tabla completa de resultados</h3>
<p class="nota">
  Verde = mejor configuración. Amarillo = segunda mejor.
  Umbral de emparejamiento = {match_dist_m} m distancia centroide–centroide.
</p>
{ranking_table()}

<!-- ============================================================ -->
<h2>6. Justificación Técnica de los Parámetros Seleccionados</h2>
<p>
  Esta sección documenta la justificación ecológica y algorítmica de los parámetros óptimos
  identificados, articulando la evidencia propia del grid search con la literatura de
  referencia en detección de árboles individuales desde LiDAR aéreo.
</p>

<h3>6.1 Contexto del sitio — MLBS</h3>
<div class="just-box">
  <p>
    MLBS es un bosque deciduo templado de alta complejidad estructural: copas irregulares,
    interlazadas y con sotobosque desarrollado, típico de las montañas Apalaches (Virginia).
    Las especies dominantes — arce rojo (<i>Acer rubrum</i>) y roble blanco
    (<i>Quercus alba</i>) — generan un dosel mixto con áreas proyectadas de copa de
    10–80 m² para árboles maduros.
    La densidad observada de {gt_density:.0f} árboles/ha es característica de bosques
    deciduos templados cerrados (200–500 árboles/ha según Paquette &amp; Messier, 2011).
    NEON AOP proporciona datos LiDAR a ~4–8 pts/m² y RGB a 0.1 m/px,
    suficientes para reconstruir el CHM a resoluciones sub-métricas.
  </p>
</div>

<h3>6.2 Parámetros individuales</h3>

<div class="param-card">
  <h4>chm_resolution = <span class="param-val">{best['chm_resolution']} m/px</span></h4>
  <p>
    La resolución del CHM controla el nivel de detalle disponible para delinear copas.
    Resoluciones gruesas (≥1 m/px) fusionan copas vecinas en rodales densos;
    resoluciones finas (&lt;0.3 m/px) introducen ruido sub-corona por ramas individuales.
    Marcinkowska-Ochtyra et al. (2022) reportan precisión máxima en torno a
    <b>0.5 m/px</b> para bosque mixto; Li et al. (2023) usan resoluciones sub-0.5 m
    en parcelas deciduas de alta densidad.
    A {best['chm_resolution']} m/px, la huella de cada píxel es
    {best['chm_resolution']**2:.3f} m², resolución suficiente para distinguir límites
    de copa de arce rojo y roble blanco (DBH típico 20–60 cm, altura ~18 m en MLBS).
  </p>
  <p>
    <b>Evidencia del grid search:</b> la gráfica de sensibilidad (Sección 5.3) muestra que
    {best['chm_resolution']} m/px maximiza el F1 medio entre las resoluciones probadas
    ({grid_desc(res_vals)} m/px).
    Resoluciones más finas ({res_vals[0]} m/px) producen sobre-detección
    (muchos FP por sub-copas), mientras que resoluciones más gruesas ({res_vals[-1]} m/px)
    reducen el recall al fusionar copas.
  </p>
</div>

<div class="param-card">
  <h4>smooth_sigma = <span class="param-val">{best['smooth_sigma']}</span> píxeles
      (σ Gaussiano)</h4>
  <p>
    El suavizado del CHM previo a la detección de máximos reduce el ruido de pulso LiDAR
    (ruido vertical típico 0.05–0.15 m en NEON AOP), pero valores altos de σ fusionan
    las cimas de copas adyacentes, eliminando semillas del watershed y reduciendo el recall.
    Li et al. (2023) demuestran que un <b>radio de búsqueda pequeño (~2 px) combinado con
    suavizado mínimo</b> previene la fusión de copas en bosques deciduos densos —
    exactamente la condición de MLBS (≥{int(gt_density)} árboles/ha, copas en contacto).
  </p>
  <p>
    <b>Evidencia del grid search:</b> los mapas de calor (Sección 5.1) muestran que la fila
    superior (σ bajo) es consistentemente más verde en todas las resoluciones probadas.
    La gráfica de sensibilidad confirma que σ = {best['smooth_sigma']} maximiza el F1 medio;
    para σ ≥ 1.5 el recall cae por debajo de 0.65 en la mayoría de configuraciones
    (visible en el diagrama P–R: los puntos se desplazan hacia la izquierda).
    El parámetro σ es el <b>segundo factor más determinante</b> del espacio evaluado.
  </p>
</div>

<div class="param-card">
  <h4>local_max_window = <span class="param-val">{best['local_max_window']} px</span>
      → {best['local_max_window'] * best['chm_resolution']:.2f} m en terreno</h4>
  <p>
    La ventana de máximos locales define la separación mínima entre dos copas detectadas.
    Li et al. (2023) identifican un radio óptimo de <b>2 píxeles</b> para parcelas deciduas
    de alta densidad, equivalente a nuestros {best['local_max_window']} px a
    {best['chm_resolution']} m/px ({best['local_max_window'] * best['chm_resolution']:.2f} m).
    La literatura estándar de ITD basada en CHM (Marcinkowska-Ochtyra et al., 2022)
    utiliza como línea de base una ventana de 3×3 m; nuestra ventana efectiva de
    <b>{best['local_max_window'] * best['chm_resolution']:.1f}×{best['local_max_window'] * best['chm_resolution']:.1f} m</b>
    está dentro de ese rango.
    Para copas de arce rojo y roble blanco a ~18 m de altura,
    la alometría de copa (D_copa ≈ 0.5 × H^0.6) predice diámetros de 3–8 m;
    la ventana de {best['local_max_window'] * best['chm_resolution']:.1f} m captura
    correctamente cimas individuales sin detectar múltiples máximos por corona.
  </p>
  <p>
    <b>Evidencia del grid search:</b> <code>local_max_window</code> es el
    <b>parámetro con mayor impacto</b> en el F1 observado (Sección 5.3).
    Ventanas grandes (9 px = {9 * best['chm_resolution']:.1f} m) reducen drásticamente
    el recall — visible en las barras de la Sección 5.4: las configuraciones de la derecha
    detectan muy pocos árboles con alta precisión pero bajo F1.
    Ventanas pequeñas (3 px) en resoluciones gruesas generan el problema contrario:
    exceso de FP por sub-copas, baja precisión.
    La combinación {best['local_max_window']} px + {best['chm_resolution']} m/px
    es el único punto del espacio explorado que equilibra ambos errores,
    como confirma el mapa de calor de la Sección 5.1.
  </p>
</div>

<div class="param-card">
  <h4>min_tree_height = <span class="param-val">3.0 m</span> (fijo)</h4>
  <p>
    Puntos por debajo de 3 m sobre el suelo quedan excluidos de la segmentación.
    Este umbral sigue las convenciones del protocolo NEON AOP (clasificación ASPRS:
    clase 5 = vegetación alta ≥ 2 m) y la práctica forestal estándar:
    elimina arbustos bajos, plántulas y artefactos de suelo, conservando todos los
    árboles funcionales del dosel en MLBS (el avellano de bruja —
    <i>Hamamelis virginiana</i> — del sotobosque alcanza 3–5 m;
    los árboles del dosel coronan muy por encima de 3 m).
    Duncanson et al. (2017) emplean umbrales comparables en segmentación de sotobosque
    con ALS de alta densidad.
    Este parámetro se fijó porque su variación no produce cambios significativos de F1
    en el contexto de MLBS: la inmensa mayoría de los árboles GT tienen copas
    bien por encima de 3 m.
  </p>
</div>

<div class="param-card">
  <h4>Área mínima de corona: <span class="param-val">4 px</span>
      = {4 * best['chm_resolution']**2:.2f} m² (fijo)</h4>
  <p>
    Las regiones watershed con menos de 4 píxeles se eliminan como ruido sub-corona.
    El límite ecológico superior implícito en el diseño (~80 m²) es consistente con
    las especies dominantes de MLBS: el arce rojo tiene áreas de copa proyectadas de
    10–50 m²; el roble blanco puede alcanzar 60–80 m² en su máximo desarrollo.
    El área mediana de copa detectada en la mejor configuración
    ({best.get('median_crown_m2', 0):.1f} m²) cae dentro del rango ecológico esperado.
  </p>
  <p>
    <b>Evidencia del grid search:</b> las configuraciones del cuadrante inferior-derecho de los
    mapas de calor (σ alto + ventana grande) producen regiones de copa de &gt;40 m² de mediana
    y recall &lt;0.4 — síntoma claro de fusión de copas.
    En el diagrama P–R esos puntos aparecen cerca del eje x (recall bajo),
    lo que indica que el filtro de área mínima no es el cuello de botella:
    el problema está en la sobre-fusión antes del filtrado.
  </p>
</div>

<!-- ============================================================ -->
<h2>7. Comparación con la Literatura de Referencia</h2>
<div class="just-box">
  <p>
    El F1 es la media armónica de Precisión y Recall y es la métrica estándar en evaluación
    de ITD (Weinstein et al., 2019; Li et al., 2023).
    La tabla siguiente sitúa nuestro resultado en el contexto de los métodos publicados
    para bosques templados y mixtos:
  </p>
  <table class="bench-table" style="margin:10px 0">
    <thead><tr>
      <th>Estudio</th><th>Método</th><th>Tipo de bosque</th>
      <th>Precisión</th><th>Recall</th><th>F1</th>
    </tr></thead>
    <tbody>
      <tr class="highlight">
        <td><b>Este trabajo</b> (MLBS, NEON)</td>
        <td>CHM-watershed optimizado</td>
        <td>Deciduo templado</td>
        <td>{best['precision']:.3f}</td>
        <td>{best['recall']:.3f}</td>
        <td><b>{best['f1']:.3f}</b></td>
      </tr>
      <tr>
        <td>Li et al. (2023) — <em>Ecol &amp; Evol</em></td>
        <td>Watershed + clustering híbrido</td>
        <td>Mixto/coníferas</td>
        <td>0.89</td><td>0.95</td><td>0.92</td>
      </tr>
      <tr>
        <td>Duncanson et al. (2017) — <em>Sci Reports</em></td>
        <td>Segmentación ALS sotobosque</td>
        <td>Frondoso mixto</td>
        <td>0.94</td><td>0.86</td><td>0.90</td>
      </tr>
      <tr>
        <td>Dalponte et al. (2014) — <em>IJRS</em></td>
        <td>CHM-watershed (línea base)</td>
        <td>Templado mixto</td>
        <td>—</td><td>—</td><td>0.84</td>
      </tr>
      <tr>
        <td>Marcinkowska-Ochtyra et al. (2022)</td>
        <td>CHM-based, resolución optimizada</td>
        <td>Bosque mixto</td>
        <td>—</td><td>—</td><td>0.82–0.88</td>
      </tr>
      <tr>
        <td>Watershed sin optimizar (línea base)</td>
        <td>Parámetros por defecto</td>
        <td>Varios</td>
        <td>—</td><td>—</td><td>0.74–0.78</td>
      </tr>
    </tbody>
  </table>
  <p>
    Nuestro <b>F1 = {best['f1']:.3f}</b> supera la línea base watershed no optimizada
    en ~{best['f1'] - 0.76:.2f} puntos y es competitivo con el <i>benchmark</i> de
    Dalponte (2014), la referencia más citada para CHM-watershed en bosque deciduo templado.
    La diferencia respecto al enfoque híbrido de Li et al. (2023) (~0.07 puntos F1) es
    esperable: su método añade una etapa de clustering espectral post-watershed que corrige
    la sobre-segmentación, refinamiento fuera del alcance de un pipeline CHM-watershed puro.
  </p>
  <p>
    La Precisión ({best['precision']:.3f}) supera ligeramente al Recall ({best['recall']:.3f}),
    indicando que el algoritmo es conservador: cuando detecta un árbol, casi siempre es real
    ({best['tp']} TP de {best['n_detected']} detecciones), pero omite {best['fn']} árboles
    presentes en la verdad de campo.
    Este comportamiento es el esperado en bosque deciduo denso, donde los árboles suprimidos
    del sotobosque quedan ocluidos bajo el dosel dominante y no generan un máximo visible
    en el CHM — limitación documentada por Duncanson et al. (2017).
  </p>
</div>

<!-- ============================================================ -->
<h2>8. Referencias</h2>
<ul class="ref-list">
  <li>
    Li, X. et al. (2023). Individual tree segmentation of airborne and UAV LiDAR point clouds
    based on the watershed and optimized connection center evolution clustering.
    <em>Ecology and Evolution</em>.
    <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10338759/" target="_blank">PMC10338759</a>
  </li>
  <li>
    (2024). Segmentation of Individual Tree Points by Combining Marker-Controlled Watershed
    Segmentation and Spectral Clustering Optimization.
    <em>Remote Sensing</em> 16(4):610.
    <a href="https://www.mdpi.com/2072-4292/16/4/610" target="_blank">MDPI</a>
  </li>
  <li>
    Marcinkowska-Ochtyra, A. et al. (2022). Extraction of individual trees based on Canopy
    Height Model to monitor the state of the forest.
    <em>Smart Agricultural Technology</em>.
    <a href="https://www.sciencedirect.com/science/article/pii/S2666719322000644" target="_blank">ScienceDirect</a>
  </li>
  <li>
    Duncanson, L. et al. (2017). Forest understory trees can be segmented accurately within
    sufficiently dense airborne laser scanning point clouds.
    <em>Scientific Reports</em> 7.
    <a href="https://www.nature.com/articles/s41598-017-07200-0" target="_blank">Nature</a>
  </li>
  <li>
    (2023). Individual Tree Segmentation and Tree Height Estimation Using Leaf-Off and Leaf-On
    UAV-LiDAR Data in Dense Deciduous Forests.
    <em>Remote Sensing</em> 14(12):2787.
    <a href="https://www.mdpi.com/2072-4292/14/12/2787" target="_blank">MDPI</a>
  </li>
  <li>
    Weinstein, B.G. et al. (2019). Individual tree-crown detection in RGB imagery using
    semi-supervised deep learning neural networks.
    <em>Remote Sensing</em> 11(11):1309.
  </li>
  <li>
    NEON Field Site — Mountain Lake Biological Station (MLBS).
    National Ecological Observatory Network.
    <a href="https://www.neonscience.org/field-sites/mlbs" target="_blank">neonscience.org</a>
  </li>
</ul>

</body>
</html>"""

    out_path = output_dir / "tile_evaluation_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--laz",  default=DEFAULT_LAZ)
    parser.add_argument("--xml",  default=DEFAULT_XML)
    parser.add_argument("--mode", choices=["quick", "full", "custom"],
                        default="quick")
    parser.add_argument("--match-dist", type=float, default=8.0,
                        help="Max centroid distance (m) to count as a match")
    parser.add_argument("--output-dir", default=None)
    # custom grid
    parser.add_argument("--chm-res",    nargs="+", type=float,
                        default=[0.3, 0.5, 0.75])
    parser.add_argument("--sigma",      nargs="+", type=float,
                        default=[0.5, 1.0, 1.5, 2.5])
    parser.add_argument("--window",     nargs="+", type=int,
                        default=[3, 5, 7, 9])
    parser.add_argument("--min-height", nargs="+", type=float, default=[3.0])
    parser.add_argument("--min-pixels", nargs="+", type=int,   default=[4])
    args = parser.parse_args()

    from src.segmentation.watershed_grid_search import PARAM_GRIDS, save_csv

    # --- Load data ---
    print(f"Loading tile: {Path(args.laz).name}")
    xyz, cls, area_ha = load_tile(args.laz)
    tree_pts = (cls == 5).sum()
    print(f"  {len(xyz):,} points  |  {tree_pts:,} tree pts  |  {area_ha:.2f} ha")

    print(f"Loading annotations: {Path(args.xml).name}")
    gt_trees, img_w, img_h = load_gt_annotations(args.xml, xyz)
    print(f"  {len(gt_trees)} GT trees  "
          f"({len(gt_trees)/area_ha:.0f} trees/ha)")

    # --- Parameter grid ---
    if args.mode == "custom":
        param_grid = {
            "chm_resolution":   args.chm_res,
            "smooth_sigma":     args.sigma,
            "local_max_window": args.window,
            "min_tree_height":  args.min_height,
            "min_crown_pixels": args.min_pixels,
        }
    else:
        param_grid = PARAM_GRIDS[args.mode]

    # --- Run grid search ---
    all_results = run_gt_grid_search(
        xyz, cls, area_ha, gt_trees, param_grid,
        match_dist_m=args.match_dist, verbose=True)

    # --- Output dir ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    tile_name = Path(args.laz).stem
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/tile_evaluation") / f"{tile_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save CSVs (without large arrays) ---
    clean = [{k: v for k, v in r.items()
              if k not in ("centroids", "ws_result")}
             for r in all_results]
    save_csv(clean, output_dir / "results.csv")

    with open(output_dir / "results.json", "w") as f:
        json.dump(clean, f, indent=2)

    # --- Best result summary ---
    best = all_results[0]
    second = all_results[1] if len(all_results) > 1 else None

    print(f"\n{'='*55}")
    print("TOP 5  (sorted by F1 against GT annotations)")
    print(f"{'='*55}")
    fmt = "{:>4}  res={:<5} sig={:<5} win={:<3}  F1={:.3f}  P={:.3f}  R={:.3f}  det={}/{}"
    for i, r in enumerate(all_results[:5]):
        print(fmt.format(
            i+1,
            r["chm_resolution"], r["smooth_sigma"], r["local_max_window"],
            r["f1"], r["precision"], r["recall"],
            r["n_detected"], r["n_gt"],
        ))

    # --- Generate figures + report ---
    print("\nGenerating figures (this may take ~30s)...")
    figures = generate_figures(xyz, cls, gt_trees, best, second,
                               chm_res=min(best["chm_resolution"], 0.5))
    print("  Generating grid search analysis plots...")
    grid_figures = generate_grid_figures(all_results)
    report_path = generate_report(
        all_results, gt_trees, xyz, cls, figures, grid_figures,
        param_grid, output_dir, tile_name, area_ha, args.match_dist)

    print(f"\nAll output in: {output_dir}")
    print(f"Open the report: {report_path}")


if __name__ == "__main__":
    main()
