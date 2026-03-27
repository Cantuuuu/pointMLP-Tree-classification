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

def generate_report(all_results, gt_trees, xyz, cls, figures,
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

    def img(key):
        if key in figures:
            return (f'<img src="data:image/png;base64,{figures[key]}" '
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

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Watershed Tile Evaluation — {tile_name}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1300px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #1a3c1b; }}
  h2 {{ color: #2c5f2e; border-bottom: 2px solid #2c5f2e; padding-bottom: 4px; margin-top: 30px; }}
  h3 {{ color: #3a7d3b; }}
  .meta-box  {{ background:#f0f7f0; border:1px solid #2c5f2e; border-radius:8px; padding:14px; margin:14px 0; }}
  .best-box  {{ background:#d4edda; border:1px solid #28a745; border-radius:8px; padding:14px; margin:14px 0; }}
  .yaml-box  {{ background:#1e1e1e; color:#d4d4d4; font-family:monospace; font-size:13px;
                border-radius:6px; padding:14px; white-space:pre; display:inline-block; margin:8px 0; }}
  .grid-2    {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .note      {{ color:#555; font-size:12px; font-style:italic; }}
  table th   {{ background:#2c5f2e; color:white; padding:4px 8px; }}
  table td   {{ padding:3px 8px; }}
  .metric    {{ font-size:22px; font-weight:bold; color:#2c5f2e; }}
  .label     {{ font-size:11px; color:#666; }}
</style>
</head>
<body>
<h1>Watershed Evaluation Report</h1>
<p><b>Tile:</b> <code>{tile_name}</code> &nbsp;|&nbsp;
   <b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
   <b>Match threshold:</b> {match_dist_m} m</p>

<div class="meta-box">
  <b>Tile statistics:</b><br>
  &nbsp;&nbsp;Area: <b>{area_ha:.2f} ha</b> &nbsp;|&nbsp;
  GT trees: <b>{n_gt}</b> &nbsp;|&nbsp;
  GT density: <b>{gt_density:.0f} trees/ha</b>
  (MLBS = Mountain Lake Biological Station, VA — dense deciduous forest)
</div>

<h2>Best Configuration (for sharing)</h2>
<div class="best-box">
  <div style="display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start">
    <div>
      <div class="label">F1 Score</div>
      <div class="metric">{best['f1']:.3f}</div>
    </div>
    <div>
      <div class="label">Precision</div>
      <div class="metric">{best['precision']:.3f}</div>
    </div>
    <div>
      <div class="label">Recall</div>
      <div class="metric">{best['recall']:.3f}</div>
    </div>
    <div>
      <div class="label">Detected / GT</div>
      <div class="metric">{best['n_detected']} / {n_gt}</div>
    </div>
    <div>
      <div class="label">True Positives</div>
      <div class="metric">{best['tp']}</div>
    </div>
  </div>
  <br>
  <b>Paste into <code>configs/segmentation.yaml</code>:</b><br>
  <div class="yaml-box">{best_yaml}</div>
</div>

<h2>Visual Comparison</h2>
<p class="note">
  Left: Ground Truth annotations ({n_gt} trees, red boxes).<br>
  Right: Best watershed detection — green fill = TP (matched), red fill = FP (unmatched),
  orange dashed = FN (GT missed).
</p>
<div class="grid-2">
  <div><h3>What should have been seen (GT)</h3>{img("gt_map")}</div>
  <div><h3>What was seen — Best config</h3>{img("best_map")}</div>
</div>

{"<h2>2nd Best Configuration</h2><div>" + img("second_map") + "</div>" if second else ""}

<h2>Point Cloud Overview</h2>
{img("pointcloud")}

<h2>Full Ranking ({len(all_results)} configurations, sorted by F1)</h2>
<p class="note">
  Green = best. Yellow = 2nd best.<br>
  F1 measures balance between Precision and Recall against the GT annotations.<br>
  Match threshold = {match_dist_m} m (centroid-to-centroid distance).
</p>
{ranking_table()}

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
    report_path = generate_report(
        all_results, gt_trees, xyz, cls, figures,
        param_grid, output_dir, tile_name, area_ha, args.match_dist)

    print(f"\nAll output in: {output_dir}")
    print(f"Open the report: {report_path}")


if __name__ == "__main__":
    main()
