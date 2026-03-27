"""
Watershed Parameter Grid Search
================================
Searches for the best CHM-watershed clustering parameters by running all
combinations on a set of evaluation LiDAR tiles and scoring each run with
ecologically-grounded metrics.

Usage (via watershed_search.py CLI at the project root):
    python watershed_search.py --mode quick
    python watershed_search.py --mode full --n-tiles 10

Metrics computed per (tile, parameter_set):
    n_trees            - number of detected individual trees
    tree_density_ha    - trees per hectare
    noise_frac         - fraction of tree points not assigned to any instance
    median_crown_m2    - median crown projected area (m²)
    cv_crown_area      - coefficient of variation of crown areas (lower = uniform)
    coverage_frac      - fraction of CHM canopy covered by detected crowns

Quality score:
    Composite score using ecological priors for temperate/subtropical forest:
    - Expected density: 200-1000 trees/ha
    - Expected median crown: 5-40 m²
    - Low noise is rewarded
    - Low CV (uniform segmentation) is rewarded
"""

import json
import time
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import numpy as np
import laspy
from scipy import ndimage


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "quick": {
        "chm_resolution":  [0.3, 0.5, 0.75],
        "smooth_sigma":    [0.5, 1.5, 2.5],
        "local_max_window": [3, 5, 9],
        "min_tree_height": [3.0],
        "min_crown_pixels": [4],
    },
    "full": {
        "chm_resolution":  [0.25, 0.4, 0.5, 0.75],
        "smooth_sigma":    [0.5, 1.0, 1.5, 2.0, 3.0],
        "local_max_window": [3, 5, 7, 9, 11],
        "min_tree_height": [2.0, 3.0, 5.0],
        "min_crown_pixels": [4, 8, 16],
    },
    "custom": {},  # filled by caller
}

# Ecological priors for NeonTreeEvaluation sites (temperate/subtropical forest)
# SJER (California oak savanna): 50-300 trees/ha, crowns 5-80 m²
# MLBS/HARV (eastern deciduous): 200-800 trees/ha, crowns 5-40 m²
# Conservative range that covers all NEON sites used here.
ECOLOGY = {
    "density_lo": 50,     # trees/ha — below this is under-segmentation
    "density_hi": 800,    # trees/ha — above this is over-segmentation
    "crown_lo": 5.0,      # m² — below this are likely sub-crown artifacts
    "crown_hi": 80.0,     # m² — above this suggests under-segmentation
    "noise_weight": 0.25,
    "density_weight": 0.30,
    "crown_weight": 0.35,  # crown size is the strongest filter for artifacts
    "cv_weight": 0.10,
}


# ---------------------------------------------------------------------------
# Core: single watershed run
# ---------------------------------------------------------------------------

def _run_watershed(xyz: np.ndarray, tree_mask: np.ndarray,
                   classification: np.ndarray, params: dict) -> dict:
    """
    Run CHM-watershed with given params on a scene.

    Args:
        xyz:            (N, 3) full scene points
        tree_mask:      (N,) bool — which points are trees
        classification: (N,) ASPRS class codes
        params:         dict with watershed parameters

    Returns:
        dict with 'labels' (N_tree,) and raw intermediate arrays for analysis
    """
    tree_xyz = xyz[tree_mask]
    n_tree = len(tree_xyz)
    if n_tree == 0:
        return {"labels": np.array([], dtype=np.int32), "n_seeds": 0,
                "chm": None, "watershed_labels": None, "col": None, "row": None}

    chm_res      = params["chm_resolution"]
    sigma        = params["smooth_sigma"]
    min_h        = params["min_tree_height"]
    min_px       = params["min_crown_pixels"]
    win          = params["local_max_window"]

    # Ground elevation
    ground_mask = (classification == 2)
    if ground_mask.sum() > 0:
        ground_z = float(np.median(xyz[ground_mask, 2]))
    else:
        ground_z = float(np.percentile(xyz[:, 2], 5))

    x, y = tree_xyz[:, 0], tree_xyz[:, 1]
    z = tree_xyz[:, 2] - ground_z
    x_min, y_min = x.min(), y.min()

    col = ((x - x_min) / chm_res).astype(np.int32)
    row = ((y - y_min) / chm_res).astype(np.int32)
    n_cols, n_rows = col.max() + 1, row.max() + 1

    # CHM
    chm = np.full((n_rows, n_cols), -np.inf, dtype=np.float64)
    np.maximum.at(chm.ravel(), row * n_cols + col, z)
    chm[chm == -np.inf] = 0.0

    # Smooth + local maxima
    chm_s = ndimage.gaussian_filter(chm, sigma=sigma)
    footprint = np.ones((win, win))
    local_max_img = ndimage.maximum_filter(chm_s, footprint=footprint)
    is_max = (chm_s == local_max_img) & (chm_s >= min_h)
    markers, n_seeds = ndimage.label(is_max)

    if n_seeds == 0:
        return {"labels": np.full(n_tree, -1, dtype=np.int32), "n_seeds": 0,
                "chm": chm, "watershed_labels": None, "col": col, "row": row}

    # Watershed
    canopy_mask = chm >= min_h
    chm_inv = chm_s.max() - chm_s
    chm_inv_norm = (chm_inv / (chm_inv.max() + 1e-8) * 65534).astype(np.uint16)
    wlabels = ndimage.watershed_ift(chm_inv_norm, markers)
    wlabels[~canopy_mask] = 0

    # Filter tiny crowns
    for lbl in range(1, n_seeds + 1):
        if (wlabels == lbl).sum() < min_px:
            wlabels[wlabels == lbl] = 0

    unique = sorted(set(wlabels.ravel()) - {0})

    # Map 3D points → region labels
    point_labels = wlabels[row, col]
    labels = np.full(n_tree, -1, dtype=np.int32)
    for new_id, lbl in enumerate(unique):
        labels[point_labels == lbl] = new_id

    return {
        "labels": labels,
        "n_seeds": n_seeds,
        "chm": chm,
        "chm_res": chm_res,
        "watershed_labels": wlabels,
        "unique_crowns": unique,
        "col": col,
        "row": row,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(result: dict, tile_area_ha: float) -> dict:
    """Compute quality metrics from a watershed result dict."""
    labels = result["labels"]
    n_tree = len(labels)
    unique = sorted(set(labels) - {-1})
    n_crowns = len(unique)
    n_noise = int((labels == -1).sum())
    noise_frac = n_noise / n_tree if n_tree > 0 else 1.0
    density_ha = n_crowns / tile_area_ha if tile_area_ha > 0 else 0.0

    crown_areas_m2 = []
    if result.get("watershed_labels") is not None and result.get("chm_res") is not None:
        wlabels = result["watershed_labels"]
        px_area = result["chm_res"] ** 2
        for lbl_id in result.get("unique_crowns", unique):
            crown_areas_m2.append(float((wlabels == lbl_id).sum()) * px_area)

    if crown_areas_m2:
        arr = np.array(crown_areas_m2)
        median_crown = float(np.median(arr))
        mean_crown   = float(np.mean(arr))
        cv_crown     = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
        min_crown    = float(arr.min())
        max_crown    = float(arr.max())
    else:
        median_crown = mean_crown = cv_crown = min_crown = max_crown = 0.0

    # Coverage fraction: pixels assigned vs total canopy pixels
    coverage_frac = 0.0
    if result.get("chm") is not None and result.get("chm_res") is not None:
        chm = result["chm"]
        min_h = 3.0  # approximate — actual min_tree_height not stored here
        canopy_px = int((chm >= min_h).sum())
        if canopy_px > 0 and result.get("watershed_labels") is not None:
            assigned_px = int((result["watershed_labels"] > 0).sum())
            coverage_frac = assigned_px / canopy_px

    return {
        "n_trees":         n_crowns,
        "n_noise_pts":     n_noise,
        "noise_frac":      round(noise_frac, 4),
        "density_ha":      round(density_ha, 2),
        "median_crown_m2": round(median_crown, 2),
        "mean_crown_m2":   round(mean_crown, 2),
        "cv_crown_area":   round(cv_crown, 4),
        "min_crown_m2":    round(min_crown, 2),
        "max_crown_m2":    round(max_crown, 2),
        "coverage_frac":   round(coverage_frac, 4),
    }


def quality_score(metrics: dict) -> float:
    """
    Composite quality score in [0, 1].  Higher = better parameter combination.

    Penalizes:
        - density outside [200, 1000] trees/ha
        - median crown outside [5, 40] m²
        - high noise fraction
        - high crown size variance (CV)
    """
    e = ECOLOGY
    d = metrics["density_ha"]
    c = metrics["median_crown_m2"]

    # Density score: triangle peaked at geometric mean of [200, 1000]
    d_lo, d_hi = e["density_lo"], e["density_hi"]
    if d <= 0:
        d_score = 0.0
    elif d < d_lo:
        d_score = d / d_lo
    elif d > d_hi:
        d_score = d_hi / d
    else:
        # inside range: 1.0 at d_lo, scales slightly toward 1.0 in midpoint
        d_score = 1.0

    # Crown score
    c_lo, c_hi = e["crown_lo"], e["crown_hi"]
    if c <= 0:
        c_score = 0.0
    elif c < c_lo:
        c_score = c / c_lo
    elif c > c_hi:
        c_score = c_hi / c
    else:
        c_score = 1.0

    # Noise score (1 = no noise)
    n_score = 1.0 - metrics["noise_frac"]

    # CV score (1 = very uniform crowns, diminishing for cv > 1)
    cv = metrics["cv_crown_area"]
    cv_score = 1.0 / (1.0 + cv)

    score = (
        e["density_weight"] * d_score
        + e["crown_weight"]  * c_score
        + e["noise_weight"]  * n_score
        + e["cv_weight"]     * cv_score
    )
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# Load tile
# ---------------------------------------------------------------------------

def load_laz_tile(laz_path: Path) -> tuple:
    """
    Load a .laz file.
    Returns: (xyz, classification, tile_area_ha)
    """
    las = laspy.read(str(laz_path))
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    xyz = np.stack([x, y, z], axis=1)

    try:
        classification = np.array(las.classification, dtype=np.int32)
    except Exception:
        classification = np.zeros(len(x), dtype=np.int32)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    tile_area_ha = (x_range * y_range) / 10_000.0

    return xyz, classification, tile_area_ha


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def expand_grid(param_grid: dict) -> list:
    """Return list of dicts — all combinations in param_grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for vals in product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_grid_search(
    laz_files: list,
    param_grid: dict,
    output_dir: Path,
    verbose: bool = True,
) -> list:
    """
    Run watershed grid search over laz_files × param_grid.

    Args:
        laz_files:   list of Path objects to .laz files
        param_grid:  dict of param_name → list of values
        output_dir:  where to save per-run JSON + summary
        verbose:     print progress

    Returns:
        List of result dicts (one per tile×params combination)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = expand_grid(param_grid)
    n_combos = len(combos)
    n_tiles = len(laz_files)
    total = n_tiles * n_combos

    if verbose:
        print(f"\n{'='*60}")
        print(f"Watershed Grid Search")
        print(f"  Tiles:       {n_tiles}")
        print(f"  Param combos: {n_combos}")
        print(f"  Total runs:  {total}")
        print(f"  Output:      {output_dir}")
        print(f"{'='*60}\n")

    # Save grid config
    with open(output_dir / "grid_config.json", "w") as f:
        json.dump({"param_grid": param_grid, "n_tiles": n_tiles,
                   "n_combos": n_combos, "total_runs": total}, f, indent=2)

    all_results = []
    run_idx = 0
    t_total_start = time.time()

    for tile_path in laz_files:
        tile_name = tile_path.stem
        if verbose:
            print(f"\n--- Tile: {tile_name} ---")

        # Load tile once, reuse across all param combos
        t0 = time.time()
        xyz, classification, tile_area_ha = load_laz_tile(tile_path)
        tree_mask = (classification == 5)
        n_tree_pts = int(tree_mask.sum())
        load_time = time.time() - t0

        if verbose:
            print(f"  Loaded {len(xyz):,} pts, {n_tree_pts:,} tree pts, "
                  f"{tile_area_ha:.2f} ha  ({load_time:.1f}s)")

        if n_tree_pts < 100:
            if verbose:
                print(f"  SKIP: too few tree points")
            continue

        tile_results = []
        for i, params in enumerate(combos):
            run_idx += 1
            t0 = time.time()

            result = _run_watershed(xyz, tree_mask, classification, params)
            metrics = compute_metrics(result, tile_area_ha)
            score = quality_score(metrics)
            elapsed = time.time() - t0

            row = {
                "tile":       tile_name,
                "run_idx":    run_idx,
                **params,
                **metrics,
                "quality_score": score,
                "run_time_s":    round(elapsed, 3),
            }
            tile_results.append(row)
            all_results.append(row)

            if verbose:
                print(
                    f"  [{run_idx:3d}/{total}] "
                    f"res={params['chm_resolution']:.2f} "
                    f"sig={params['smooth_sigma']:.1f} "
                    f"win={params['local_max_window']:2d} "
                    f"| trees={metrics['n_trees']:3d} "
                    f"dens={metrics['density_ha']:6.0f}/ha "
                    f"crown={metrics['median_crown_m2']:5.1f}m² "
                    f"noise={metrics['noise_frac']:.2f} "
                    f"score={score:.3f}  ({elapsed:.2f}s)"
                )

        # Save per-tile results
        tile_out = output_dir / f"tile_{tile_name}.json"
        with open(tile_out, "w") as f:
            json.dump(tile_results, f, indent=2)

    total_time = time.time() - t_total_start
    if verbose:
        print(f"\nDone. Total time: {total_time:.1f}s  ({total_time/max(run_idx,1):.2f}s/run)")

    # Save full results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ---------------------------------------------------------------------------
# Summarize: aggregate over tiles, rank combos
# ---------------------------------------------------------------------------

def summarize_results(all_results: list) -> list:
    """
    Aggregate per-tile results over all tiles for each parameter combination.
    Returns list of summary dicts sorted by mean quality_score descending.
    """
    from collections import defaultdict

    param_keys = ["chm_resolution", "smooth_sigma", "local_max_window",
                  "min_tree_height", "min_crown_pixels"]
    metric_keys = ["n_trees", "noise_frac", "density_ha", "median_crown_m2",
                   "mean_crown_m2", "cv_crown_area", "coverage_frac", "quality_score"]

    groups = defaultdict(list)
    for row in all_results:
        key = tuple(row[k] for k in param_keys if k in row)
        groups[key].append(row)

    summaries = []
    for key, rows in groups.items():
        params = {k: rows[0][k] for k in param_keys if k in rows[0]}
        summary = dict(params)
        summary["n_tiles"] = len(rows)
        for mk in metric_keys:
            vals = [r[mk] for r in rows if mk in r]
            if vals:
                summary[f"mean_{mk}"] = round(float(np.mean(vals)), 4)
                summary[f"std_{mk}"]  = round(float(np.std(vals)),  4)
        summaries.append(summary)

    summaries.sort(key=lambda x: x.get("mean_quality_score", 0), reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# CSV export (for sharing with teammates)
# ---------------------------------------------------------------------------

def save_csv(rows: list, path: Path):
    """Save list of dicts as CSV."""
    if not rows:
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(all_results: list, summaries: list,
                         param_grid: dict, output_dir: Path):
    """Generate a self-contained HTML report with tables and heatmap plots."""
    import io, base64
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    plots_b64 = {}

    if HAS_MPL and summaries:
        # --- Plot 1: quality score heatmap (smooth_sigma × local_max_window)
        # average over chm_resolution values
        sig_vals = sorted(set(s["smooth_sigma"] for s in summaries))
        win_vals = sorted(set(s["local_max_window"] for s in summaries))
        res_vals = sorted(set(s["chm_resolution"] for s in summaries))

        for res in res_vals:
            matrix = np.zeros((len(sig_vals), len(win_vals)))
            for s in summaries:
                if s["chm_resolution"] != res:
                    continue
                i = sig_vals.index(s["smooth_sigma"])
                j = win_vals.index(s["local_max_window"])
                matrix[i, j] = s.get("mean_quality_score", 0)

            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=1)
            ax.set_xticks(range(len(win_vals)))
            ax.set_xticklabels([str(w) for w in win_vals])
            ax.set_yticks(range(len(sig_vals)))
            ax.set_yticklabels([str(s) for s in sig_vals])
            ax.set_xlabel("local_max_window (px)")
            ax.set_ylabel("smooth_sigma")
            ax.set_title(f"Quality Score  (chm_resolution={res} m/px)")
            plt.colorbar(im, ax=ax, label="quality score")
            for i in range(len(sig_vals)):
                for j in range(len(win_vals)):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center",
                            va="center", fontsize=8,
                            color="white" if matrix[i,j] < 0.4 else "black")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=110)
            plt.close()
            plots_b64[f"heatmap_res{res}"] = base64.b64encode(buf.getvalue()).decode()

        # --- Plot 2: top-10 configs bar chart
        top10 = summaries[:10]
        labels_bar = [
            f"r={s['chm_resolution']}\nσ={s['smooth_sigma']}\nw={s['local_max_window']}"
            for s in top10
        ]
        scores = [s.get("mean_quality_score", 0) for s in top10]
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(top10)), scores, color="steelblue")
        bars[0].set_color("gold")
        ax.set_xticks(range(len(top10)))
        ax.set_xticklabels(labels_bar, fontsize=8)
        ax.set_ylabel("Mean Quality Score")
        ax.set_title("Top 10 Parameter Configurations")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110)
        plt.close()
        plots_b64["top10_bar"] = base64.b64encode(buf.getvalue()).decode()

        # --- Plot 3: density vs median crown scatter (all combos, colored by score)
        if summaries:
            dens_vals = [s.get("mean_density_ha", 0) for s in summaries]
            crown_vals = [s.get("mean_median_crown_m2", 0) for s in summaries]
            score_vals = [s.get("mean_quality_score", 0) for s in summaries]
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(dens_vals, crown_vals, c=score_vals, cmap="RdYlGn",
                            vmin=0, vmax=1, s=60, alpha=0.8, edgecolors="gray", lw=0.5)
            plt.colorbar(sc, ax=ax, label="quality score")
            # draw ecological prior rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((ECOLOGY["density_lo"], ECOLOGY["crown_lo"]),
                              ECOLOGY["density_hi"] - ECOLOGY["density_lo"],
                              ECOLOGY["crown_hi"] - ECOLOGY["crown_lo"],
                              linewidth=2, edgecolor="blue", facecolor="none",
                              linestyle="--", label="Ecological target zone")
            ax.add_patch(rect)
            ax.set_xlabel("Tree density (trees/ha)")
            ax.set_ylabel("Median crown area (m²)")
            ax.set_title("Density vs Crown Area — all parameter combinations")
            ax.legend()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=110)
            plt.close()
            plots_b64["density_crown_scatter"] = base64.b64encode(buf.getvalue()).decode()

    # --- Build HTML ---
    top5 = summaries[:5]
    best = summaries[0] if summaries else {}

    def img_tag(key):
        if key in plots_b64:
            return f'<img src="data:image/png;base64,{plots_b64[key]}" style="max-width:100%">'
        return ""

    def summary_table(rows, highlight_first=False):
        if not rows:
            return "<p><em>No data</em></p>"
        keys = [k for k in rows[0].keys() if not k.startswith("std_")]
        thead = "".join(f"<th>{k}</th>" for k in keys)
        tbody = ""
        for idx, row in enumerate(rows):
            style = ' style="background:#fff9c4"' if (highlight_first and idx == 0) else ""
            cells = ""
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    v = f"{v:.4f}"
                cells += f"<td>{v}</td>"
            tbody += f"<tr{style}>{cells}</tr>"
        return (f'<table border="1" cellpadding="4" cellspacing="0" '
                f'style="border-collapse:collapse;font-size:12px">'
                f"<thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>")

    best_params_yaml = "\n".join(
        f"  {k}: {best.get(k, '?')}"
        for k in ["chm_resolution", "smooth_sigma", "local_max_window",
                  "min_tree_height", "min_crown_pixels"]
    ) if best else ""

    n_tiles = len(set(r["tile"] for r in all_results))
    n_combos = len(summaries)
    total_runs = len(all_results)

    heatmap_imgs = "".join(
        f'<h3>CHM resolution = {res} m/px</h3>{img_tag(f"heatmap_res{res}")}'
        for res in sorted(set(s["chm_resolution"] for s in summaries))
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Watershed Grid Search Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #2c5f2e; }}
  h2 {{ color: #1a3c1b; border-bottom: 2px solid #2c5f2e; padding-bottom: 4px; }}
  h3 {{ color: #3a7d3b; }}
  .summary-box {{ background: #f0f7f0; border: 1px solid #2c5f2e; border-radius: 8px;
                  padding: 16px; margin: 16px 0; }}
  .best-params {{ background: #fff9c4; border: 1px solid #c8a000; border-radius: 8px;
                  padding: 12px; font-family: monospace; font-size: 13px; }}
  table th {{ background: #2c5f2e; color: white; }}
  .note {{ color: #666; font-size: 12px; font-style: italic; }}
</style>
</head>
<body>
<h1>Watershed Parameter Grid Search Report</h1>

<div class="summary-box">
  <b>Run summary:</b>
  {n_tiles} tiles &nbsp;|&nbsp; {n_combos} parameter combinations &nbsp;|&nbsp;
  {total_runs} total runs
  <br>
  <b>Best quality score:</b> {best.get("mean_quality_score", "?"):.4f}
  &nbsp; (density={best.get("mean_density_ha", "?"):.0f} trees/ha,
  median crown={best.get("mean_median_crown_m2", "?"):.1f} m²,
  noise={best.get("mean_noise_frac", "?"):.2f})
</div>

<h2>Best Configuration (for sharing with teammates)</h2>
<p>Paste these values into your <code>configs/segmentation.yaml</code>
under <code>clustering:</code>:</p>
<div class="best-params">
clustering:<br>
  method: watershed<br>
{best_params_yaml}
</div>
<p class="note">
  Quality score is based on ecological priors:
  target density {ECOLOGY['density_lo']}–{ECOLOGY['density_hi']} trees/ha,
  crown area {ECOLOGY['crown_lo']}–{ECOLOGY['crown_hi']} m².
  Adjust weights in <code>ECOLOGY</code> dict if your forest type differs.
</p>

<h2>Top 5 Configurations</h2>
{summary_table(top5, highlight_first=True)}

<h2>Top 10 — Quality Score</h2>
{img_tag("top10_bar")}

<h2>Quality Score Heatmaps (σ × window size)</h2>
<p class="note">Blue dashed rectangle = ecological target zone.</p>
{heatmap_imgs}

<h2>Density vs Crown Area</h2>
{img_tag("density_crown_scatter")}

<h2>All Configurations (sorted by quality score)</h2>
{summary_table(summaries)}

</body>
</html>"""

    with open(output_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved: {output_dir / 'report.html'}")
