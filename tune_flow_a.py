"""
Tune Flow A (Geometric Baseline) HAG threshold on FORinstance dev plots.

Loads each dev .las file, excludes Out-points (class 3), runs a grid search
over hag_threshold values, and reports per-plot and aggregate IoU / F1.
The best threshold is saved back into the config file.

Usage:
    python tune_flow_a.py [--config configs/segmentation_forinstance.yaml]
"""

import sys
import json
import argparse
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.segmentation.flow_a_geometric import compute_hag, tune_threshold, segmentation_metrics

OUT_POINTS_CLASS = 3


def load_plot(las_path: Path) -> dict | None:
    if not las_path.exists():
        return None
    las  = laspy.read(str(las_path))
    xyz  = np.stack([np.array(las.x), np.array(las.y), np.array(las.z)], axis=-1).astype(np.float64)
    clf  = np.array(las.classification, dtype=np.int32)
    tid  = np.array(las["treeID"], dtype=np.int32)
    # Exclude Out-points
    keep = clf != OUT_POINTS_CLASS
    return {"xyz": xyz[keep], "clf": clf[keep], "tree_id": tid[keep]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation_forinstance.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(cfg["data"]["forinstance_dir"])
    meta        = pd.read_csv(dataset_dir / "data_split_metadata.csv")
    dev_plots   = meta[meta["split"] == "dev"]

    hag_range = cfg["flow_a"]["hag_range"]
    cell_size = 1.0   # 1m ground cells (fixed)

    print(f"Grid search HAG thresholds: {hag_range}")
    print(f"Dev plots: {len(dev_plots)}\n")

    # Aggregate accumulators: tp/fp/fn per threshold across all dev points
    accum = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in hag_range}
    per_plot = []

    for _, row in dev_plots.iterrows():
        las_path = dataset_dir / row["path"]
        if not las_path.exists():
            print(f"  [SKIP] {row['path']}")
            continue

        plot_name = Path(row["path"]).stem
        site      = row["folder"]
        print(f"  {site}/{plot_name} ... ", end="", flush=True)

        plot = load_plot(las_path)
        xyz  = plot["xyz"].astype(np.float32)
        gt   = (plot["tree_id"] > 0).astype(np.int64)

        n_tree    = int(gt.sum())
        n_total   = len(gt)
        tree_frac = n_tree / n_total if n_total > 0 else 0

        # Pre-compute HAG once for all thresholds
        hag = compute_hag(xyz, cell_size=cell_size)

        plot_results = []
        for thr in hag_range:
            pred = hag > thr
            tp   = int(( pred &  gt.astype(bool)).sum())
            fp   = int(( pred & ~gt.astype(bool)).sum())
            fn   = int((~pred &  gt.astype(bool)).sum())
            iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            plot_results.append({
                "threshold": float(thr), "iou": iou,
                "f1": f1, "precision": prec, "recall": rec,
            })
            accum[thr]["tp"] += tp
            accum[thr]["fp"] += fp
            accum[thr]["fn"] += fn

        best_r = max(plot_results, key=lambda r: r["iou"])
        print(f"{n_total:,} pts (tree {tree_frac:.2f}) | "
              f"best thr={best_r['threshold']:.1f} m → "
              f"IoU={best_r['iou']:.3f} F1={best_r['f1']:.3f}")

        per_plot.append({
            "site": site, "plot": plot_name,
            "n_pts": n_total, "tree_frac": float(tree_frac),
            "results": plot_results,
            "best_threshold": best_r["threshold"],
            "best_iou": best_r["iou"],
            "best_f1": best_r["f1"],
        })
        del plot, hag, pred

    # --- Aggregate metrics (macro-average over plots) ---
    print("\n" + "=" * 65)
    print(f"{'Threshold':>12} | {'IoU':>7} | {'F1':>7} | {'Precision':>10} | {'Recall':>7}")
    print("-" * 65)

    agg_results = []
    for thr in hag_range:
        tp = accum[thr]["tp"]
        fp = accum[thr]["fp"]
        fn = accum[thr]["fn"]
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{thr:>11.1f}m | {iou:>7.4f} | {f1:>7.4f} | {prec:>10.4f} | {rec:>7.4f}")
        agg_results.append({
            "threshold": float(thr), "iou": iou,
            "f1": f1, "precision": prec, "recall": rec,
        })

    best_agg = max(agg_results, key=lambda r: r["iou"])
    print("=" * 65)
    print(f"\nBest threshold (aggregate IoU): {best_agg['threshold']:.1f} m")
    print(f"  IoU={best_agg['iou']:.4f}  F1={best_agg['f1']:.4f}  "
          f"Precision={best_agg['precision']:.4f}  Recall={best_agg['recall']:.4f}")

    # --- Save results ---
    out_dir = Path(cfg["experiment"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "flow_a_tuning.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_threshold": best_agg["threshold"],
            "best_iou": best_agg["iou"],
            "best_f1": best_agg["f1"],
            "aggregate": agg_results,
            "per_plot": per_plot,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Update config with best threshold ---
    with open(args.config) as f:
        config_text = f.read()

    old_line = f"hag_threshold:      {cfg['flow_a']['hag_threshold']}"
    new_line = f"hag_threshold:      {best_agg['threshold']}"
    if old_line in config_text:
        config_text = config_text.replace(old_line, new_line)
        with open(args.config, "w") as f:
            f.write(config_text)
        print(f"Config updated: flow_a.hag_threshold = {best_agg['threshold']} m")
    else:
        print(f"[WARN] Could not auto-update config. Set flow_a.hag_threshold = {best_agg['threshold']} manually.")


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
