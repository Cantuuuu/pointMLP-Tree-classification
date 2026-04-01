"""
Flow A — Geometric Baseline for TLS Tree Segmentation.

Uses ONLY raw XYZ coordinates. No semantic labels, no trained model.
Designed for terrestrial LiDAR (TLS) where the primary discriminator
between tree and non-tree is Height Above Ground (HAG).

Physical rationale:
  - TLS non-tree points (terrain + low vegetation) concentrate near z=0
    after ground normalisation: 95th percentile < 0.4 m across all sites.
  - TLS tree points (crown + stem) are well above ground: 5th percentile
    ranges from 3 m (RMIT, shorter trees) to 11 m (CULS, tall broadleaf).
  - HAG is estimated from the local 5th-percentile Z in 1 m x 1 m ground
    cells — no classification labels used.

Algorithm (per plot):
  1. Partition the plot into 1 m × 1 m XY cells.
  2. For each cell, estimate ground elevation = 5th-percentile Z of all
     points in that cell (the ground hugs the minimum Z).
  3. HAG for every point = Z - cell_ground_z.
  4. Classify as TREE if HAG > hag_threshold.

Public API
----------
predict_plot(xyz, hag_threshold, cell_size) -> pred_mask (N,) bool
compute_hag(xyz, cell_size) -> hag (N,) float32
segmentation_metrics(pred_mask, gt_labels) -> dict
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# HAG estimation
# ---------------------------------------------------------------------------

def compute_hag(xyz: np.ndarray, cell_size: float = 1.0) -> np.ndarray:
    """Compute Height Above Ground (HAG) per point.

    Partitions the scene into `cell_size` × `cell_size` metre XY cells.
    Ground elevation per cell = 5th-percentile Z (purely geometric, no
    classification labels).

    Parameters
    ----------
    xyz       : (N, 3) float — raw coordinates in metres
    cell_size : XY cell size in metres for ground estimation

    Returns
    -------
    hag : (N,) float32 — height above local ground in metres
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_min, y_min = x.min(), y.min()

    cx = ((x - x_min) / cell_size).astype(np.int32)
    cy = ((y - y_min) / cell_size).astype(np.int32)
    ny = int(cy.max()) + 2
    cell_id = cx * ny + cy

    hag = np.empty(len(xyz), dtype=np.float32)
    for cid in np.unique(cell_id):
        m = cell_id == cid
        ground_z = float(np.percentile(z[m], 5))
        hag[m] = z[m] - ground_z

    return hag


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_plot(xyz: np.ndarray, hag_threshold: float,
                 cell_size: float = 1.0) -> np.ndarray:
    """Predict tree/non-tree mask for a full TLS plot.

    Parameters
    ----------
    xyz           : (N, 3) raw coordinates in metres (Out-points already removed)
    hag_threshold : minimum HAG to classify a point as tree (metres)
    cell_size     : cell size for HAG ground estimation (metres)

    Returns
    -------
    pred_mask : (N,) bool — True = tree
    """
    hag = compute_hag(xyz, cell_size=cell_size)
    return hag > hag_threshold


# ---------------------------------------------------------------------------
# Threshold tuning (dev set only)
# ---------------------------------------------------------------------------

def tune_threshold(xyz: np.ndarray, gt_labels: np.ndarray,
                   hag_range: list[float],
                   cell_size: float = 1.0) -> dict:
    """Find the best HAG threshold on a single dev scene.

    Optimises for tree-class IoU (intersection-over-union).

    Parameters
    ----------
    xyz        : (N, 3) raw coordinates in metres
    gt_labels  : (N,) int — 1 = tree, 0 = non-tree
    hag_range  : list of HAG thresholds to evaluate (metres)
    cell_size  : HAG estimation cell size

    Returns
    -------
    dict with keys: best_threshold, best_iou, results (list per threshold)
    """
    hag = compute_hag(xyz, cell_size=cell_size)
    gt  = gt_labels.astype(bool)

    results = []
    for thr in hag_range:
        pred = hag > thr
        tp = int(( pred &  gt).sum())
        fp = int(( pred & ~gt).sum())
        fn = int((~pred &  gt).sum())
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            "threshold": float(thr),
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "tp": tp, "fp": fp, "fn": fn,
        })

    best = max(results, key=lambda r: r["iou"])
    return {"best_threshold": best["threshold"], "best_iou": best["iou"],
            "results": results}


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def segmentation_metrics(pred_mask: np.ndarray,
                          gt_labels: np.ndarray) -> dict:
    """Binary segmentation metrics for the tree class."""
    pred = pred_mask.astype(bool)
    gt   = gt_labels.astype(bool)

    tp = int(( pred &  gt).sum())
    fp = int(( pred & ~gt).sum())
    fn = int((~pred &  gt).sum())
    tn = int((~pred & ~gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "iou":       float(iou),
        "accuracy":  float(accuracy),
    }
