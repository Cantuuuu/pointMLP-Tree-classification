"""
Instance segmentation via 2D Watershed on the Z-max raster.

Pipeline (mirrors CHM-Watershed used in ALS/UAV-LS ITD literature):
  1. Build Z-max raster from predicted-tree points  (same as CHM step)
  2. Gaussian smoothing                             (suppress branch noise)
  3. Detect local maxima → seeds                   (one seed per tree top)
  4. Watershed from seeds on the inverted raster    (grow regions downward)
  5. Assign each 3D point to the instance whose seed
     is nearest in the XY raster                   (back-projection)

The result is an integer array instance_ids (N,) where:
  0  = non-tree (pred_mask == False)
 -1  = tree point not covered by any watershed region (border artefact)
 >0  = tree instance ID (1-indexed, matching a specific tree top seed)

Public API
----------
segment_instances(xyz, pred_mask, cfg)  -> instance_ids (N,)
evaluate_instances(pred_ids, gt_ids)    -> dict with PQ, mIoU, TP/FP/FN
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label as nd_label
from skimage.segmentation import watershed as skimage_watershed


# ──────────────────────────────────────────────────────────────────────────────
# Step 1-4 : build raster + watershed labels
# ──────────────────────────────────────────────────────────────────────────────

def _build_zmax_raster(xyz_tree: np.ndarray, cell_size: float):
    """Build Z-max raster from predicted tree points.

    Returns
    -------
    z_raster : (nx, ny) float32  — Z-max per cell (NaN where no points)
    x_min, y_min : float         — raster origin in metres
    """
    if len(xyz_tree) == 0:
        return np.zeros((1, 1), dtype=np.float32), 0.0, 0.0

    x, y, z = xyz_tree[:, 0], xyz_tree[:, 1], xyz_tree[:, 2]
    x_min = float(x.min()); y_min = float(y.min())

    nx = int((x.max() - x_min) / cell_size) + 2
    ny = int((y.max() - y_min) / cell_size) + 2

    z_raster = np.full((nx, ny), np.nan, dtype=np.float32)
    cx = ((x - x_min) / cell_size).astype(np.int32)
    cy = ((y - y_min) / cell_size).astype(np.int32)

    z_order = np.argsort(-z)                         # descending Z
    z_raster[cx[z_order], cy[z_order]] = z[z_order]  # first = max

    return z_raster, x_min, y_min


def _detect_seeds(z_raster: np.ndarray, smooth_sigma: float,
                  local_max_window: int, z_fill: float):
    """Gaussian smooth + 2-D local maxima → seed marker array for watershed."""
    z_filled = np.where(np.isnan(z_raster), z_fill, z_raster)
    z_smooth = gaussian_filter(z_filled.astype(np.float64), sigma=smooth_sigma)

    z_max_filt = maximum_filter(z_smooth, size=local_max_window)
    local_max  = (z_smooth == z_max_filt) & (~np.isnan(z_raster))

    # Label each local maximum as a unique seed
    markers, n_seeds = nd_label(local_max)
    return markers, z_smooth, n_seeds


def _run_watershed(z_smooth: np.ndarray, markers: np.ndarray,
                   z_raster: np.ndarray) -> np.ndarray:
    """Apply scikit-image watershed on the inverted Z-smooth raster.

    Watershed fills from local minima outward; inverting Z-smooth turns
    tree-top maxima into minima, so each tree crown becomes one basin.
    Cells with NaN in z_raster (no point data) are masked out.
    """
    mask      = ~np.isnan(z_raster)
    z_inv     = -z_smooth                           # invert so tops = minima
    ws_labels = skimage_watershed(z_inv, markers=markers, mask=mask)
    return ws_labels                                 # (nx, ny) int, 0=background


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 : back-project raster labels to 3D points
# ──────────────────────────────────────────────────────────────────────────────

def _assign_labels_to_points(xyz_tree: np.ndarray, ws_labels: np.ndarray,
                              x_min: float, y_min: float,
                              cell_size: float) -> np.ndarray:
    """Map each 3D tree point to the watershed region it falls in.

    Returns instance_ids_tree (len(xyz_tree),) int — 0 means background/edge.
    """
    cx = ((xyz_tree[:, 0] - x_min) / cell_size).astype(np.int32)
    cy = ((xyz_tree[:, 1] - y_min) / cell_size).astype(np.int32)

    nx, ny = ws_labels.shape
    cx = np.clip(cx, 0, nx - 1)
    cy = np.clip(cy, 0, ny - 1)

    return ws_labels[cx, cy].astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Public: full instance segmentation
# ──────────────────────────────────────────────────────────────────────────────

def segment_instances(xyz: np.ndarray, pred_mask: np.ndarray,
                      cell_size: float = 0.5,
                      smooth_sigma: float = 1.0,
                      local_max_window: int = 5) -> np.ndarray:
    """Full instance segmentation via 2D Watershed on Z-max raster.

    Parameters
    ----------
    xyz            : (N, 3) raw coordinates in metres (absolute, not normalised)
    pred_mask      : (N,) bool — True for points predicted as tree
    cell_size      : raster resolution in metres
    smooth_sigma   : Gaussian smoothing (cells) to suppress branch noise
    local_max_window: local-maxima window (cells) — ~crown diameter / cell_size

    Returns
    -------
    instance_ids : (N,) int
        0  = non-tree point
       -1  = tree point outside all watershed regions (rare border case)
       >0  = tree instance ID
    """
    n = len(xyz)
    instance_ids = np.zeros(n, dtype=np.int32)

    pred_mask_arr = np.asarray(pred_mask, dtype=bool)[:n]
    tree_idx      = np.where(pred_mask_arr)[0]
    if len(tree_idx) == 0:
        return instance_ids

    xyz_tree = xyz[tree_idx]

    # 1. Z-max raster
    z_raster, x_min, y_min = _build_zmax_raster(xyz_tree, cell_size)
    z_fill = float(np.nanmin(z_raster)) if not np.all(np.isnan(z_raster)) else 0.0

    # 2-3. Seeds via local maxima
    markers, z_smooth, n_seeds = _detect_seeds(
        z_raster, smooth_sigma, local_max_window, z_fill)

    if n_seeds == 0:
        instance_ids[tree_idx] = -1
        return instance_ids

    # 4. Watershed
    ws_labels = _run_watershed(z_smooth, markers, z_raster)

    # 5. Back-project to 3D points
    labels_tree = _assign_labels_to_points(
        xyz_tree, ws_labels, x_min, y_min, cell_size)

    instance_ids[tree_idx] = np.where(labels_tree > 0, labels_tree, -1)
    return instance_ids


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation: Panoptic Quality + instance IoU
# ──────────────────────────────────────────────────────────────────────────────

def _instance_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    inter = (mask_pred & mask_gt).sum()
    union = (mask_pred | mask_gt).sum()
    return float(inter / union) if union > 0 else 0.0


def evaluate_instances(instance_ids: np.ndarray,
                       gt_tree_id: np.ndarray,
                       iou_threshold: float = 0.5) -> dict:
    """Evaluate predicted instance segmentation against GT treeID.

    Metrics
    -------
    Panoptic Quality (PQ) = SQ × RQ  (Kirillov et al., 2019)
      SQ (Segmentation Quality)  = mean IoU of matched TP pairs
      RQ (Recognition Quality)   = TP / (TP + 0.5·FP + 0.5·FN)

    Mean Instance IoU
      For each GT instance: IoU with the best-matching predicted instance.
      Averaged over all GT instances.

    Detection metrics (IoU threshold = iou_threshold)
      TP, FP, FN, Precision, Recall, F1

    Parameters
    ----------
    instance_ids  : (N,) predicted instance IDs (>0 = tree instance, ≤0 = other)
    gt_tree_id    : (N,) GT treeID per point (>0 = annotated tree)
    iou_threshold : minimum IoU to count a pair as TP (default 0.5)
    """
    pred_ids_set = set(int(i) for i in np.unique(instance_ids) if i > 0)
    gt_ids_set   = set(int(i) for i in np.unique(gt_tree_id)   if i > 0)

    if not pred_ids_set and not gt_ids_set:
        return _empty_result(0, 0, iou_threshold)
    if not pred_ids_set:
        return _empty_result(0, len(gt_ids_set), iou_threshold)
    if not gt_ids_set:
        return _empty_result(len(pred_ids_set), 0, iou_threshold)

    pred_ids = sorted(pred_ids_set)
    gt_ids   = sorted(gt_ids_set)

    # Build binary masks lazily via index lookups
    # IoU matrix  (P × G)
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for pi, pid in enumerate(pred_ids):
        pmask = instance_ids == pid
        for gi, gid in enumerate(gt_ids):
            gmask = gt_tree_id == gid
            iou_matrix[pi, gi] = _instance_iou(pmask, gmask)

    # Greedy matching (highest IoU first, threshold enforced)
    matched_pred = set(); matched_gt = set(); matched_pairs = []
    for flat_idx in np.argsort(-iou_matrix.ravel()):
        pi = flat_idx // len(gt_ids)
        gi = flat_idx %  len(gt_ids)
        if iou_matrix[pi, gi] < iou_threshold:
            break
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi); matched_gt.add(gi)
        matched_pairs.append((pred_ids[pi], gt_ids[gi],
                               float(iou_matrix[pi, gi])))

    tp = len(matched_pairs)
    fp = len(pred_ids) - tp
    fn = len(gt_ids)   - tp

    # SQ, RQ, PQ
    sq = float(np.mean([iou for _, _, iou in matched_pairs])) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    pq = sq * rq

    # Detection metrics
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Mean instance IoU (for each GT, best-matching pred IoU)
    per_gt_iou = []
    for gi in range(len(gt_ids)):
        best = float(iou_matrix[:, gi].max()) if len(pred_ids) > 0 else 0.0
        per_gt_iou.append(best)
    mean_inst_iou = float(np.mean(per_gt_iou)) if per_gt_iou else 0.0

    return {
        "n_pred": len(pred_ids),
        "n_gt":   len(gt_ids),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "sq":  round(sq,  4),
        "rq":  round(rq,  4),
        "pq":  round(pq,  4),
        "mean_inst_iou": round(mean_inst_iou, 4),
        "iou_threshold": iou_threshold,
        "matched_pairs": matched_pairs,
    }


def _empty_result(n_pred, n_gt, iou_threshold):
    fn = n_gt; fp = n_pred
    rq = 0 / (0 + 0.5*fp + 0.5*fn) if (fp + fn) > 0 else 0.0
    return {
        "n_pred": n_pred, "n_gt": n_gt,
        "tp": 0, "fp": fp, "fn": fn,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "sq": 0.0, "rq": float(rq), "pq": 0.0,
        "mean_inst_iou": 0.0,
        "iou_threshold": iou_threshold,
        "matched_pairs": [],
    }
