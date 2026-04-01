"""
Individual Tree Detection (ITD) via 3D DBSCAN clustering.

Replaces the CHM-watershed approach used for the NEON/ALS pipeline.
TLS point clouds have full 3D structure, so instance separation is done
directly in 3D space rather than on a 2D canopy height model.

Used identically for all three flows (Geometric / PointNet++ / RF) to
ensure a fair comparison of segmentation quality only.

Public API
----------
cluster_trees(xyz, pred_mask, cfg) -> instance_ids (N,)
evaluate_itd(pred_ids, gt_ids, match_dist) -> dict
compute_gt_centroids(xyz, tree_id_field) -> dict[int, np.ndarray]
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Z-raster local maxima ITD  (método principal para TLS)
# ---------------------------------------------------------------------------

def detect_tree_tops_raster(xyz_tree: np.ndarray,
                             cell_size: float = 0.5,
                             smooth_sigma: float = 1.5,
                             local_max_window: int = 5) -> np.ndarray:
    """Detecta copas de árboles individuales en una nube TLS como máximos locales
    de un raster Z.

    Analogía con el pipeline aéreo: en ALS se usa un Canopy Height Model (CHM)
    que registra la altura máxima por celda desde arriba. En TLS se aplica el
    mismo principio: el punto más alto de cada árbol visible es su cima, y el
    raster Z-max de puntos arbóreos predichos revela las cimas locales.

    Parámetros justificados:
      cell_size        : resolución del raster (m). 0.5 m es estándar en TLS ITD.
      smooth_sigma     : suavizado Gaussiano (celdas). Elimina falsos máximos por
                         ramas individuales; valor típico 1–2 celdas.
      local_max_window : tamaño ventana máximos locales (celdas). Una copa de árbol
                         típica TLS (2–6 m diámetro) → window = 3–7 celdas a 0.5 m.

    Returns
    -------
    tops : (K, 3) float — coordenadas XYZ de las K cimas predichas (metros)
    """
    if len(xyz_tree) == 0:
        return np.empty((0, 3), dtype=np.float32)

    x, y, z = xyz_tree[:, 0], xyz_tree[:, 1], xyz_tree[:, 2]
    x_min, y_min = float(x.min()), float(y.min())

    nx = int((x.max() - x_min) / cell_size) + 2
    ny = int((y.max() - y_min) / cell_size) + 2

    # Raster Z-max
    z_raster = np.full((nx, ny), np.nan, dtype=np.float32)
    cx = ((x - x_min) / cell_size).astype(np.int32)
    cy = ((y - y_min) / cell_size).astype(np.int32)

    # Vectorizado: ordenar por Z desc y usar índice para tomar máximo
    z_order = np.argsort(-z)
    z_raster[cx[z_order], cy[z_order]] = z[z_order]  # primero = máximo

    # Rellenar NaN con mínimo global (para no crear máximos artificiales)
    z_fill = np.nanmin(z_raster) if not np.all(np.isnan(z_raster)) else 0.0
    z_raster = np.where(np.isnan(z_raster), z_fill, z_raster)

    # Suavizado Gaussiano
    z_smooth = gaussian_filter(z_raster.astype(np.float64), sigma=smooth_sigma)

    # Máximos locales en 2D
    z_max_filt = maximum_filter(z_smooth, size=local_max_window)
    local_max_mask = (z_smooth == z_max_filt) & (z_raster > z_fill)

    tops_cx, tops_cy = np.where(local_max_mask)
    if len(tops_cx) == 0:
        return np.empty((0, 3), dtype=np.float32)

    tops_x = x_min + (tops_cx + 0.5) * cell_size
    tops_y = y_min + (tops_cy + 0.5) * cell_size
    tops_z = z_raster[tops_cx, tops_cy]

    return np.column_stack([tops_x, tops_y, tops_z]).astype(np.float32)


def match_tree_tops(pred_tops: np.ndarray, gt_centroids: dict[int, np.ndarray],
                    match_dist: float) -> dict:
    """Empareja cimas predichas con centroides GT por distancia horizontal mínima.

    Protocolo idéntico a match_trees() pero acepta pred_tops como array (K,3)
    en lugar de dict, ya que los máximos del raster no tienen ID previo.
    """
    if len(pred_tops) == 0:
        fn = len(gt_centroids)
        return {"tp": 0, "fp": 0, "fn": fn, "n_pred": 0, "n_gt": fn,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_pairs": []}
    if len(gt_centroids) == 0:
        return {"tp": 0, "fp": len(pred_tops), "fn": 0,
                "n_pred": len(pred_tops), "n_gt": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_pairs": []}

    gt_ids   = list(gt_centroids.keys())
    gt_arr   = np.array([gt_centroids[i][:2] for i in gt_ids])  # XY
    pred_arr = pred_tops[:, :2]                                   # XY

    dists = np.linalg.norm(pred_arr[:, None, :] - gt_arr[None, :, :], axis=-1)

    matched_pred = set()
    matched_gt   = set()
    matched_pairs = []

    flat_idx = np.argsort(dists.ravel())
    for idx in flat_idx:
        pi = idx // len(gt_ids)
        gi = idx  % len(gt_ids)
        if pi in matched_pred or gi in matched_gt:
            continue
        if dists[pi, gi] > match_dist:
            break
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_pairs.append((int(pi), gt_ids[gi], float(dists[pi, gi])))

    tp = len(matched_pairs)
    fp = len(pred_tops)    - tp
    fn = len(gt_centroids) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)
                 if (precision+recall) > 0 else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn,
            "n_pred": int(len(pred_tops)), "n_gt": int(len(gt_centroids)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "matched_pairs": matched_pairs}


def evaluate_itd_raster(xyz: np.ndarray, pred_mask: np.ndarray,
                         gt_tree_id: np.ndarray,
                         cell_size: float, smooth_sigma: float,
                         local_max_window: int, match_dist: float) -> dict:
    """Evaluación ITD completa usando el método raster Z-max.

    GT reference: apex (highest-Z point per tree), no centroide.
    Justificación: el detector raster Z-max localiza el punto más alto
    por región; la comparación natural es el punto más alto por árbol GT.
    Protocolo estándar en ITD con CHM (Chen et al. 2006, Jakubowski 2013).
    """
    tree_mask = np.asarray(pred_mask, dtype=bool)[:len(xyz)]
    xyz_tree  = xyz[tree_mask]

    pred_tops  = detect_tree_tops_raster(
        xyz_tree, cell_size, smooth_sigma, local_max_window)
    gt_apexes  = compute_gt_apexes(xyz, gt_tree_id)

    return match_tree_tops(pred_tops, gt_apexes, match_dist)


# ---------------------------------------------------------------------------
# DBSCAN-based clustering  (mantenido por compatibilidad)
# ---------------------------------------------------------------------------

def cluster_trees(xyz: np.ndarray, pred_mask: np.ndarray,
                  eps: float, min_samples: int,
                  min_cluster_points: int) -> np.ndarray:
    """Cluster predicted tree points into individual tree instances.

    Parameters
    ----------
    xyz               : (N, 3) float — raw (non-normalised) coordinates in metres
    pred_mask         : (N,) bool/int — True/1 for points predicted as tree
    eps               : DBSCAN neighbourhood radius in metres
    min_samples       : DBSCAN minimum core-point neighbours
    min_cluster_points: discard clusters smaller than this (noise filter)

    Returns
    -------
    instance_ids : (N,) int
        0  = non-tree (pred_mask == 0)
        -1 = noise (DBSCAN outlier among tree points)
        >0 = tree instance ID (1-indexed)
    """
    n = len(xyz)
    instance_ids = np.zeros(n, dtype=np.int32)

    # Garantizar que pred_mask tenga la misma longitud que xyz.
    # Puede ocurrir una discrepancia mínima cuando los bloques reensamblados
    # cubren ligeramente más puntos que los que recibió predict_flow_b.
    pred_mask_arr = np.asarray(pred_mask, dtype=bool)
    if len(pred_mask_arr) != n:
        pred_mask_arr = pred_mask_arr[:n]

    tree_pts_idx = np.where(pred_mask_arr)[0]
    if len(tree_pts_idx) == 0:
        return instance_ids

    tree_xyz = xyz[tree_pts_idx]
    # Proyección 2D (XY): evita fusionar coronas que se superponen verticalmente
    # pero pertenecen a árboles distintos. En TLS el separador natural entre
    # árboles adyacentes está en el plano horizontal (troncos separados en XY),
    # no en la dimensión vertical (copas que se solapan en Z).
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=1)
    raw_labels = db.fit_predict(tree_xyz[:, :2])  # XY only — separación horizontal

    # Map cluster labels to instance IDs, filter small clusters
    next_id = 1
    label_to_id: dict[int, int] = {}
    for lbl in np.unique(raw_labels):
        if lbl == -1:
            continue
        size = int((raw_labels == lbl).sum())
        if size < min_cluster_points:
            continue
        label_to_id[int(lbl)] = next_id
        next_id += 1

    for raw_lbl, inst_id in label_to_id.items():
        mask = raw_labels == raw_lbl
        instance_ids[tree_pts_idx[mask]] = inst_id

    # Points predicted as tree but assigned to a filtered cluster get -1
    noise_tree = tree_pts_idx[raw_labels == -1]
    filtered_tree = tree_pts_idx[
        (raw_labels != -1) & np.isin(raw_labels, list(label_to_id.keys()), invert=True)
    ]
    instance_ids[noise_tree] = -1
    instance_ids[filtered_tree] = -1

    return instance_ids


def compute_instance_centroids(xyz: np.ndarray,
                                instance_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Compute XYZ centroid for each instance (ID > 0)."""
    centroids = {}
    for iid in np.unique(instance_ids):
        if iid <= 0:
            continue
        mask = instance_ids == iid
        centroids[int(iid)] = xyz[mask].mean(axis=0)
    return centroids


def compute_gt_centroids(xyz: np.ndarray,
                          tree_id: np.ndarray) -> dict[int, np.ndarray]:
    """Compute XYZ centroid per ground-truth tree (treeID > 0)."""
    centroids = {}
    for tid in np.unique(tree_id):
        if tid <= 0:
            continue
        mask = tree_id == tid
        centroids[int(tid)] = xyz[mask].mean(axis=0)
    return centroids


def compute_gt_apexes(xyz: np.ndarray,
                      tree_id: np.ndarray) -> dict[int, np.ndarray]:
    """Compute XYZ apex (highest-Z point) per ground-truth tree (treeID > 0).

    For ITD evaluation with raster Z-max detection, the apex is the correct
    GT reference: both the detector and the GT are defined as "the highest
    visible point per tree".  This mirrors the CHM-watershed ITD convention
    in ALS literature (Chen et al. 2006; Jakubowski et al. 2013).
    """
    apexes = {}
    for tid in np.unique(tree_id):
        if tid <= 0:
            continue
        mask = tree_id == tid
        pts = xyz[mask]
        apex_idx = int(np.argmax(pts[:, 2]))
        apexes[int(tid)] = pts[apex_idx]
    return apexes


def match_trees(pred_centroids: dict[int, np.ndarray],
                gt_centroids:   dict[int, np.ndarray],
                match_dist: float) -> dict:
    """Greedy nearest-neighbour matching of predicted vs GT tree centroids.

    Each predicted tree is matched to the nearest unmatched GT tree within
    match_dist metres (2D horizontal distance in the XY plane).
    This mirrors the standard ITD benchmark protocol.

    Returns
    -------
    dict with keys: tp, fp, fn, precision, recall, f1,
                    matched_pairs [(pred_id, gt_id, dist), ...]
    """
    pred_ids = list(pred_centroids.keys())
    gt_ids   = list(gt_centroids.keys())

    if not pred_ids:
        return {
            "tp": 0, "fp": 0, "fn": len(gt_ids),
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "matched_pairs": [],
        }

    if not gt_ids:
        return {
            "tp": 0, "fp": len(pred_ids), "fn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "matched_pairs": [],
        }

    pred_arr = np.array([pred_centroids[i][:2] for i in pred_ids])   # XY only
    gt_arr   = np.array([gt_centroids[i][:2]   for i in gt_ids])

    # Pairwise distances (predicted x GT)
    dists = np.linalg.norm(
        pred_arr[:, None, :] - gt_arr[None, :, :], axis=-1
    )   # (P, G)

    matched_gt   = set()
    matched_pred = set()
    matched_pairs = []

    # Sort candidate matches by distance (greedy closest-first)
    flat_idx = np.argsort(dists.ravel())
    for idx in flat_idx:
        pi = idx // len(gt_ids)
        gi = idx  % len(gt_ids)
        if pi in matched_pred or gi in matched_gt:
            continue
        if dists[pi, gi] > match_dist:
            break   # remaining distances are all >= current (sorted)
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_pairs.append((pred_ids[pi], gt_ids[gi], float(dists[pi, gi])))

    tp = len(matched_pairs)
    fp = len(pred_ids) - tp
    fn = len(gt_ids)   - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "n_pred":    len(pred_ids),
        "n_gt":      len(gt_ids),
        "matched_pairs": matched_pairs,
    }


def evaluate_itd(xyz: np.ndarray, instance_ids: np.ndarray,
                 gt_tree_id: np.ndarray, match_dist: float) -> dict:
    """Full ITD evaluation for a single plot / scene.

    Parameters
    ----------
    xyz          : (N, 3) raw coordinates in metres
    instance_ids : (N,) predicted instance IDs from cluster_trees()
    gt_tree_id   : (N,) ground-truth treeID (0 = non-tree)
    match_dist   : matching threshold in metres (horizontal)

    Returns
    -------
    Combined dict with segmentation + ITD metrics.
    """
    pred_centroids = compute_instance_centroids(xyz, instance_ids)
    gt_centroids   = compute_gt_centroids(xyz, gt_tree_id)
    return match_trees(pred_centroids, gt_centroids, match_dist)
