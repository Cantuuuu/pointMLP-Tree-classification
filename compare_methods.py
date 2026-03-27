"""
Comparación de Métodos: Watershed SOLO vs Watershed + PointNet++
================================================================
Evalúa dos flujos de detección de árboles individuales contra las
anotaciones de referencia (ground truth) del dataset NeonTreeEvaluation:

  Flujo A — Watershed SOLO:
    clasificación ASPRS clase 5 (alta vegetación) → watershed

  Flujo B — Watershed + PointNet++:
    inferencia PointNet++ (punto a punto) → watershed

Mide P / R / F1 para detección de árboles individuales (ITD) y
compara la calidad de la máscara semántica antes del clustering.

Uso:
    python compare_methods.py
    python compare_methods.py --tile mlbs
    python compare_methods.py --tile sjer --match-dist 8
    python compare_methods.py --laz path/to/tile.laz --xml path/to/ann.xml

Notas sobre sesgo:
    MLBS tile → está en los datos de ENTRENAMIENTO de PointNet++
               → Flujo B puede mostrar métricas infladas (sesgo de entrenamiento)
    SJER tile → está en el conjunto de TEST de PointNet++
              → comparación justa / sin sesgo
"""

import argparse
import io
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Tiles disponibles
# ---------------------------------------------------------------------------
TILES = {
    "mlbs": {
        "laz": "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_MLBS_3_541000_4140000_image_crop.laz",
        "xml": "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_MLBS_3_541000_4140000_image_crop.xml",
        "site": "MLBS",
        "description": "Mountain Lake Biological Station (Virginia, EE.UU.) — bosque templado caducifolio denso",
        "bias_warning": "⚠️ SESGO: Este tile estuvo en el conjunto de ENTRENAMIENTO de PointNet++. Las métricas del Flujo B pueden estar infladas.",
        "split": "train",
        "density_ha": 324,
    },
    "sjer": {
        "laz": "C:/Users/cantu/Downloads/NeonTreeEvaluation/evaluation/evaluation/LiDAR/2018_SJER_3_252000_4104000_image_628.laz",
        "xml": "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_SJER_3_252000_4104000_image_628.xml",
        "site": "SJER",
        "description": "San Joaquin Experimental Range (California, EE.UU.) — sabana/robledal mediterráneo abierto",
        "bias_warning": "✅ SIN SESGO: Este tile pertenece al conjunto de TEST de PointNet++. Comparación justa.",
        "split": "test",
        "density_ha": None,
    },
}

SEG_CONFIG   = "configs/segmentation.yaml"
SEG_CKPT     = "results/segmentation/best_model.pth"
OUTPUT_BASE  = "results/comparison"


# ---------------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tile(laz_path: str):
    import laspy
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
    import xml.etree.ElementTree as ET
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

        wx_min, wy_max = px_to_world(xmin_px, ymin_px)
        wx_max, wy_min = px_to_world(xmax_px, ymax_px)

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

    return trees_gt


# ---------------------------------------------------------------------------
# Watershed (extraído de pipeline.py — idéntico)
# ---------------------------------------------------------------------------

def run_watershed(xyz: np.ndarray, tree_mask: np.ndarray,
                  classification: np.ndarray, cfg: dict) -> dict:
    """Ejecuta watershed y retorna centroides + resultado."""
    from scipy import ndimage

    cluster_cfg = cfg["clustering"]
    chm_res     = cluster_cfg.get("chm_resolution", 0.65)
    sigma       = cluster_cfg.get("smooth_sigma", 0.7)
    min_h       = cluster_cfg.get("min_tree_height", 3.0)
    min_px      = cluster_cfg.get("min_crown_pixels", 4)
    win         = cluster_cfg.get("local_max_window", 3)

    tree_xyz = xyz[tree_mask]
    if len(tree_xyz) == 0:
        return {"centroids": [], "n_trees": 0, "n_tree_points": 0}

    # Ground elevation
    ground_mask = (classification == 2)
    if ground_mask.sum() > 0:
        ground_z = float(np.median(xyz[ground_mask, 2]))
    else:
        ground_z = float(np.percentile(xyz[:, 2], 5))

    x, y  = tree_xyz[:, 0], tree_xyz[:, 1]
    z_abs = tree_xyz[:, 2]
    z     = z_abs - ground_z
    x_min, y_min = x.min(), y.min()

    col = ((x - x_min) / chm_res).astype(np.int32)
    row = ((y - y_min) / chm_res).astype(np.int32)
    n_cols, n_rows = col.max() + 1, row.max() + 1

    chm = np.full((n_rows, n_cols), -np.inf, dtype=np.float64)
    np.maximum.at(chm.ravel(), row * n_cols + col, z)
    chm[chm == -np.inf] = 0.0

    chm_smooth = ndimage.gaussian_filter(chm, sigma=sigma)
    fp         = np.ones((win, win))
    local_max  = ndimage.maximum_filter(chm_smooth, footprint=fp)
    is_max     = (chm_smooth == local_max) & (chm_smooth >= min_h)
    markers, n_seeds = ndimage.label(is_max)

    if n_seeds == 0:
        return {"centroids": [], "n_trees": 0, "n_tree_points": len(tree_xyz)}

    chm_inv      = chm_smooth.max() - chm_smooth
    chm_inv_norm = chm_inv / (chm_inv.max() + 1e-8) * 65534
    wlabels      = ndimage.watershed_ift(chm_inv_norm.astype(np.uint16), markers)
    wlabels[chm < min_h] = 0

    for lbl in range(1, n_seeds + 1):
        if (wlabels == lbl).sum() < min_px:
            wlabels[wlabels == lbl] = 0

    unique = set(np.unique(wlabels)) - {0}
    point_labels = wlabels[row, col]

    centroids = []
    for lbl in sorted(unique):
        m = point_labels == lbl
        if m.sum() == 0:
            continue
        cx = float(x[m].mean())
        cy = float(y[m].mean())
        cz = float(z_abs[m].mean())
        centroids.append({"cx": cx, "cy": cy, "cz": cz,
                          "n_points": int(m.sum()),
                          "crown_pixels": int((wlabels == lbl).sum()),
                          "label": int(lbl)})

    return {
        "centroids": centroids,
        "n_trees":   len(centroids),
        "n_tree_points": len(tree_xyz),
        "chm_shape": (n_rows, n_cols),
        "chm_res":   chm_res,
        "chm_smooth": chm_smooth,
        "watershed_labels": wlabels,
        "x_min": float(x_min), "y_min": float(y_min),
    }


# ---------------------------------------------------------------------------
# Evaluación contra GT
# ---------------------------------------------------------------------------

def match_detections(detected: list, gt_trees: list, match_dist: float = 8.0):
    """Greedy nearest-neighbour matching (distancia centroide)."""
    if not detected or not gt_trees:
        return [], list(range(len(detected))), list(range(len(gt_trees)))

    det_xy = np.array([[d["cx"], d["cy"]] for d in detected])
    gt_xy  = np.array([[g["cx"], g["cy"]] for g in gt_trees])

    matched_det = set()
    matched_gt  = set()
    tp_pairs    = []

    dists = np.sqrt(
        ((det_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=-1)
    )  # (n_det, n_gt)

    while True:
        if dists.min() > match_dist:
            break
        i, j = np.unravel_index(dists.argmin(), dists.shape)
        if i in matched_det or j in matched_gt:
            dists[i, j] = np.inf
            continue
        tp_pairs.append((int(i), int(j)))
        matched_det.add(i)
        matched_gt.add(j)
        dists[i, :] = np.inf
        dists[:, j] = np.inf

    fp_ids = [i for i in range(len(detected)) if i not in matched_det]
    fn_ids = [j for j in range(len(gt_trees)) if j not in matched_gt]
    return tp_pairs, fp_ids, fn_ids


def compute_itd_metrics(tp_pairs, fp_ids, fn_ids):
    tp = len(tp_pairs)
    fp = len(fp_ids)
    fn = len(fn_ids)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def mask_quality(tree_mask: np.ndarray, classification: np.ndarray):
    """Calidad de la máscara semántica: cobertura ASPRS clase 5 como referencia."""
    asprs_tree = (classification == 5)
    n_total   = len(tree_mask)
    n_pred    = int(tree_mask.sum())
    n_asprs   = int(asprs_tree.sum())

    if n_asprs > 0:
        tp = int((tree_mask & asprs_tree).sum())
        fp = int((tree_mask & ~asprs_tree).sum())
        fn = int((~tree_mask & asprs_tree).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_mask   = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    else:
        precision = recall = f1_mask = iou = float("nan")
        tp = fp = fn = 0

    return {
        "n_total": n_total,
        "n_predicted_tree": n_pred,
        "n_asprs_tree": n_asprs,
        "tree_ratio_pred": n_pred / n_total,
        "tree_ratio_asprs": n_asprs / n_total,
        "mask_vs_asprs_precision": precision,
        "mask_vs_asprs_recall": recall,
        "mask_vs_asprs_f1": f1_mask,
        "mask_vs_asprs_iou": iou,
    }


# ---------------------------------------------------------------------------
# PointNet++ inference
# ---------------------------------------------------------------------------

def run_pointnet2(xyz: np.ndarray, classification: np.ndarray,
                  cfg: dict, ckpt_path: str, device: torch.device) -> np.ndarray:
    """Retorna máscara binaria (N,) desde PointNet++."""
    from src.segmentation.pointnet2_model import build_model
    from src.segmentation.pipeline import segment_scene

    print("  Cargando modelo PointNet++...")
    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("  Ejecutando inferencia por ventanas deslizantes...")
    predictions = segment_scene(xyz, classification, model, device, cfg)
    return predictions == 1


# ---------------------------------------------------------------------------
# Generar figuras SVG inline
# ---------------------------------------------------------------------------

def make_svg_map(xyz, tree_mask, gt_trees, detected, tp_pairs, fp_ids, fn_ids,
                 title, width=600, height=600, margin=30):
    """Genera SVG con árb GT (verde), TP (azul), FP (rojo), FN (naranja)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    scale_x = (width - 2 * margin) / max(x_max - x_min, 1e-6)
    scale_y = (height - 2 * margin) / max(y_max - y_min, 1e-6)

    def proj(wx, wy):
        px = margin + (wx - x_min) * scale_x
        py = height - margin - (wy - y_min) * scale_y
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>',
        f'<text x="{width//2}" y="18" text-anchor="middle" fill="white" font-size="13" font-family="sans-serif">{title}</text>',
    ]

    # Subsample point cloud (max 8000 pts for SVG performance)
    n = len(xyz)
    if n > 8000:
        idx = np.random.choice(n, 8000, replace=False)
    else:
        idx = np.arange(n)
    for i in idx:
        if tree_mask[i]:
            color = "#4CAF50"
            r = 1.2
        else:
            color = "#404060"
            r = 0.8
        px, py = proj(x[i], y[i])
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" opacity="0.6"/>')

    # FN (GT trees missed) — naranja
    for j in fn_ids:
        g = gt_trees[j]
        px, py = proj(g["cx"], g["cy"])
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="none" stroke="#FF9800" stroke-width="2"/>')
        lines.append(f'<text x="{px:.1f}" y="{py-7:.1f}" text-anchor="middle" fill="#FF9800" font-size="8">FN</text>')

    # FP (detected but no GT) — rojo
    for i in fp_ids:
        d = detected[i]
        px, py = proj(d["cx"], d["cy"])
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="none" stroke="#F44336" stroke-width="2"/>')
        lines.append(f'<text x="{px:.1f}" y="{py-7:.1f}" text-anchor="middle" fill="#F44336" font-size="8">FP</text>')

    # TP — azul
    for di, gi in tp_pairs:
        d = detected[di]
        px, py = proj(d["cx"], d["cy"])
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="none" stroke="#2196F3" stroke-width="2"/>')

    # Leyenda
    legend_items = [
        ("#4CAF50", "Punto árbol (pred)"),
        ("#2196F3", "TP — Árbol detectado"),
        ("#FF9800", "FN — Árbol perdido"),
        ("#F44336", "FP — Falsa detección"),
    ]
    lx, ly = margin + 5, height - margin - 5 - len(legend_items) * 14
    for color, label in legend_items:
        lines.append(f'<circle cx="{lx+6}" cy="{ly+4}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{lx+14}" y="{ly+8}" fill="white" font-size="9" font-family="sans-serif">{label}</text>')
        ly += 14

    lines.append("</svg>")
    return "\n".join(lines)


def make_svg_mask_comparison(xyz, asprs_mask, pnet_mask, width=700, height=300, margin=20):
    """Comparación lado a lado: máscara ASPRS vs PointNet++."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    panel_w = (width - 3 * margin) // 2
    panel_h = height - 2 * margin - 20

    def proj(wx, wy, offset_x):
        scale_x = panel_w / max(x_max - x_min, 1e-6)
        scale_y = panel_h / max(y_max - y_min, 1e-6)
        px = offset_x + margin + (wx - x_min) * scale_x
        py = height - margin - 5 - (wy - y_min) * scale_y
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#1a1a2e"/>',
        f'<text x="{margin + panel_w//2}" y="14" text-anchor="middle" fill="#aaa" font-size="11" font-family="sans-serif">ASPRS Clase 5</text>',
        f'<text x="{2*margin + panel_w + panel_w//2}" y="14" text-anchor="middle" fill="#aaa" font-size="11" font-family="sans-serif">PointNet++</text>',
    ]

    n = len(xyz)
    idx = np.random.choice(n, min(n, 6000), replace=False)

    for panel_idx, (mask, offset_x) in enumerate([
        (asprs_mask,  0),
        (pnet_mask,   margin + panel_w),
    ]):
        for i in idx:
            if mask[i]:
                color = "#4CAF50"
                r = 1.2
            else:
                color = "#404060"
                r = 0.7
            px, py = proj(x[i], y[i], offset_x)
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" opacity="0.7"/>')

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generar reporte HTML en español
# ---------------------------------------------------------------------------

def generate_report(tile_info: dict,
                    xyz: np.ndarray,
                    cls: np.ndarray,
                    gt_trees: list,
                    area_ha: float,
                    result_solo: dict,
                    result_pnet: dict,
                    metrics_solo: dict,
                    metrics_pnet: dict,
                    mq_solo: dict,
                    mq_pnet: dict,
                    seg_time: float,
                    match_dist: float,
                    output_dir: Path):
    """Genera reporte HTML comparativo en español."""

    ts     = datetime.now().strftime("%Y-%m-%d %H:%M")
    site   = tile_info["site"]
    desc   = tile_info["description"]
    bias   = tile_info["bias_warning"]
    split  = tile_info["split"]
    n_gt   = len(gt_trees)
    density= n_gt / area_ha

    # --- SVGs ---
    np.random.seed(42)

    det_solo  = result_solo["centroids"]
    det_pnet  = result_pnet["centroids"]
    tp_s, fp_s, fn_s = match_detections(det_solo, gt_trees, match_dist)
    tp_p, fp_p, fn_p = match_detections(det_pnet, gt_trees, match_dist)

    svg_gt_solo = make_svg_map(
        xyz, cls == 5, gt_trees, det_solo, tp_s, fp_s, fn_s,
        "Flujo A — Watershed SOLO")
    svg_gt_pnet = make_svg_map(
        xyz, (cls == 5), gt_trees, det_pnet, tp_p, fp_p, fn_p,
        "Flujo B — Watershed + PointNet++")
    svg_masks = make_svg_mask_comparison(
        xyz, cls == 5, mq_pnet["_pnet_mask"])

    def pct(v): return f"{v*100:.1f}%"
    def fmt(v, dec=3): return f"{v:.{dec}f}"

    # Diferencias de métricas
    delta_f1 = metrics_pnet["f1"] - metrics_solo["f1"]
    delta_r  = metrics_pnet["recall"] - metrics_solo["recall"]
    delta_p  = metrics_pnet["precision"] - metrics_solo["precision"]
    sign     = lambda v: f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Comparación de Métodos ITD — {site}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f8f9fa; color: #333; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 30px 20px; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; margin-top: 40px; }}
  h3 {{ color: #16a085; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
           box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
  .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px;
              border-radius: 4px; margin: 10px 0; font-size: 0.95em; }}
  .info {{ background: #d4edda; border-left: 4px solid #28a745; padding: 12px 16px;
           border-radius: 4px; margin: 10px 0; font-size: 0.95em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th {{ background: #2c3e50; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f5f5f5; }}
  .metric-big {{ font-size: 2em; font-weight: bold; text-align: center; }}
  .winner {{ background: #d4edda; }}
  .loser {{ background: #f8d7da; }}
  .neutral {{ background: #fff3cd; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }}
  .metric-card {{ background: white; border-radius: 8px; padding: 16px;
                  box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center; }}
  .label {{ font-size: 0.85em; color: #666; margin-bottom: 5px; }}
  .value-a {{ color: #e67e22; font-size: 1.8em; font-weight: bold; }}
  .value-b {{ color: #2980b9; font-size: 1.8em; font-weight: bold; }}
  .delta {{ font-size: 0.9em; margin-top: 4px; color: #555; }}
  svg {{ display: block; margin: 0 auto; }}
  .ref {{ font-size: 0.85em; color: #555; }}
  .badge-train {{ background: #dc3545; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
  .badge-test  {{ background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; }}
  .section-num {{ color: #3498db; font-size: 0.9em; font-weight: normal; }}
  code {{ background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #2c3e50; color: #ecf0f1; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="container">

<h1>Comparación de Métodos: Watershed SOLO vs Watershed + PointNet++</h1>
<p>Sitio: <strong>{site}</strong> — {desc}<br>
   Split PointNet++: <span class="badge-{'test' if split=='test' else 'train'}">{split.upper()}</span> &nbsp;
   Fecha: {ts} &nbsp; Distancia de emparejamiento: {match_dist} m</p>

<div class="{'info' if split=='test' else 'warning'}">{bias}</div>

<!-- ================================================================ -->
<h2><span class="section-num">§1</span> Descripción del Experimento</h2>
<!-- ================================================================ -->
<div class="card">
<p>Este experimento compara dos flujos de detección de árboles individuales (ITD —
<em>Individual Tree Detection</em>) sobre el mismo tile LiDAR:</p>

<table>
<tr><th>Flujo</th><th>Máscara semántica</th><th>Clustering</th><th>Ventaja</th><th>Limitación</th></tr>
<tr>
  <td><strong>A — Watershed SOLO</strong></td>
  <td>ASPRS clase 5 (alta vegetación, etiquetado manual)</td>
  <td>Watershed CHM</td>
  <td>Sin modelo aprendido; techo teórico del watershed</td>
  <td>Depende de clasificación experta</td>
</tr>
<tr>
  <td><strong>B — Watershed + PointNet++</strong></td>
  <td>Predicciones por punto de PointNet++</td>
  <td>Watershed CHM</td>
  <td>Automatizable en tiles sin clasificación ASPRS</td>
  <td>Introduce errores del modelo de segmentación</td>
</tr>
</table>

<p>Ambos flujos usan los mismos parámetros de watershed (optimizados en búsqueda
de hiperparámetros previa):</p>

<pre>chm_resolution: 0.65 m/px  |  smooth_sigma: 0.7  |  local_max_window: 3
min_tree_height: 3.0 m     |  min_crown_pixels: 4</pre>

<p>La evaluación sigue el protocolo estándar de <strong>detección de árbol individual</strong>:
emparejamiento voraz por distancia entre centroides (umbral {match_dist} m) y
cálculo de Precisión / Recall / F1 a nivel de árbol.</p>
</div>

<!-- ================================================================ -->
<h2><span class="section-num">§2</span> Datos del Tile</h2>
<!-- ================================================================ -->
<div class="card">
<div class="grid3">
  <div class="metric-card">
    <div class="label">Puntos LiDAR</div>
    <div class="value-a">{len(xyz):,}</div>
  </div>
  <div class="metric-card">
    <div class="label">Área</div>
    <div class="value-a">{area_ha:.2f} ha</div>
  </div>
  <div class="metric-card">
    <div class="label">Árboles GT</div>
    <div class="value-a">{n_gt}</div>
    <div class="delta">{density:.0f} árboles/ha</div>
  </div>
  <div class="metric-card">
    <div class="label">Puntos clase 5 (ASPRS)</div>
    <div class="value-a">{mq_solo['n_asprs_tree']:,}</div>
    <div class="delta">{pct(mq_solo['tree_ratio_asprs'])} del total</div>
  </div>
  <div class="metric-card">
    <div class="label">Puntos PointNet++ (árbol)</div>
    <div class="value-b">{mq_pnet['n_predicted_tree']:,}</div>
    <div class="delta">{pct(mq_pnet['tree_ratio_pred'])} del total</div>
  </div>
  <div class="metric-card">
    <div class="label">Tiempo inferencia PointNet++</div>
    <div class="value-b">{seg_time:.1f}s</div>
  </div>
</div>
</div>

<!-- ================================================================ -->
<h2><span class="section-num">§3</span> Calidad de la Máscara Semántica</h2>
<!-- ================================================================ -->
<div class="card">
<p>La calidad de la máscara semántica (árbol vs no árbol) se mide usando
ASPRS clase 5 como referencia. Para el <strong>Flujo A</strong> la máscara
<em>es</em> ASPRS clase 5, por lo que todas las métricas son 1.0 por definición.</p>

{svg_masks}

<br>
<table>
<tr>
  <th>Métrica de máscara</th>
  <th>Flujo A (ASPRS cls 5)</th>
  <th>Flujo B (PointNet++)</th>
  <th>Δ (B − A)</th>
</tr>
<tr>
  <td>Puntos árbol predichos</td>
  <td>{mq_solo['n_asprs_tree']:,}</td>
  <td>{mq_pnet['n_predicted_tree']:,}</td>
  <td>{mq_pnet['n_predicted_tree'] - mq_solo['n_asprs_tree']:+,}</td>
</tr>
<tr>
  <td>Ratio árbol</td>
  <td>{pct(mq_solo['tree_ratio_asprs'])}</td>
  <td>{pct(mq_pnet['tree_ratio_pred'])}</td>
  <td>—</td>
</tr>
<tr>
  <td>Precisión vs ASPRS</td>
  <td>1.000</td>
  <td>{fmt(mq_pnet['mask_vs_asprs_precision'])}</td>
  <td>{mq_pnet['mask_vs_asprs_precision'] - 1:.3f}</td>
</tr>
<tr>
  <td>Recall vs ASPRS</td>
  <td>1.000</td>
  <td>{fmt(mq_pnet['mask_vs_asprs_recall'])}</td>
  <td>{mq_pnet['mask_vs_asprs_recall'] - 1:.3f}</td>
</tr>
<tr>
  <td>F1 vs ASPRS</td>
  <td>1.000</td>
  <td>{fmt(mq_pnet['mask_vs_asprs_f1'])}</td>
  <td>{mq_pnet['mask_vs_asprs_f1'] - 1:.3f}</td>
</tr>
<tr>
  <td>IoU vs ASPRS</td>
  <td>1.000</td>
  <td>{fmt(mq_pnet['mask_vs_asprs_iou'])}</td>
  <td>{mq_pnet['mask_vs_asprs_iou'] - 1:.3f}</td>
</tr>
</table>
<p class="ref">Métricas de PointNet++ en el conjunto de prueba (conjunto completo, no solo este tile):
Tree F1 = 0.892, Tree IoU = 0.804, mIoU = 0.874 (ver <code>results/segmentation/eval_test.json</code>).</p>
</div>

<!-- ================================================================ -->
<h2><span class="section-num">§4</span> Métricas de Detección de Árbol Individual (ITD)</h2>
<!-- ================================================================ -->
<div class="card">
<div class="grid3">
  <div class="metric-card {'winner' if metrics_pnet['f1'] >= metrics_solo['f1'] else 'loser'}">
    <div class="label">F1 — Flujo A (SOLO)</div>
    <div class="value-a">{fmt(metrics_solo['f1'])}</div>
  </div>
  <div class="metric-card {'winner' if metrics_pnet['f1'] >= metrics_solo['f1'] else 'neutral'}">
    <div class="label">F1 — Flujo B (PointNet++)</div>
    <div class="value-b">{fmt(metrics_pnet['f1'])}</div>
  </div>
  <div class="metric-card">
    <div class="label">Δ F1 (B − A)</div>
    <div style="font-size:1.8em;font-weight:bold;color:{'#27ae60' if delta_f1>=0 else '#c0392b'}">{sign(delta_f1)}</div>
  </div>
</div>
<br>
<table>
<tr>
  <th>Métrica ITD</th>
  <th>Flujo A — Watershed SOLO</th>
  <th>Flujo B — Watershed + PointNet++</th>
  <th>Δ (B − A)</th>
</tr>
<tr>
  <td>Árboles detectados</td>
  <td>{result_solo['n_trees']}</td>
  <td>{result_pnet['n_trees']}</td>
  <td>{result_pnet['n_trees'] - result_solo['n_trees']:+}</td>
</tr>
<tr>
  <td>Árboles GT</td>
  <td>{n_gt}</td>
  <td>{n_gt}</td>
  <td>—</td>
</tr>
<tr class="{'winner' if metrics_pnet['precision'] >= metrics_solo['precision'] else ''}">
  <td>Precisión</td>
  <td>{fmt(metrics_solo['precision'])}</td>
  <td>{fmt(metrics_pnet['precision'])}</td>
  <td>{sign(delta_p)}</td>
</tr>
<tr class="{'winner' if metrics_pnet['recall'] >= metrics_solo['recall'] else ''}">
  <td>Recall</td>
  <td>{fmt(metrics_solo['recall'])}</td>
  <td>{fmt(metrics_pnet['recall'])}</td>
  <td>{sign(delta_r)}</td>
</tr>
<tr class="{'winner' if metrics_pnet['f1'] >= metrics_solo['f1'] else ''}">
  <td><strong>F1</strong></td>
  <td><strong>{fmt(metrics_solo['f1'])}</strong></td>
  <td><strong>{fmt(metrics_pnet['f1'])}</strong></td>
  <td><strong>{sign(delta_f1)}</strong></td>
</tr>
<tr>
  <td>TP (árboles detectados correctamente)</td>
  <td>{metrics_solo['tp']}</td>
  <td>{metrics_pnet['tp']}</td>
  <td>{metrics_pnet['tp'] - metrics_solo['tp']:+}</td>
</tr>
<tr>
  <td>FP (detecciones falsas)</td>
  <td>{metrics_solo['fp']}</td>
  <td>{metrics_pnet['fp']}</td>
  <td>{metrics_pnet['fp'] - metrics_solo['fp']:+}</td>
</tr>
<tr>
  <td>FN (árboles perdidos)</td>
  <td>{metrics_solo['fn']}</td>
  <td>{metrics_pnet['fn']}</td>
  <td>{metrics_pnet['fn'] - metrics_solo['fn']:+}</td>
</tr>
</table>
</div>

<!-- ================================================================ -->
<h2><span class="section-num">§5</span> Comparación Visual</h2>
<!-- ================================================================ -->
<div class="card">
<p>Los mapas muestran la nube de puntos con la máscara árbol predicha por cada flujo.
Círculos: <span style="color:#2196F3">■</span> TP
          <span style="color:#F44336">■</span> FP
          <span style="color:#FF9800">■</span> FN (árbol GT perdido).</p>
<div class="grid2">
  {svg_gt_solo}
  {svg_gt_pnet}
</div>
</div>

<!-- ================================================================ -->
<h2><span class="section-num">§6</span> Análisis e Interpretación</h2>
<!-- ================================================================ -->
<div class="card">
<h3>6.1 ¿Qué significa la diferencia de F1?</h3>
<p>
La diferencia F1 de <strong>{sign(delta_f1)}</strong> entre ambos flujos refleja principalmente
la calidad de la máscara semántica previa al watershed.
El Flujo A usa la clasificación ASPRS (clase 5 = alta vegetación), que es el
<em>techo teórico</em> del watershed con etiquetas perfectas.
El Flujo B introduce los errores del modelo PointNet++
(F1 de máscara = {fmt(mq_pnet['mask_vs_asprs_f1'])}, IoU = {fmt(mq_pnet['mask_vs_asprs_iou'])})
que propagan al paso de clustering.
</p>

<h3>6.2 Ventajas del Flujo B en la práctica</h3>
<p>
Aunque el Flujo B puede no superar al Flujo A en tiles ya clasificados por ASPRS,
su ventaja real aparece cuando:
</p>
<ul>
  <li><strong>El tile no tiene clasificación ASPRS confiable</strong> (datos brutos de escáner aéreo).</li>
  <li><strong>Se procesan múltiples tiles en lote</strong>: PointNet++ es completamente automático.</li>
  <li><strong>Se desea uniformidad</strong>: diferentes operadores ASPRS pueden clasificar clase 5 de forma inconsistente.</li>
</ul>

<h3>6.3 Nota sobre sesgo ({split.upper()})</h3>
{"<p>⚠️ <strong>El tile MLBS está en el conjunto de ENTRENAMIENTO de PointNet++.</strong> Esto significa que el modelo vio bloques de 4096 puntos extraídos de este tile durante el entrenamiento. Las métricas del Flujo B en este tile son potencialmente infladas (el modelo ha memorizado parcialmente la distribución espacial del tile). Para una evaluación científicamente válida, usar el tile SJER (split=test).</p>" if split == "train" else "<p>✅ <strong>El tile SJER pertenece al conjunto de TEST de PointNet++.</strong> El modelo nunca vio este tile durante el entrenamiento. Los resultados representan una comparación justa y generalizable.</p>"}

<h3>6.4 Comparación con la literatura</h3>
<table>
<tr>
  <th>Referencia</th><th>Sitio</th><th>Método</th><th>F1 reportado</th>
</tr>
<tr>
  <td>Weinstein et al. (2020) — NeonTreeEvaluation benchmark</td>
  <td>Múltiples NEON</td>
  <td>DeepForest (RGB CNN)</td>
  <td>0.66</td>
</tr>
<tr>
  <td>Dalponte et al. (2016)</td>
  <td>Bosque boreal</td>
  <td>Watershed CHM</td>
  <td>0.71–0.85</td>
</tr>
<tr>
  <td>Zhen et al. (2016)</td>
  <td>Bosque templado</td>
  <td>Watershed multiscala</td>
  <td>0.80</td>
</tr>
<tr class="{'winner' if metrics_solo['f1'] > 0.8 else ''}">
  <td><strong>Este trabajo — Flujo A (SOLO)</strong></td>
  <td><strong>{site}</strong></td>
  <td><strong>Watershed CHM (ASPRS cls 5)</strong></td>
  <td><strong>{fmt(metrics_solo['f1'])}</strong></td>
</tr>
<tr class="{'winner' if metrics_pnet['f1'] > 0.8 else ''}">
  <td><strong>Este trabajo — Flujo B (PointNet++)</strong></td>
  <td><strong>{site}</strong></td>
  <td><strong>PointNet++ + Watershed CHM</strong></td>
  <td><strong>{fmt(metrics_pnet['f1'])}</strong></td>
</tr>
</table>
</div>

<!-- ================================================================ -->
<h2><span class="section-num">§7</span> Referencias</h2>
<!-- ================================================================ -->
<div class="card ref">
<ol>
  <li>Weinstein, B.G. et al. (2020). <em>Individual tree-crown detection in RGB imagery using semi-supervised deep learning neural networks.</em> Remote Sensing, 11(11), 1309. doi:10.3390/rs11111309</li>
  <li>Dalponte, M. et al. (2016). <em>Delineation of individual tree crowns from ALS and hyperspectral data.</em> IEEE Journal of Selected Topics in Applied Earth Observations, 8(6). doi:10.1109/JSTARS.2015.2422498</li>
  <li>Zhen, Z. et al. (2016). <em>Trends in automatic individual tree crown detection and delineation.</em> ISPRS Journal of Photogrammetry, 114. doi:10.1016/j.isprsjprs.2016.01.011</li>
  <li>Qi, C.R. et al. (2017). <em>PointNet++: Deep hierarchical feature learning on point sets in metric space.</em> NeurIPS 2017.</li>
  <li>Koch, B. et al. (2006). <em>Detection of individual tree crowns in airborne lidar data.</em> Photogrammetric Engineering & Remote Sensing, 72(4).</li>
  <li>Popescu, S.C. & Wynne, R.H. (2004). <em>Seeing the trees in the forest: Using lidar and multispectral data fusion with local filtering and variable window size for estimating tree height.</em> Photogrammetric Engineering & Remote Sensing, 70(5).</li>
</ol>
</div>

<p style="text-align:center;color:#999;font-size:0.8em;margin-top:30px">
  Generado: {ts} | compare_methods.py | PointMLP-Trees
</p>
</div>
</body>
</html>"""

    out_path = output_dir / "comparison_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Reporte: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparación Watershed SOLO vs Watershed + PointNet++")
    parser.add_argument("--tile", default="mlbs",
                        choices=list(TILES.keys()),
                        help="Tile predefinido a evaluar (default: mlbs)")
    parser.add_argument("--laz",  default=None, help="Ruta manual al .laz")
    parser.add_argument("--xml",  default=None, help="Ruta manual al .xml de anotaciones")
    parser.add_argument("--seg-config",   default=SEG_CONFIG)
    parser.add_argument("--seg-ckpt",     default=SEG_CKPT)
    parser.add_argument("--output-dir",   default=OUTPUT_BASE)
    parser.add_argument("--match-dist",   type=float, default=8.0,
                        help="Distancia máxima de emparejamiento GT (metros)")
    args = parser.parse_args()

    # Tile info
    if args.laz and args.xml:
        tile_info = {
            "laz": args.laz, "xml": args.xml,
            "site": Path(args.laz).stem,
            "description": "Tile personalizado",
            "bias_warning": "Verifica manualmente si este tile estuvo en entrenamiento.",
            "split": "unknown",
            "density_ha": None,
        }
    else:
        tile_info = TILES[args.tile]

    # Output dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = Path(args.output_dir) / f"{tile_info['site']}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Comparación de Métodos ITD")
    print(f"  Sitio: {tile_info['site']}")
    print(f"  Split: {tile_info['split'].upper()}")
    print("=" * 60)

    # 1. Cargar datos
    print("\n1. Cargando datos...")
    xyz, cls, area_ha = load_tile(tile_info["laz"])
    print(f"   Puntos: {len(xyz):,}   Área: {area_ha:.2f} ha")

    gt_trees = load_gt_annotations(tile_info["xml"], xyz)
    print(f"   GT árboles: {len(gt_trees)} ({len(gt_trees)/area_ha:.0f}/ha)")

    # 2. Flujo A — Watershed SOLO
    print("\n2. Flujo A — Watershed SOLO (ASPRS clase 5)...")
    cfg = load_config(args.seg_config)
    asprs_mask = (cls == 5)
    print(f"   Puntos árbol (ASPRS): {asprs_mask.sum():,} ({100*asprs_mask.sum()/len(xyz):.1f}%)")
    result_solo = run_watershed(xyz, asprs_mask, cls, cfg)
    print(f"   Árboles detectados: {result_solo['n_trees']}")

    mq_solo = mask_quality(asprs_mask, cls)

    tp_s, fp_s, fn_s = match_detections(result_solo["centroids"], gt_trees, args.match_dist)
    metrics_solo = compute_itd_metrics(tp_s, fp_s, fn_s)
    print(f"   P={metrics_solo['precision']:.3f}  R={metrics_solo['recall']:.3f}  F1={metrics_solo['f1']:.3f}")

    # 3. Flujo B — Watershed + PointNet++
    print("\n3. Flujo B — Watershed + PointNet++...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Dispositivo: {device}")

    t0 = time.time()
    pnet_mask = run_pointnet2(xyz, cls, cfg, args.seg_ckpt, device)
    seg_time  = time.time() - t0
    print(f"   Puntos árbol (PointNet++): {pnet_mask.sum():,} ({100*pnet_mask.sum()/len(xyz):.1f}%)")
    print(f"   Tiempo de inferencia: {seg_time:.1f}s")

    result_pnet = run_watershed(xyz, pnet_mask, cls, cfg)
    print(f"   Árboles detectados: {result_pnet['n_trees']}")

    mq_pnet = mask_quality(pnet_mask, cls)
    mq_pnet["_pnet_mask"] = pnet_mask  # pass to report for SVG

    tp_p, fp_p, fn_p = match_detections(result_pnet["centroids"], gt_trees, args.match_dist)
    metrics_pnet = compute_itd_metrics(tp_p, fp_p, fn_p)
    print(f"   P={metrics_pnet['precision']:.3f}  R={metrics_pnet['recall']:.3f}  F1={metrics_pnet['f1']:.3f}")

    # 4. Comparación rápida
    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES")
    print("=" * 60)
    print(f"  {'Métrica':<20} {'Flujo A (SOLO)':>16} {'Flujo B (PNet2)':>16} {'Δ':>8}")
    print("-" * 64)
    for key in ["f1", "precision", "recall"]:
        a = metrics_solo[key]
        b = metrics_pnet[key]
        print(f"  {key:<20} {a:>16.3f} {b:>16.3f} {b-a:>+8.3f}")
    print(f"  {'Árboles detectados':<20} {result_solo['n_trees']:>16} {result_pnet['n_trees']:>16} {result_pnet['n_trees']-result_solo['n_trees']:>+8}")
    print(f"  {'GT árboles':<20} {len(gt_trees):>16} {len(gt_trees):>16}")

    # 5. Guardar JSON
    summary = {
        "tile": tile_info["site"],
        "split": tile_info["split"],
        "timestamp": ts,
        "n_gt_trees": len(gt_trees),
        "area_ha": area_ha,
        "match_dist_m": args.match_dist,
        "seg_inference_time_s": round(seg_time, 1),
        "flujo_a_solo": {
            "n_detected": result_solo["n_trees"],
            "metrics": metrics_solo,
            "mask_quality": {k: v for k, v in mq_solo.items()},
        },
        "flujo_b_pointnet2": {
            "n_detected": result_pnet["n_trees"],
            "metrics": metrics_pnet,
            "mask_quality": {k: v for k, v in mq_pnet.items() if not k.startswith("_")},
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Reporte HTML
    print("\n4. Generando reporte HTML...")
    report_path = generate_report(
        tile_info=tile_info,
        xyz=xyz, cls=cls, gt_trees=gt_trees,
        area_ha=area_ha,
        result_solo=result_solo, result_pnet=result_pnet,
        metrics_solo=metrics_solo, metrics_pnet=metrics_pnet,
        mq_solo=mq_solo, mq_pnet=mq_pnet,
        seg_time=seg_time,
        match_dist=args.match_dist,
        output_dir=out_dir,
    )

    print(f"\nTodo guardado en: {out_dir}")
    print(f"Abre el reporte: {report_path}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
