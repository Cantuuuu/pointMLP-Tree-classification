"""
Evaluación completa ITD (Individual Tree Detection) — FORinstance dataset.

Ejecuta los 3 flujos sobre cada plot de TEST y reporta métricas de:
  (a) Segmentación punto a punto (árbol / no-árbol)
  (b) Detección de árboles individuales (ITD): Precision, Recall, F1

Flujos comparados (mismo método ITD raster Z-max post-segmentación — comparación justa):
  A — Geométrico:  HAG > hag_threshold  (sin modelo entrenado)
  B — PointNet++:  inferencia por bloques + ensamblado + raster Z-max ITD
  C — Random Forest: inferencia por bloques + ensamblado + raster Z-max ITD

ITD: raster Z-max + suavizado Gaussiano + máximos locales 2D.
     Análogo al método CHM-watershed usado en ALS, adaptado para TLS.
     Parámetros se sintonizan en plots DEV antes de aplicar a TEST.

Uso:
    python evaluate_itd_forinstance.py
    python evaluate_itd_forinstance.py --tune-only      # solo sintonizar raster ITD
    python evaluate_itd_forinstance.py --skip-tune      # usar params del config
    python evaluate_itd_forinstance.py --flow A         # solo flujo A
"""

import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import time
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import yaml
import torch
import joblib

from src.segmentation.flow_a_geometric import compute_hag, segmentation_metrics
from src.segmentation.itd_clustering  import (
    evaluate_itd_raster, compute_gt_apexes)
from src.segmentation.pointnet2_model import build_model
from src.segmentation.rf_model        import extract_features

OUT_POINTS_CLASS = 3
BLOCK_SIZE  = 5.0
BLOCK_STRIDE = 2.5
NUM_POINTS  = 4096


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de carga
# ══════════════════════════════════════════════════════════════════════════════

def load_plot(las_path: Path) -> dict | None:
    """Carga un plot .las de FORinstance."""
    if not las_path.exists():
        return None
    las = laspy.read(str(las_path))
    xyz = np.stack([np.array(las.x), np.array(las.y), np.array(las.z)],
                   axis=-1).astype(np.float32)
    clf = np.array(las.classification, dtype=np.int32)
    tid = np.array(las["treeID"], dtype=np.int32)
    # Excluir Out-points (clase 3)
    keep = clf != OUT_POINTS_CLASS
    return {"xyz": xyz[keep], "clf": clf[keep], "tree_id": tid[keep]}


# ══════════════════════════════════════════════════════════════════════════════
# Extracción de bloques sobre un plot completo (para inferencia de modelos)
# ══════════════════════════════════════════════════════════════════════════════

def extract_inference_blocks(xyz: np.ndarray) -> list[dict]:
    """
    Extrae bloques normalizados de un plot completo para inferencia.

    TLS plots have very high point density (~2000 pts/m²), so a 5m×5m block
    contains ~50,000 points. Subsampling to 4096 would leave ~92% of points
    uncovered. To ensure full coverage, each block is processed in multiple
    passes: ceil(n_block_pts / NUM_POINTS) non-overlapping subsamples.
    The predict functions accumulate probability sums across all passes.

    Devuelve lista de dicts:
      'points'    : (NUM_POINTS, 3) float32 normalizado
      'orig_idx'  : (NUM_POINTS,)  int64   — índice en xyz original
    """
    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Índice espacial
    cell_size = BLOCK_STRIDE
    cx_idx = ((x - x_min) / cell_size).astype(np.int32)
    cy_idx = ((y - y_min) / cell_size).astype(np.int32)
    ny = int(cy_idx.max()) + 2
    cell_ids = cx_idx * ny + cy_idx

    order = np.argsort(cell_ids)
    sorted_cells = cell_ids[order]
    unique_cells, starts = np.unique(sorted_cells, return_index=True)
    ends = np.append(starts[1:], len(order))
    grid = {int(c): order[starts[i]:ends[i]] for i, c in enumerate(unique_cells)}
    cells_per_block = int(np.ceil(BLOCK_SIZE / cell_size))

    blocks = []
    x_starts = np.arange(x_min, x_max - BLOCK_SIZE / 2, BLOCK_STRIDE)
    y_starts = np.arange(y_min, y_max - BLOCK_SIZE / 2, BLOCK_STRIDE)

    for x0 in x_starts:
        for y0 in y_starts:
            cx0 = int((x0 - x_min) / cell_size)
            cy0 = int((y0 - y_min) / cell_size)
            cand = []
            for dcx in range(cells_per_block + 1):
                for dcy in range(cells_per_block + 1):
                    cid = (cx0 + dcx) * ny + (cy0 + dcy)
                    if cid in grid:
                        cand.append(grid[cid])
            if not cand:
                continue
            cand = np.concatenate(cand)
            in_b = ((x[cand] >= x0) & (x[cand] < x0 + BLOCK_SIZE) &
                    (y[cand] >= y0) & (y[cand] < y0 + BLOCK_SIZE))
            idx = cand[in_b]
            if len(idx) < 64:
                continue

            b_xyz = xyz[idx].copy()

            # Normalización idéntica al preprocesamiento
            b_xyz[:, 0] -= (x0 + BLOCK_SIZE / 2)
            b_xyz[:, 1] -= (y0 + BLOCK_SIZE / 2)
            b_xyz[:, 0] /= (BLOCK_SIZE / 2)
            b_xyz[:, 1] /= (BLOCK_SIZE / 2)
            # Altura sobre suelo estimada (percentil 5 de Z local)
            ground_z = float(np.percentile(b_xyz[:, 2], 5))
            b_xyz[:, 2] -= ground_z
            z_range2 = b_xyz[:, 2].max() - b_xyz[:, 2].min()
            if z_range2 > 0.1:
                b_xyz[:, 2] = (b_xyz[:, 2] - b_xyz[:, 2].min()) / z_range2
            else:
                b_xyz[:, 2] = 0.0

            b_xyz = b_xyz.astype(np.float32)
            n = len(idx)

            # Multi-pass subsampling: shuffle all points and slice into chunks
            # of NUM_POINTS. This guarantees every point is seen at least once,
            # matching the block-level test evaluation (preprocessed .npy files
            # cover all points with uniform subsampling).
            perm = np.random.permutation(n)
            # Pad to a multiple of NUM_POINTS
            n_passes = max(1, int(np.ceil(n / NUM_POINTS)))
            pad_len = n_passes * NUM_POINTS
            if pad_len > n:
                # Pad with random repeats (as in preprocessing)
                extra = np.random.choice(n, pad_len - n, replace=True)
                perm_padded = np.concatenate([perm, perm[extra]])
            else:
                perm_padded = perm[:pad_len]

            for p in range(n_passes):
                sel_local = perm_padded[p * NUM_POINTS: (p + 1) * NUM_POINTS]
                blocks.append({
                    "points":   b_xyz[sel_local],
                    "orig_idx": idx[sel_local],
                })

    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# Inferencia por flujo sobre un plot completo
# ══════════════════════════════════════════════════════════════════════════════

def predict_flow_a(xyz: np.ndarray, hag_threshold: float) -> np.ndarray:
    """Flow A: HAG threshold. Devuelve pred_mask (N,) bool."""
    hag = compute_hag(xyz, cell_size=1.0)
    return hag > hag_threshold


@torch.no_grad()
def predict_flow_b(xyz: np.ndarray, model: torch.nn.Module,
                   device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Flow B: PointNet++ inferencia por bloques, ensamblado por voto de probabilidad."""
    n = len(xyz)
    prob_sum = np.zeros(n, dtype=np.float32)
    count    = np.zeros(n, dtype=np.int32)

    blocks = extract_inference_blocks(xyz)
    model.eval()

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]
        pts_t = torch.from_numpy(
            np.stack([b["points"] for b in batch])
        ).to(device)                                    # (B, N, 3)
        logits = model(pts_t)                           # (B, N, 2)
        probs  = torch.softmax(logits, dim=-1)[:, :, 1]  # (B, N) P(tree)
        probs_np = probs.cpu().numpy()

        for j, b in enumerate(batch):
            oi = b["orig_idx"]
            prob_sum[oi] += probs_np[j]
            count[oi]    += 1

    # Puntos sin ningún bloque → probabilidad 0 (predicción no-árbol)
    with np.errstate(invalid="ignore"):
        avg_prob = np.where(count > 0, prob_sum / count, 0.0)

    return avg_prob > 0.5


def predict_flow_c(xyz: np.ndarray, clf) -> np.ndarray:
    """Flow C: RF inferencia por bloques, ensamblado por voto de probabilidad."""
    n = len(xyz)
    prob_sum = np.zeros(n, dtype=np.float32)
    count    = np.zeros(n, dtype=np.int32)

    blocks = extract_inference_blocks(xyz)
    for b in blocks:
        feats = extract_features(b["points"])          # (NUM_POINTS, 15)
        probs = clf.predict_proba(feats)[:, 1]        # P(tree) por punto
        oi    = b["orig_idx"]
        prob_sum[oi] += probs
        count[oi]    += 1

    avg_prob = np.where(count > 0, prob_sum / count, 0.0)
    return avg_prob > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Sintonización DBSCAN en plots DEV
# ══════════════════════════════════════════════════════════════════════════════

MAX_TUNE_PTS = 150_000   # máx puntos árbol por plot en tuneo (velocidad)


def tune_raster_itd(dev_plots: list[dict], flow_preds: dict,
                    cell_size_range: list, smooth_range: list,
                    window_range: list, match_dist: float) -> dict:
    """
    Grid search de (eps, min_samples) sobre plots DEV optimizando F1 ITD.

    flow_preds: {plot_name: pred_mask (N,) bool}  — predicciones del flujo B
    Se usa flujo B para sintonizar (representativo del mejor modelo).
    Los mismos parámetros se aplican a los 3 flujos en test.
    """
    print("  Sintonizando raster Z-max ITD en plots DEV...", flush=True)
    best = {"f1": -1.0, "cell_size": cell_size_range[0],
            "smooth_sigma": smooth_range[0], "local_max_window": window_range[0]}

    for cs in cell_size_range:
        for sigma in smooth_range:
            for win in window_range:
                tp = fp = fn = 0
                for p in dev_plots:
                    name = p["name"]
                    if name not in flow_preds:
                        continue
                    pred_mask = flow_preds[name]
                    itd = evaluate_itd_raster(
                        p["xyz"], pred_mask, p["tree_id"],
                        cs, sigma, win, match_dist)
                    tp += itd["tp"]; fp += itd["fp"]; fn += itd["fn"]

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                print(f"    cell={cs:.1f}m  sigma={sigma:.1f}  win={win}  "
                      f"tp={tp} fp={fp} fn={fn}  "
                      f"P={prec:.3f} R={rec:.3f} F1={f1:.4f}", flush=True)
                if f1 > best["f1"]:
                    best = {"f1": f1, "cell_size": cs,
                            "smooth_sigma": sigma, "local_max_window": win,
                            "tp": tp, "fp": fp, "fn": fn}

    print(f"  → Mejor: cell={best['cell_size']}m  "
          f"sigma={best['smooth_sigma']}  win={best['local_max_window']}  "
          f"F1={best['f1']:.4f}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Evaluación de un flujo en un conjunto de plots
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_flow_on_plots(plots: list[dict], flow_name: str,
                            predict_fn,
                            cell_size: float, smooth_sigma: float,
                            local_max_window: int, match_dist: float) -> dict:
    """
    Evalúa un flujo sobre una lista de plots.

    ITD: raster Z-max (análogo a CHM-watershed para ALS, adaptado para TLS).
    Devuelve métricas de segmentación + ITD por plot y agregadas.
    """
    results = []
    seg_agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    itd_agg = {"tp": 0, "fp": 0, "fn": 0}

    for p in plots:
        t0 = time.time()
        pred_mask = predict_fn(p["xyz"])
        gt_labels = (p["tree_id"] > 0).astype(np.int64)

        # Métricas de segmentación
        seg = segmentation_metrics(pred_mask, gt_labels)
        for k in seg_agg:
            seg_agg[k] += seg[k]

        # ITD: raster Z-max local maxima
        itd = evaluate_itd_raster(
            p["xyz"], pred_mask, p["tree_id"],
            cell_size, smooth_sigma, local_max_window, match_dist)
        for k in itd_agg:
            itd_agg[k] += itd[k]

        dt = time.time() - t0
        print(f"    {p['site']}/{p['name']:30s} "
              f"seg_F1={seg['f1']:.3f} seg_IoU={seg['iou']:.3f} | "
              f"ITD n_gt={itd['n_gt']:3d} n_pred={itd['n_pred']:3d} "
              f"P={itd['precision']:.3f} R={itd['recall']:.3f} "
              f"F1={itd['f1']:.3f}  ({dt:.1f}s)", flush=True)

        results.append({
            "site": p["site"], "plot": p["name"],
            "n_pts": int(len(p["xyz"])),
            "n_gt_trees": itd["n_gt"],
            "segmentation": {k: round(v, 4) for k, v in seg.items()
                             if k not in ("tp","fp","fn","tn")},
            "itd": {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in itd.items() if k != "matched_pairs"},
        })

    # Métricas agregadas
    def _seg_agg_metrics(a):
        tp, fp, fn, tn = a["tp"], a["fp"], a["fn"], a["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        iou0 = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
        return {"precision": round(prec,4), "recall": round(rec,4),
                "f1": round(f1,4), "tree_iou": round(iou,4),
                "non_tree_iou": round(iou0,4),
                "mean_iou": round((iou+iou0)/2,4)}

    def _itd_agg_metrics(a):
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec,4), "recall": round(rec,4),
                "f1": round(f1,4)}

    return {
        "flow": flow_name,
        "per_plot": results,
        "segmentation_aggregate": _seg_agg_metrics(seg_agg),
        "itd_aggregate":          _itd_agg_metrics(itd_agg),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/segmentation_forinstance.yaml")
    parser.add_argument("--tune-only",   action="store_true")
    parser.add_argument("--skip-tune",   action="store_true")
    parser.add_argument("--flow", default="ABC",
                        help="Flujos a evaluar, ej. A, B, C, AB, ABC")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(cfg["data"]["forinstance_dir"])
    log_dir     = Path(cfg["experiment"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(dataset_dir / "data_split_metadata.csv")

    np.random.seed(cfg["experiment"]["seed"])

    # ── Cargar modelos ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pnet_model = rf_clf = None
    if "B" in args.flow:
        print("Cargando PointNet++...", flush=True)
        pnet_model = build_model(cfg).to(device)
        ckpt = torch.load(log_dir / "best_model.pth",
                          weights_only=False, map_location=device)
        pnet_model.load_state_dict(ckpt["model_state_dict"])
        pnet_model.eval()
        print(f"  Epoch {ckpt['epoch']}  val_iou={ckpt['best_metric']:.4f}")

    if "C" in args.flow:
        print("Cargando Random Forest...", flush=True)
        rf_clf = joblib.load(log_dir / "rf_model.pkl")
        print("  OK")

    # ── Cargar plots por split ───────────────────────────────────────────────
    def load_split_plots(split_name: str) -> list[dict]:
        rows = meta[meta["split"] == split_name]
        plots = []
        for _, row in rows.iterrows():
            las_path = dataset_dir / row["path"]
            plot = load_plot(las_path)
            if plot is None:
                continue
            plots.append({
                "name":    Path(row["path"]).stem,
                "site":    row["folder"],
                **plot,
            })
        return plots

    print("\nCargando plots DEV para sintonización (1 por sitio)...", flush=True)
    all_dev = load_split_plots("dev")
    # Un plot por sitio — suficiente para calibrar DBSCAN espacialmente;
    # se elige el primero de cada sitio (orden del CSV).
    seen_sites: set = set()
    dev_plots = []
    for p in all_dev:
        if p["site"] not in seen_sites:
            dev_plots.append(p)
            seen_sites.add(p["site"])
    print(f"  {len(dev_plots)} plots DEV seleccionados (1 por sitio): "
          f"{[p['site'] for p in dev_plots]}")

    print("Cargando plots TEST para evaluación...", flush=True)
    test_plots = load_split_plots("test")
    print(f"  {len(test_plots)} plots TEST cargados")

    # ── Sintonización raster Z-max ITD en DEV ───────────────────────────────
    hag_thr     = cfg["flow_a"]["hag_threshold"]
    match_dist  = cfg["clustering"]["match_distance"]

    if args.skip_tune:
        best_cs    = cfg["clustering"].get("cell_size", 0.5)
        best_sigma = cfg["clustering"].get("smooth_sigma", 1.5)
        best_win   = cfg["clustering"].get("local_max_window", 5)
        print(f"\n[SKIP-TUNE] Usando cell_size={best_cs}  "
              f"smooth_sigma={best_sigma}  local_max_window={best_win}")
    else:
        # Sintonizar raster Z-max con Flow B (mejor modelo); mismos params para todos.
        # cell_size: resolución del raster en metros (estándar TLS ITD: 0.25–1.0 m).
        # smooth_sigma: suavizado Gaussiano para eliminar falsos máximos por ramas.
        # local_max_window: ventana de máximos locales en celdas (copa TLS ~2–6 m diam).
        print("\n" + "="*65)
        print("Sintonización raster Z-max ITD (Flow B en plots DEV)")
        print("="*65, flush=True)

        # Predicciones Flow B sobre dev plots
        tune_preds = {}
        for p in dev_plots:
            if "B" in args.flow and pnet_model is not None:
                tune_preds[p["name"]] = predict_flow_b(
                    p["xyz"], pnet_model, device)
            else:
                # Fallback: Flow A para sintonizar
                tune_preds[p["name"]] = predict_flow_a(p["xyz"], hag_thr)

        # Rangos de búsqueda para raster Z-max ITD (GT reference = apex):
        # - cell_size 0.5 m: resolución estándar TLS ITD.
        # - smooth_sigma en celdas: elimina falsos máximos por ramas.
        #   Copas TLS 2–6 m diam → sigma óptimo 1–3 celdas (0.5–1.5 m).
        # - local_max_window en celdas: asegura un solo máximo por copa.
        #   Separación troncos FORinstance ≈ 3–7 m → ventana 3–11 celdas.
        cell_size_range = [0.5, 1.0]
        smooth_range    = [1.0, 1.5, 2.0, 3.0]
        window_range    = [3, 5, 7, 9, 11]
        best_params = tune_raster_itd(
            dev_plots, tune_preds,
            cell_size_range, smooth_range, window_range, match_dist)

        best_cs    = best_params["cell_size"]
        best_sigma = best_params["smooth_sigma"]
        best_win   = best_params["local_max_window"]

        # Actualizar config
        with open(args.config) as f:
            ct = f.read()
        # Append or replace raster ITD params under clustering section
        for key, val in [("cell_size", best_cs),
                         ("smooth_sigma", best_sigma),
                         ("local_max_window", best_win)]:
            old_val = cfg["clustering"].get(key)
            if old_val is not None:
                ct = ct.replace(f"  {key}: {old_val}", f"  {key}: {val}")
        with open(args.config, "w") as f:
            f.write(ct)
        print(f"\nConfig actualizado: cell_size={best_cs}  "
              f"smooth_sigma={best_sigma}  local_max_window={best_win}")

        with open(log_dir / "raster_itd_tuning.json", "w") as f:
            json.dump(best_params, f, indent=2)

    if args.tune_only:
        print("--tune-only: terminando.")
        return

    # ── Evaluación en TEST ───────────────────────────────────────────────────
    all_results = {}

    flows_to_run = []
    if "A" in args.flow:
        flows_to_run.append(("A", lambda xyz: predict_flow_a(xyz, hag_thr)))
    if "B" in args.flow and pnet_model is not None:
        flows_to_run.append(
            ("B", lambda xyz, m=pnet_model, d=device:
             predict_flow_b(xyz, m, d)))
    if "C" in args.flow and rf_clf is not None:
        flows_to_run.append(
            ("C", lambda xyz, c=rf_clf: predict_flow_c(xyz, c)))

    for flow_id, predict_fn in flows_to_run:
        flow_names = {"A": "Geometric (HAG)", "B": "PointNet++", "C": "Random Forest"}
        print(f"\n{'='*65}")
        print(f"Flujo {flow_id} — {flow_names[flow_id]}")
        print(f"{'='*65}", flush=True)

        res = evaluate_flow_on_plots(
            test_plots, flow_names[flow_id], predict_fn,
            best_cs, best_sigma, best_win, match_dist)
        all_results[flow_id] = res

        seg = res["segmentation_aggregate"]
        itd = res["itd_aggregate"]
        print(f"\n  TOTAL  seg: mIoU={seg['mean_iou']:.4f}  "
              f"tree_IoU={seg['tree_iou']:.4f}  F1={seg['f1']:.4f}")
        print(f"         ITD: P={itd['precision']:.4f}  "
              f"R={itd['recall']:.4f}  F1={itd['f1']:.4f}  "
              f"(tp={itd['tp']} fp={itd['fp']} fn={itd['fn']})")

    # ── Tabla comparativa final ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("TABLA COMPARATIVA (test plots)")
    print(f"{'='*65}")
    hdr = f"{'Flujo':<22} {'seg_mIoU':>9} {'tree_IoU':>9} {'seg_F1':>7} | {'ITD_P':>6} {'ITD_R':>6} {'ITD_F1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for fid, res in all_results.items():
        s = res["segmentation_aggregate"]
        i = res["itd_aggregate"]
        fn = {"A":"Geometric (HAG)","B":"PointNet++","C":"Random Forest"}[fid]
        print(f"  {fn:<20} {s['mean_iou']:>9.4f} {s['tree_iou']:>9.4f} "
              f"{s['f1']:>7.4f} | {i['precision']:>6.4f} "
              f"{i['recall']:>6.4f} {i['f1']:>7.4f}")

    # ── Guardar resultados ───────────────────────────────────────────────────
    out = {
        "raster_itd_params": {"cell_size": best_cs, "smooth_sigma": best_sigma,
                              "local_max_window": best_win,
                              "match_distance": match_dist},
        "flow_a_hag_threshold": hag_thr,
        "flows": {fid: {
            "segmentation_aggregate": r["segmentation_aggregate"],
            "itd_aggregate": r["itd_aggregate"],
            "per_plot": r["per_plot"],
        } for fid, r in all_results.items()},
    }
    out_path = log_dir / "itd_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
