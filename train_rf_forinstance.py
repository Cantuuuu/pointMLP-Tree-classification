"""
Entrenamiento del Random Forest para segmentación LiDAR — FORinstance dataset.

Comparación justa con PointNet++:
  - Mismos 2,902 bloques de entrenamiento (split oficial dev).
  - Mismos 511 bloques de validación.
  - Mismo split de test (1,484 bloques).
  - Subsample de 1,024 pts/bloque para entrenamiento del RF:
      PointNet++ itera sobre los 4,096 pts/bloque en cada forward pass;
      RF no es un modelo secuencial y no se beneficia de la densidad
      redundante al entrenar. 1,024 pts capturan la distribución geométrica
      del bloque de forma representativa. Esta práctica es estándar en ML
      sobre nubes de puntos (ver Weinmann et al. 2015, Thomas et al. 2019).
  - Inferencia sobre 4,096 pts/bloque = idéntico a PointNet++ en test.

Parámetros justificados para publicación:
  - n_estimators=200: curva de convergencia RF estabilizada (validada vía OOB).
  - max_depth=20: suficiente profundidad para capturar no linealidades; más
      profundidad no mejora OOB en datos de esta escala (observado en NEON).
  - min_samples_leaf=5: regularización estándar.
  - max_features='sqrt': reducción de varianza entre árboles (Breiman 2001).
  - class_weight='balanced': manejo de desbalance árbol/no-árbol (≈46/54 %).
  - n_jobs=-1: paralelismo completo (sin restricción de VRAM como PointNet++).

Uso:
  python train_rf_forinstance.py
  python train_rf_forinstance.py --config configs/segmentation_forinstance.yaml
  python train_rf_forinstance.py --n-estimators 100  # más rápido, menos preciso
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

# PointNet training subsample para RF (justificado arriba)
TRAIN_PTS_PER_BLOCK = 1024
VAL_PTS_PER_BLOCK   = 4096   # misma resolución que PointNet++ en test


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(data_dir: Path, split: str) -> tuple:
    pts  = np.load(data_dir / split / "points.npy")   # (B, 4096, 3)
    lbls = np.load(data_dir / split / "labels.npy")   # (B, 4096)
    return pts, lbls


def subsample_blocks(pts: np.ndarray, lbls: np.ndarray,
                     n_per_block: int, rng: np.random.Generator) -> tuple:
    """Submuestrea cada bloque a n_per_block puntos (sin reemplazo si posible)."""
    B, N, D = pts.shape
    if n_per_block >= N:
        return pts, lbls
    idx = rng.choice(N, n_per_block, replace=False)
    return pts[:, idx, :], lbls[:, idx]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    iou  = tp / max(tp + fp + fn, 1)
    tn_iou = tn / max(tn + fn + fp, 1)
    return {
        "accuracy": round(acc, 4),
        "tree_precision": round(prec, 4), "tree_recall": round(rec, 4),
        "tree_f1": round(f1, 4), "tree_iou": round(iou, 4),
        "non_tree_iou": round(tn_iou, 4),
        "mean_iou": round((iou + tn_iou) / 2, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation_forinstance.yaml")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth",    type=int, default=20)
    parser.add_argument("--output", default="results/segmentation_forinstance/rf_model.pkl")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_dir = Path(cfg["data"]["processed_dir"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    print("=" * 65)
    print("  Random Forest — FORinstance TLS Segmentation")
    print("=" * 65)
    print(f"  n_estimators : {args.n_estimators}")
    print(f"  max_depth    : {args.max_depth}")
    print(f"  Train pts/block (RF): {TRAIN_PTS_PER_BLOCK}  "
          f"(PointNet++: 4096 — mismo conjunto de bloques)")
    print(f"  Val/test pts/block  : {VAL_PTS_PER_BLOCK}  (idéntico a PointNet++)")

    # ------------------------------------------------------------------
    # 1. Cargar datos
    # ------------------------------------------------------------------
    print("\n[1/4] Cargando datos...", flush=True)
    t0 = time.time()
    train_pts, train_lbl = load_split(data_dir, "train")
    val_pts,   val_lbl   = load_split(data_dir, "val")
    test_pts,  test_lbl  = load_split(data_dir, "test")
    print(f"  Bloques — train:{len(train_pts)}  val:{len(val_pts)}  "
          f"test:{len(test_pts)}  ({time.time()-t0:.1f}s)")

    # Subsample para entrenamiento RF (ver docstring)
    train_pts_sub, train_lbl_sub = subsample_blocks(
        train_pts, train_lbl, TRAIN_PTS_PER_BLOCK, rng)
    n_train_pts = train_pts_sub.shape[0] * train_pts_sub.shape[1]
    print(f"  Submuestra train: {TRAIN_PTS_PER_BLOCK} pts/bloque "
          f"= {n_train_pts/1e6:.2f}M pts totales")

    # ------------------------------------------------------------------
    # 2. Extracción de features
    # ------------------------------------------------------------------
    print("\n[2/4] Extrayendo features (15 por punto)...", flush=True)
    from src.segmentation.rf_model import extract_features_batch

    t0 = time.time()
    X_train, _ = extract_features_batch(train_pts_sub, verbose=True)
    y_train = train_lbl_sub.ravel()
    t_feat_train = time.time() - t0
    print(f"  Train: {X_train.shape}  →  {t_feat_train:.1f}s")

    t0 = time.time()
    X_val, _ = extract_features_batch(val_pts, verbose=False)
    y_val = val_lbl.ravel()
    t_feat_val = time.time() - t0
    print(f"  Val:   {X_val.shape}  →  {t_feat_val:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Entrenamiento con warm_start para mostrar progreso + ETA
    # ------------------------------------------------------------------
    print("\n[3/4] Entrenando Random Forest (warm_start, lotes de 20)...",
          flush=True)

    BATCH = 20   # árboles por lote de progreso
    n_batches = args.n_estimators // BATCH

    # Pre-calcular pesos de clase para evitar el warning de sklearn
    # cuando class_weight='balanced' se combina con warm_start.
    from sklearn.utils.class_weight import compute_class_weight
    classes   = np.array([0, 1])
    cw_values = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {0: float(cw_values[0]), 1: float(cw_values[1])}
    print(f"  class_weight calculado: non-tree={class_weight_dict[0]:.3f}  "
          f"tree={class_weight_dict[1]:.3f}", flush=True)

    clf = RandomForestClassifier(
        n_estimators=0,
        max_depth=args.max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=42,
        oob_score=True,
        warm_start=True,
    )

    t_train_start = time.time()
    batch_times = []

    for b in range(n_batches):
        trees_so_far = (b + 1) * BATCH
        clf.set_params(n_estimators=trees_so_far)

        t_batch = time.time()
        clf.fit(X_train, y_train)
        elapsed_batch = time.time() - t_batch
        batch_times.append(elapsed_batch)

        elapsed_total = time.time() - t_train_start
        avg_batch     = np.mean(batch_times)
        remaining     = avg_batch * (n_batches - b - 1)
        oob_str       = f"  OOB={clf.oob_score_:.4f}" if hasattr(clf, "oob_score_") else ""

        print(f"  Lote {b+1:2d}/{n_batches}  "
              f"[{trees_so_far:3d}/{args.n_estimators} árboles]  "
              f"t={elapsed_batch:.1f}s  "
              f"total={elapsed_total:.0f}s  "
              f"ETA≈{remaining:.0f}s{oob_str}",
              flush=True)

    t_train_total = time.time() - t_train_start
    print(f"\n  Entrenamiento completo: {t_train_total:.1f}s  "
          f"OOB={clf.oob_score_:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluación
    # ------------------------------------------------------------------
    print("\n[4/4] Evaluando...", flush=True)

    # Validación
    t0 = time.time()
    y_pred_val = clf.predict(X_val)
    m_val = compute_metrics(y_val, y_pred_val)
    print(f"\n  === Validación ===")
    print(f"  mIoU={m_val['mean_iou']:.4f}  tree_IoU={m_val['tree_iou']:.4f}  "
          f"F1={m_val['tree_f1']:.4f}  Prec={m_val['tree_precision']:.4f}  "
          f"Rec={m_val['tree_recall']:.4f}  ({time.time()-t0:.1f}s)")

    # Test
    print("  Extrayendo features test...", flush=True)
    t0 = time.time()
    X_test, _ = extract_features_batch(test_pts, verbose=False)
    y_test = test_lbl.ravel()
    t_feat_test = time.time() - t0
    print(f"  Test features: {X_test.shape}  ({t_feat_test:.1f}s)")

    t0 = time.time()
    y_pred_test = clf.predict(X_test)
    m_test = compute_metrics(y_test, y_pred_test)
    print(f"\n  === Test ===")
    print(f"  mIoU={m_test['mean_iou']:.4f}  tree_IoU={m_test['tree_iou']:.4f}  "
          f"F1={m_test['tree_f1']:.4f}  Prec={m_test['tree_precision']:.4f}  "
          f"Rec={m_test['tree_recall']:.4f}  ({time.time()-t0:.1f}s)")

    # Feature importances
    imp_names = ["z","x","y","z_rank_pct","z_above_mean","z_norm_block",
                 "dist_center","local_cnt_k16","local_mean_k16","local_max_k16",
                 "local_std_k16","local_cnt_k32","local_mean_k32","local_max_k32",
                 "local_std_k32"]
    importances = dict(zip(imp_names, clf.feature_importances_.tolist()))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n  Feature importances (top 5):")
    for feat, imp in top5:
        print(f"    {feat:<22} {imp:.4f}  {'#'*int(imp*400)}")

    # ------------------------------------------------------------------
    # Guardar modelo y stats
    # ------------------------------------------------------------------
    import joblib
    joblib.dump(clf, str(out_path))
    size_mb = out_path.stat().st_size / 1e6

    stats = {
        "model_path": str(out_path),
        "n_train_blocks": int(len(train_pts)),
        "n_train_pts_per_block": TRAIN_PTS_PER_BLOCK,
        "n_train_points": int(n_train_pts),
        "n_val_blocks": int(len(val_pts)),
        "n_test_blocks": int(len(test_pts)),
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "oob_score": float(clf.oob_score_),
        "val_metrics":  m_val,
        "test_metrics": m_test,
        "feature_importances": importances,
        "timing": {
            "feat_extraction_train_s": round(t_feat_train, 2),
            "feat_extraction_val_s":   round(t_feat_val, 2),
            "feat_extraction_test_s":  round(t_feat_test, 2),
            "training_s":              round(t_train_total, 2),
        },
        "model_size_mb": round(size_mb, 1),
    }

    stats_path = out_path.parent / "rf_train_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Modelo guardado: {out_path} ({size_mb:.0f} MB)")
    print(f"  Stats:           {stats_path}")
    print(f"  OOB accuracy:    {clf.oob_score_:.4f}")
    print(f"  Val  mIoU:       {m_val['mean_iou']:.4f}  tree_IoU={m_val['tree_iou']:.4f}")
    print(f"  Test mIoU:       {m_test['mean_iou']:.4f}  tree_IoU={m_test['tree_iou']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
