"""
Entrenamiento del modelo Random Forest para segmentación semántica LiDAR.

Usa los mismos bloques preprocesados que PointNet++:
  data/processed/segmentation/train/points.npy  — (B, 4096, 3)
  data/processed/segmentation/train/labels.npy  — (B, 4096)

Flujo:
  1. Carga bloques de entrenamiento (submuestreo si hay muchos)
  2. Extrae 15 features por punto con extract_features()
  3. Entrena RandomForestClassifier con class_weight='balanced'
  4. Evalúa en split de validación (mIoU, F1 árbol, F1 no-árbol)
  5. Guarda modelo en results/segmentation/rf_model.pkl

Uso:
  python train_rf.py
  python train_rf.py --max-train-blocks 800 --n-estimators 300
  python train_rf.py --config configs/segmentation.yaml
"""

import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(data_dir: Path, split: str) -> tuple:
    """Carga points.npy y labels.npy de un split."""
    pts  = np.load(data_dir / split / "points.npy")   # (B, N, 3)
    lbls = np.load(data_dir / split / "labels.npy")   # (B, N)
    return pts, lbls


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de segmentación binaria."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    acc  = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    iou  = tp / max(tp + fp + fn, 1)

    prec0 = tn / max(tn + fn, 1)
    rec0  = tn / max(tn + fp, 1)
    f1_0  = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-8)
    iou0  = tn / max(tn + fn + fp, 1)

    miou = (iou + iou0) / 2

    return {
        "accuracy": round(acc, 4),
        "tree_precision": round(prec, 4),
        "tree_recall": round(rec, 4),
        "tree_f1": round(f1, 4),
        "tree_iou": round(iou, 4),
        "notree_f1": round(f1_0, 4),
        "notree_iou": round(iou0, 4),
        "miou": round(miou, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Entrena RF segmentación LiDAR")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--max-train-blocks", type=int, default=800,
                        help="Máx bloques de entrenamiento (submuestreo si hay más)")
    parser.add_argument("--max-val-blocks", type=int, default=200,
                        help="Máx bloques de validación")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--output", default="results/segmentation/rf_model.pkl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["data"]["processed_dir"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Entrenamiento Random Forest — Segmentación LiDAR")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1/4] Cargando datos preprocesados...", flush=True)
    t0 = time.time()

    train_pts, train_lbl = load_split(data_dir, "train")
    val_pts,   val_lbl   = load_split(data_dir, "val")

    print(f"  Train: {len(train_pts)} bloques x {train_pts.shape[1]} pts")
    print(f"  Val:   {len(val_pts)} bloques x {val_pts.shape[1]} pts")
    print(f"  Tree ratio (train): {train_lbl.mean():.3f}", flush=True)

    # Submuestreo
    rng = np.random.default_rng(42)
    if len(train_pts) > args.max_train_blocks:
        sel = rng.choice(len(train_pts), args.max_train_blocks, replace=False)
        train_pts = train_pts[sel]
        train_lbl = train_lbl[sel]
        print(f"  Submuestreado train a {args.max_train_blocks} bloques")

    if len(val_pts) > args.max_val_blocks:
        sel = rng.choice(len(val_pts), args.max_val_blocks, replace=False)
        val_pts = val_pts[sel]
        val_lbl = val_lbl[sel]
        print(f"  Submuestreado val a {args.max_val_blocks} bloques")

    print(f"  Carga: {time.time()-t0:.1f}s", flush=True)

    # 2. Extracción de features
    print("\n[2/4] Extrayendo features...", flush=True)
    from src.segmentation.rf_model import extract_features_batch, RFSegmentationModel

    t0 = time.time()
    X_train, _ = extract_features_batch(train_pts, verbose=True)
    y_train = train_lbl.ravel()
    t_feat_train = time.time() - t0
    print(f"  Train features: {X_train.shape}  tiempo: {t_feat_train:.1f}s")

    t0 = time.time()
    X_val, _ = extract_features_batch(val_pts, verbose=False)
    y_val = val_lbl.ravel()
    t_feat_val = time.time() - t0
    print(f"  Val features:   {X_val.shape}  tiempo: {t_feat_val:.1f}s", flush=True)

    # 3. Entrenamiento
    print("\n[3/4] Entrenando Random Forest...", flush=True)
    print(f"  n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"class_weight=balanced, oob_score=True", flush=True)

    model = RFSegmentationModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0

    print(f"  Tiempo entrenamiento: {t_train:.1f}s")
    if "oob_score" in model.train_stats:
        print(f"  OOB score: {model.train_stats['oob_score']:.4f}", flush=True)

    # 4. Evaluación
    print("\n[4/4] Evaluando en validación...", flush=True)
    t0 = time.time()
    y_pred_val = model.predict(X_val)
    t_infer_val = time.time() - t0

    metrics_val = compute_metrics(y_val, y_pred_val)

    print(f"\n  === Métricas de Validación ===")
    print(f"  Accuracy:      {metrics_val['accuracy']:.4f}")
    print(f"  Tree  F1:      {metrics_val['tree_f1']:.4f}")
    print(f"  Tree  IoU:     {metrics_val['tree_iou']:.4f}")
    print(f"  NoTree F1:     {metrics_val['notree_f1']:.4f}")
    print(f"  mIoU:          {metrics_val['miou']:.4f}")
    print(f"  Inferencia val:{t_infer_val:.2f}s ({len(y_val)/1e6:.1f}M pts)")

    # Feature importances
    print(f"\n  === Feature Importances (top 10) ===")
    imps = model.feature_importances()
    top10 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in top10:
        bar = "#" * int(imp * 400)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    # Guardar
    model.save(str(out_path))

    # Stats JSON
    stats = {
        "model_path": str(out_path),
        "n_train_blocks": len(train_pts),
        "n_val_blocks": len(val_pts),
        "n_train_points": int(len(y_train)),
        "n_val_points": int(len(y_val)),
        "n_features": model.n_features,
        "feature_names": model.feature_names,
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        },
        "train_stats": model.train_stats,
        "val_metrics": metrics_val,
        "feature_importances": imps,
        "timing": {
            "feat_extraction_train_s": round(t_feat_train, 2),
            "feat_extraction_val_s": round(t_feat_val, 2),
            "training_s": round(t_train, 2),
            "inference_val_s": round(t_infer_val, 3),
        },
        "model_size_mb": round(model.model_size_mb(), 2),
    }

    stats_path = out_path.parent / "rf_train_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats: {stats_path}")
    print(f"\n  === Resumen ===")
    print(f"  mIoU val: {metrics_val['miou']:.4f}  |  "
          f"Tree F1 val: {metrics_val['tree_f1']:.4f}")
    print(f"  Modelo: {out_path} ({model.model_size_mb():.1f} MB)")
    print(f"  Entrenamiento completado en {t_train:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
