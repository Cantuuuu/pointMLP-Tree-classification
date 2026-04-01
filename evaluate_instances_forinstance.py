"""
Evaluación completa de segmentación de instancias — FORinstance dataset.

Pipeline por flujo:
  1. Segmentación semántica (árbol / no-árbol)
  2. Watershed 2D sobre raster Z-max → ID de instancia por punto
  3. Comparar instancias predichas vs GT treeID

Métricas de instancia:
  PQ  = Panoptic Quality (SQ × RQ)     — métrica estándar de instancias
  SQ  = Segmentation Quality            — IoU medio de pares TP
  RQ  = Recognition Quality             — TP / (TP + 0.5·FP + 0.5·FN)
  mInstIoU = mean Instance IoU          — IoU medio por árbol GT

Uso:
  python evaluate_instances_forinstance.py
  python evaluate_instances_forinstance.py --flow A
  python evaluate_instances_forinstance.py --skip-seg   # usar segmentación cacheada
"""

import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, json, time
from pathlib import Path

import laspy, numpy as np, pandas as pd, yaml, torch, joblib

from src.segmentation.flow_a_geometric    import compute_hag, segmentation_metrics
from src.segmentation.instance_watershed  import segment_instances, evaluate_instances
from src.segmentation.pointnet2_model     import build_model
from src.segmentation.rf_model            import extract_features
from evaluate_itd_forinstance import (
    load_plot, extract_inference_blocks, predict_flow_a,
    predict_flow_b, predict_flow_c,
    NUM_POINTS, BLOCK_SIZE, BLOCK_STRIDE
)

OUT_POINTS_CLASS = 3


# ── Watershed params (sintonizados previamente en DEV) ────────────────────────
WS_CELL_SIZE        = 0.5   # m  — resolución raster
WS_SMOOTH_SIGMA     = 1.0   # celdas — suavizado Gaussiano
WS_LOCAL_MAX_WINDOW = 5     # celdas — ventana de máximos locales


def evaluate_flow_instances(plots, flow_name, predict_fn, iou_threshold=0.5):
    """Evalúa segmentación de instancias de un flujo sobre una lista de plots."""
    all_results = []
    agg = {"tp": 0, "fp": 0, "fn": 0,
           "sq_sum": 0.0, "inst_iou_sum": 0.0, "n_gt_total": 0}

    for p in plots:
        t0 = time.time()
        xyz     = p["xyz"]
        gt_tid  = p["tree_id"]

        # 1. Segmentación semántica
        pred_mask = predict_fn(xyz)
        seg       = segmentation_metrics(pred_mask, (gt_tid > 0).astype(np.int64))

        # 2. Watershed → instance IDs por punto
        inst_ids = segment_instances(
            xyz, pred_mask,
            cell_size=WS_CELL_SIZE,
            smooth_sigma=WS_SMOOTH_SIGMA,
            local_max_window=WS_LOCAL_MAX_WINDOW)

        # 3. Evaluación de instancias
        ev = evaluate_instances(inst_ids, gt_tid, iou_threshold=iou_threshold)

        # Acumular métricas PQ
        agg["tp"]  += ev["tp"];  agg["fp"] += ev["fp"];  agg["fn"] += ev["fn"]
        agg["sq_sum"]       += ev["sq"] * ev["tp"]        # SQ ponderado por TP
        agg["inst_iou_sum"] += ev["mean_inst_iou"] * ev["n_gt"]
        agg["n_gt_total"]   += ev["n_gt"]

        dt = time.time() - t0
        print(f"    {p['site']}/{p['name']:30s} "
              f"seg_F1={seg['f1']:.3f} | "
              f"inst: n_gt={ev['n_gt']:3d} n_pred={ev['n_pred']:3d} "
              f"PQ={ev['pq']:.3f} SQ={ev['sq']:.3f} RQ={ev['rq']:.3f} "
              f"mIoU={ev['mean_inst_iou']:.3f}  ({dt:.1f}s)", flush=True)

        all_results.append({
            "site": p["site"], "plot": p["name"],
            "n_pts": int(len(xyz)),
            "n_gt_trees": ev["n_gt"],
            "segmentation": {k: round(v, 4) for k, v in seg.items()
                             if k not in ("tp","fp","fn","tn")},
            "instances": {k: (round(v,4) if isinstance(v,float) else v)
                          for k, v in ev.items() if k != "matched_pairs"},
        })

    # Métricas agregadas
    tp = agg["tp"]; fp = agg["fp"]; fn = agg["fn"]
    sq_agg  = agg["sq_sum"] / tp if tp > 0 else 0.0
    rq_agg  = tp / (tp + 0.5*fp + 0.5*fn) if (tp+fp+fn) > 0 else 0.0
    pq_agg  = sq_agg * rq_agg
    prec    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_agg  = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    miou    = agg["inst_iou_sum"] / agg["n_gt_total"] if agg["n_gt_total"] > 0 else 0.0

    aggregate = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1_agg, 4),
        "sq": round(sq_agg, 4), "rq": round(rq_agg, 4), "pq": round(pq_agg, 4),
        "mean_inst_iou": round(miou, 4),
    }

    return {"flow": flow_name, "per_plot": all_results, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation_forinstance.yaml")
    parser.add_argument("--flow",   default="ABC")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(cfg["data"]["forinstance_dir"])
    log_dir     = Path(cfg["experiment"]["log_dir"])
    meta        = pd.read_csv(dataset_dir / "data_split_metadata.csv")
    np.random.seed(cfg["experiment"]["seed"])

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

    # Cargar plots test
    def load_split_plots(split):
        rows = meta[meta["split"] == split]
        plots = []
        for _, row in rows.iterrows():
            plot = load_plot(dataset_dir / row["path"])
            if plot is None:
                continue
            plots.append({"name": Path(row["path"]).stem,
                          "site": row["folder"], **plot})
        return plots

    print("\nCargando plots TEST...", flush=True)
    test_plots = load_split_plots("test")
    print(f"  {len(test_plots)} plots cargados")

    hag_thr = cfg["flow_a"]["hag_threshold"]

    flows_to_run = []
    if "0" in args.flow:
        flows_to_run.append(("0", "Watershed Crudo (sin segmentación)",
                             lambda xyz: np.ones(len(xyz), dtype=bool)))
    if "A" in args.flow:
        flows_to_run.append(("A", "Geométrico (HAG)",
                             lambda xyz: predict_flow_a(xyz, hag_thr)))
    if "B" in args.flow and pnet_model is not None:
        flows_to_run.append(("B", "PointNet++",
                             lambda xyz, m=pnet_model, d=device:
                             predict_flow_b(xyz, m, d)))
    if "C" in args.flow and rf_clf is not None:
        flows_to_run.append(("C", "Random Forest",
                             lambda xyz, c=rf_clf: predict_flow_c(xyz, c)))

    all_results = {}
    for fid, fname, predict_fn in flows_to_run:
        print(f"\n{'='*65}\nFlujo {fid} — {fname}\n{'='*65}", flush=True)
        res = evaluate_flow_instances(
            test_plots, fname, predict_fn, args.iou_threshold)
        all_results[fid] = res

        agg = res["aggregate"]
        print(f"\n  TOTAL  PQ={agg['pq']:.4f}  SQ={agg['sq']:.4f}  "
              f"RQ={agg['rq']:.4f}  mInstIoU={agg['mean_inst_iou']:.4f}")
        print(f"         TP={agg['tp']} FP={agg['fp']} FN={agg['fn']}  "
              f"Prec={agg['precision']:.4f} Rec={agg['recall']:.4f} "
              f"F1={agg['f1']:.4f}")

    # Tabla comparativa
    print(f"\n{'='*65}")
    print("TABLA COMPARATIVA — Segmentación de Instancias")
    print(f"{'='*65}")
    print(f"{'Flujo':<22} {'PQ':>7} {'SQ':>7} {'RQ':>7} "
          f"{'mInstIoU':>10} {'F1':>7}")
    print("-"*65)
    NAMES = {"0":"Watershed Crudo","A":"Geometric (HAG)","B":"PointNet++","C":"Random Forest"}
    for fid, res in all_results.items():
        a = res["aggregate"]
        print(f"  {NAMES[fid]:<20} {a['pq']:>7.4f} {a['sq']:>7.4f} "
              f"{a['rq']:>7.4f} {a['mean_inst_iou']:>10.4f} {a['f1']:>7.4f}")

    # Guardar — merge con resultados previos si existen
    out_path = log_dir / "instance_results.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        existing_flows = existing.get("flows", {})
    else:
        existing_flows = {}

    existing_flows.update({fid: {
        "aggregate": r["aggregate"],
        "per_plot": r["per_plot"],
    } for fid, r in all_results.items()})

    out = {
        "watershed_params": {
            "cell_size": WS_CELL_SIZE,
            "smooth_sigma": WS_SMOOTH_SIGMA,
            "local_max_window": WS_LOCAL_MAX_WINDOW,
        },
        "iou_threshold": args.iou_threshold,
        "flows": existing_flows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResultados guardados: {out_path}")


if __name__ == "__main__":
    main()
