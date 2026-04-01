"""
Visualización de segmentación de instancias — FORinstance.

Genera figuras comparativas por plot:
  Panel 1 — ANTES   : nube cruda coloreada por altura (Z)
  Panel 2 — DESPUÉS : instancias predichas (cada árbol = un color)
  Panel 3 — ESPERADO: GT treeID (ground truth)

Vista: proyección XY cenital (top-down) — análoga a vista de dron.
       Sub-figura lateral XZ para contexto de altura.

Uso:
  python visualize_instances_forinstance.py
  python visualize_instances_forinstance.py --plot CULS/plot_2_annotated.las
  python visualize_instances_forinstance.py --flow A --max-plots 3
"""

import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import yaml
import joblib

from src.segmentation.instance_watershed import segment_instances
from src.segmentation.flow_a_geometric   import compute_hag
from src.segmentation.pointnet2_model    import build_model
from evaluate_itd_forinstance import (
    load_plot, predict_flow_a, predict_flow_b, predict_flow_c
)

OUT_POINTS_CLASS = 3
MAX_VIS_PTS      = 80_000   # subsample for fast plotting
WS_CELL_SIZE     = 0.5
WS_SMOOTH_SIGMA  = 1.0
WS_LOCAL_MAX_WIN = 5


# ── Color utilities ───────────────────────────────────────────────────────────

def _instance_colors(ids: np.ndarray, background_color=(0.85, 0.85, 0.85)):
    """Map instance IDs to RGBA colors. 0/-1 → background grey."""
    unique_ids = sorted(set(int(i) for i in ids if i > 0))
    # Use a qualitative colormap for up to 20 colors, then repeat
    cmap_base  = plt.cm.get_cmap("tab20", max(len(unique_ids), 1))
    id_to_color = {uid: cmap_base(i % 20) for i, uid in enumerate(unique_ids)}

    colors = np.ones((len(ids), 4), dtype=np.float32)
    colors[:, :3] = background_color
    colors[:, 3]  = 0.3   # semi-transparent background
    for uid, col in id_to_color.items():
        mask = ids == uid
        colors[mask] = col
        colors[mask, 3] = 0.9
    return colors, id_to_color


def _height_colors(z: np.ndarray):
    """Map Z values to RGBA using a green→yellow→white colormap."""
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    cmap   = plt.cm.get_cmap("YlGn")
    return cmap(z_norm)


# ── Single plot figure ────────────────────────────────────────────────────────

def visualize_plot(xyz, gt_tid, pred_mask, inst_ids, plot_name, site,
                   out_path: Path, flow_label: str):
    """4-panel figure: input height | predicted instances | GT instances | stats."""

    # Subsample for speed
    n = len(xyz)
    if n > MAX_VIS_PTS:
        idx = np.random.choice(n, MAX_VIS_PTS, replace=False)
    else:
        idx = np.arange(n)

    xyz_s   = xyz[idx]
    gt_s    = gt_tid[idx]
    inst_s  = inst_ids[idx]
    mask_s  = pred_mask[idx]

    x, y, z = xyz_s[:, 0], xyz_s[:, 1], xyz_s[:, 2]
    # Centre coordinates for cleaner axes
    x = x - x.mean(); y = y - y.mean()

    # ── Colors ────────────────────────────────────────────────────────────
    col_height = _height_colors(z)

    col_pred, pred_id_map = _instance_colors(inst_s)

    # GT colors
    col_gt, gt_id_map = _instance_colors(gt_s)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5.5), facecolor="#1a1a2e")
    fig.suptitle(
        f"{site} / {plot_name.replace('_annotated','')}  —  {flow_label}",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    axes_cfg = dict(facecolor="#0f0f1e")
    pt_size  = max(0.3, 20_000 / len(idx))

    # Panel 1 — ANTES (altura Z)
    ax1 = fig.add_subplot(1, 3, 1, **axes_cfg)
    ax1.scatter(x, y, c=col_height, s=pt_size, linewidths=0)
    ax1.set_title("ANTES\n(Nube cruda — altura Z)", color="white", fontsize=10)
    _style_ax(ax1)

    # Panel 2 — DESPUÉS (instancias predichas)
    ax2 = fig.add_subplot(1, 3, 2, **axes_cfg)
    ax2.scatter(x, y, c=col_pred, s=pt_size, linewidths=0)
    n_inst = len(pred_id_map)
    ax2.set_title(f"DESPUÉS\n(Instancias predichas — {n_inst} árboles)",
                  color="white", fontsize=10)
    _style_ax(ax2)

    # Panel 3 — ESPERADO (GT)
    ax3 = fig.add_subplot(1, 3, 3, **axes_cfg)
    ax3.scatter(x, y, c=col_gt, s=pt_size, linewidths=0)
    n_gt = len(gt_id_map)
    ax3.set_title(f"ESPERADO\n(GT treeID — {n_gt} árboles)",
                  color="white", fontsize=10)
    _style_ax(ax3)

    # Color bar for height (panel 1)
    sm = plt.cm.ScalarMappable(cmap="YlGn",
                               norm=mcolors.Normalize(z.min(), z.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label("Altura (m)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    # Stats annotation
    n_pred_tree = int((inst_ids > 0).sum())
    n_gt_tree   = int((gt_tid > 0).sum())
    seg_rec     = n_pred_tree / n_gt_tree if n_gt_tree > 0 else 0
    stats_txt   = (f"Puntos totales: {n:,}\n"
                   f"Puntos árbol GT: {n_gt_tree:,}\n"
                   f"Puntos árbol pred: {n_pred_tree:,}\n"
                   f"Instancias predichas: {n_inst}\n"
                   f"Instancias GT: {n_gt}")
    fig.text(0.5, -0.04, stats_txt, ha="center", color="#aaaacc",
             fontsize=8, fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Guardado: {out_path}")


def _style_ax(ax):
    ax.set_aspect("equal")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.set_xlabel("X (m)", color="#aaaacc", fontsize=8)
    ax.set_ylabel("Y (m)", color="#aaaacc", fontsize=8)
    ax.tick_params(axis="x", colors="#aaaacc")
    ax.tick_params(axis="y", colors="#aaaacc")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/segmentation_forinstance.yaml")
    parser.add_argument("--flow",      default="A",
                        help="Flujo a visualizar: A, B o C")
    parser.add_argument("--max-plots", type=int, default=11,
                        help="Número máximo de plots a visualizar")
    parser.add_argument("--split",     default="test",
                        help="Split a usar: dev o test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = Path(cfg["data"]["forinstance_dir"])
    log_dir     = Path(cfg["experiment"]["log_dir"])
    out_dir     = log_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta        = pd.read_csv(dataset_dir / "data_split_metadata.csv")
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cargar modelo según flujo
    fid = args.flow.upper()
    flow_labels = {"0": "Watershed Crudo (sin segmentación)",
                   "A": "Geométrico (HAG)", "B": "PointNet++", "C": "Random Forest"}
    flow_label  = flow_labels.get(fid, fid)

    pnet_model = rf_clf = None
    hag_thr    = cfg["flow_a"]["hag_threshold"]

    if fid == "B":
        print("Cargando PointNet++...")
        pnet_model = build_model(cfg).to(device)
        ckpt = torch.load(log_dir / "best_model.pth",
                          weights_only=False, map_location=device)
        pnet_model.load_state_dict(ckpt["model_state_dict"])
        pnet_model.eval()
    elif fid == "C":
        print("Cargando Random Forest...")
        rf_clf = joblib.load(log_dir / "rf_model.pkl")

    if fid == "0":
        predict_fn = lambda xyz: np.ones(len(xyz), dtype=bool)
    elif fid == "A":
        predict_fn = lambda xyz: predict_flow_a(xyz, hag_thr)
    elif fid == "B":
        predict_fn = lambda xyz, m=pnet_model, d=device: predict_flow_b(xyz, m, d)
    else:
        predict_fn = lambda xyz, c=rf_clf: predict_flow_c(xyz, c)

    # Cargar plots
    rows = meta[meta["split"] == args.split].head(args.max_plots)
    print(f"\nGenerando {min(len(rows), args.max_plots)} visualizaciones "
          f"(Flujo {fid} — {flow_label})...\n")

    for _, row in rows.iterrows():
        las_path = dataset_dir / row["path"]
        plot_data = load_plot(las_path)
        if plot_data is None:
            continue

        plot_name = Path(row["path"]).stem
        site      = row["folder"]
        xyz       = plot_data["xyz"]
        gt_tid    = plot_data["tree_id"]

        print(f"  Procesando {site}/{plot_name} ({len(xyz):,} pts)...",
              flush=True)

        t0 = __import__("time").time()
        pred_mask = predict_fn(xyz)
        inst_ids  = segment_instances(
            xyz, pred_mask,
            cell_size=WS_CELL_SIZE,
            smooth_sigma=WS_SMOOTH_SIGMA,
            local_max_window=WS_LOCAL_MAX_WIN)
        dt = __import__("time").time() - t0

        n_inst = int((inst_ids > 0).max() if (inst_ids > 0).any() else 0)
        n_inst = len(set(int(i) for i in inst_ids if i > 0))
        n_gt   = len(set(int(i) for i in gt_tid if i > 0))
        print(f"    → {n_inst} instancias predichas | {n_gt} GT | {dt:.1f}s")

        out_path = out_dir / f"flow{fid}_{site}_{plot_name}.png"
        visualize_plot(
            xyz, gt_tid, pred_mask, inst_ids,
            plot_name, site, out_path, flow_label)

    print(f"\nTodas las figuras guardadas en: {out_dir}")


if __name__ == "__main__":
    main()
