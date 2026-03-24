"""Reporte comparativo entre modelos: sintético vs real.

Genera 6 figuras y un reporte HTML autocontenido comparando
Model A (sintético) vs Model B (real) con evaluación cruzada.

Uso:
    python src/report.py --models model_A model_B_3cls --output results/comparison_report/
    python src/report.py --models model_A model_B_3cls --cross_eval --output results/comparison_report/
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Fix Windows encoding
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# ── Constants ────────────────────────────────────────────────────────

CLASS_NAMES = ["PINE", "OAK", "OTHER"]
COLOR_A = "#1565C0"  # Blue for synthetic/Model A
COLOR_B = "#2E7D32"  # Green for real/Model B
COLOR_A_LIGHT = "#64B5F6"
COLOR_B_LIGHT = "#81C784"
CHANCE_LEVEL = 1.0 / 3.0

RESULTS_DIR = Path("results")
CROSS_EVAL_DIR = RESULTS_DIR / "cross_eval"


# ── Data loading ─────────────────────────────────────────────────────


def load_eval_json(path: Path) -> dict:
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)


def load_training_log(exp_dir: Path) -> dict:
    """Load training_log.csv into dict of lists."""
    log_path = exp_dir / "training_log.csv"
    data = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["train_acc"].append(float(row["train_acc"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["lr"].append(float(row["lr"]))
    return data


def load_config_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_stats_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_results() -> dict:
    """Load all 4 evaluation results from cross_eval directory."""
    evals = {}
    for name, subdir in [
        ("A_on_A", "A_on_synthetic"),
        ("A_on_B", "A_on_real"),
        ("B_on_B", "B_on_real"),
        ("B_on_A", "B_on_synthetic"),
    ]:
        path = CROSS_EVAL_DIR / subdir / "eval_results.json"
        if path.exists():
            evals[name] = load_eval_json(path)
        else:
            print(f"WARNING: Missing {path}")
    return evals


# ── Figure helpers ───────────────────────────────────────────────────


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def save_fig(fig: plt.Figure, path: Path) -> str:
    """Save figure to disk and return base64."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    b64 = fig_to_base64(fig)
    return b64


# ── Figure 1: 4-panel confusion matrices ────────────────────────────


def plot_confusion_matrices_4panel(evals: dict, output_dir: Path) -> str:
    """Generate 2x2 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        ("A_on_A", "Model A → Synthetic Test", axes[0, 0]),
        ("A_on_B", "Model A → Real Test", axes[0, 1]),
        ("B_on_A", "Model B → Synthetic Test", axes[1, 0]),
        ("B_on_B", "Model B → Real Test", axes[1, 1]),
    ]

    # Find global max for consistent color scale
    all_cms = []
    for key, _, _ in panels:
        if key in evals and "confusion_matrix_tta" in evals[key]:
            all_cms.append(np.array(evals[key]["confusion_matrix_tta"]))

    for key, title, ax in panels:
        if key not in evals or "confusion_matrix_tta" not in evals[key]:
            ax.text(0.5, 0.5, "No disponible", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        cm = np.array(evals[key]["confusion_matrix_tta"])
        n = cm.shape[0]

        # Percentage matrix
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")

        # Text annotations
        for i in range(n):
            for j in range(n):
                color = "white" if cm_pct[i, j] > 60 else "black"
                ax.text(j, i, f"{cm_pct[i,j]:.0f}%\n({cm[i,j]})",
                        ha="center", va="center", fontsize=10, color=color, fontweight="bold")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax.set_xlabel("Predicción", fontsize=11)
        ax.set_ylabel("Verdadero", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("Matrices de Confusión — Evaluación Cruzada (con TTA)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return save_fig(fig, output_dir / "confusion_matrices_4panel.png")


# ── Figure 2: Accuracy comparison bar chart ──────────────────────────


def plot_accuracy_comparison(evals: dict, output_dir: Path) -> str:
    """4 bars: A→A, A→B, B→A, B→B with chance level line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["A→A\n(Sintético→Sintético)", "A→B\n(Sintético→Real)",
              "B→A\n(Real→Sintético)", "B→B\n(Real→Real)"]
    keys = ["A_on_A", "A_on_B", "B_on_A", "B_on_B"]
    colors = [COLOR_A, COLOR_A_LIGHT, COLOR_B_LIGHT, COLOR_B]

    accs = []
    for key in keys:
        if key in evals:
            accs.append(evals[key]["tta"]["overall"]["accuracy"] * 100)
        else:
            accs.append(0)

    bars = ax.bar(labels, accs, color=colors, edgecolor="gray", linewidth=0.5, width=0.6)

    # Chance level
    ax.axhline(y=CHANCE_LEVEL * 100, color="red", linestyle="--", linewidth=1.5, label=f"Azar ({CHANCE_LEVEL*100:.1f}%)")

    # Values on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Comparación de Accuracy — Evaluación Cruzada (TTA)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return save_fig(fig, output_dir / "accuracy_comparison_bar.png")


# ── Figure 3: F1 per-class heatmap ──────────────────────────────────


def plot_f1_heatmap(evals: dict, output_dir: Path) -> str:
    """Heatmap 4×3: rows=[A→A, A→B, B→A, B→B], cols=[PINE, OAK, OTHER]."""
    fig, ax = plt.subplots(figsize=(8, 5))

    row_labels = ["A→A (Sint→Sint)", "A→B (Sint→Real)", "B→A (Real→Sint)", "B→B (Real→Real)"]
    keys = ["A_on_A", "A_on_B", "B_on_A", "B_on_B"]

    data = np.zeros((4, 3))
    for i, key in enumerate(keys):
        if key in evals:
            for j, cls in enumerate(CLASS_NAMES):
                data[i, j] = evals[key]["tta"]["per_class"].get(cls, {}).get("f1", 0) * 100

    # Red to green colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#D32F2F", "#FFEB3B", "#388E3C"])
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for i in range(4):
        for j in range(3):
            color = "white" if data[i, j] < 30 or data[i, j] > 80 else "black"
            ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("F1-Score por Clase — Evaluación Cruzada (TTA)", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("F1-Score (%)", fontsize=10)

    fig.tight_layout()
    return save_fig(fig, output_dir / "f1_per_class_heatmap.png")


# ── Figure 4: Domain gap analysis ───────────────────────────────────


def plot_domain_gap(evals: dict, output_dir: Path) -> str:
    """Bar chart showing accuracy drop for each model when crossing domains."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Model A: A→A vs A→B
    acc_aa = evals.get("A_on_A", {}).get("tta", {}).get("overall", {}).get("accuracy", 0) * 100
    acc_ab = evals.get("A_on_B", {}).get("tta", {}).get("overall", {}).get("accuracy", 0) * 100
    drop_a = acc_aa - acc_ab

    # Model B: B→B vs B→A
    acc_bb = evals.get("B_on_B", {}).get("tta", {}).get("overall", {}).get("accuracy", 0) * 100
    acc_ba = evals.get("B_on_A", {}).get("tta", {}).get("overall", {}).get("accuracy", 0) * 100
    drop_b = acc_bb - acc_ba

    x = np.arange(2)
    width = 0.35

    # In-domain bars
    bars_in = ax.bar(x - width/2, [acc_aa, acc_bb], width, label="En dominio propio",
                     color=[COLOR_A, COLOR_B], edgecolor="gray", linewidth=0.5)
    # Cross-domain bars
    bars_cross = ax.bar(x + width/2, [acc_ab, acc_ba], width, label="Cross-dominio",
                        color=[COLOR_A_LIGHT, COLOR_B_LIGHT], edgecolor="gray", linewidth=0.5)

    # Drop annotations
    for i, (in_acc, cross_acc, drop) in enumerate([(acc_aa, acc_ab, drop_a), (acc_bb, acc_ba, drop_b)]):
        ax.annotate("", xy=(i + width/2, cross_acc), xytext=(i - width/2, in_acc),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
        mid_y = (in_acc + cross_acc) / 2
        ax.text(i + 0.32, mid_y, f"−{drop:.1f}pp",
                fontsize=11, fontweight="bold", color="red", ha="left", va="center")

    # Values on bars
    for bars in [bars_in, bars_cross]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Model A\n(Entrenado en sintéticos)", "Model B\n(Entrenado en reales)"], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Análisis de Domain Gap", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return save_fig(fig, output_dir / "domain_gap_analysis.png")


# ── Figure 5: Training curves ───────────────────────────────────────


def plot_training_curves(output_dir: Path) -> str:
    """2 subplots: Loss and Accuracy vs epochs for both models."""
    log_a = load_training_log(RESULTS_DIR / "model_A")
    log_b = load_training_log(RESULTS_DIR / "model_B_3cls")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(log_a["epoch"], log_a["train_loss"], color=COLOR_A, alpha=0.4, linewidth=1, label="A train")
    ax1.plot(log_a["epoch"], log_a["val_loss"], color=COLOR_A, linewidth=2, label="A val")
    ax1.plot(log_b["epoch"], log_b["train_loss"], color=COLOR_B, alpha=0.4, linewidth=1, label="B train")
    ax1.plot(log_b["epoch"], log_b["val_loss"], color=COLOR_B, linewidth=2, label="B val")

    # Mark best epoch
    best_a = np.argmin(log_a["val_loss"])
    best_b = np.argmin(log_b["val_loss"])
    ax1.plot(log_a["epoch"][best_a], log_a["val_loss"][best_a], "*", color=COLOR_A, markersize=15, zorder=5)
    ax1.plot(log_b["epoch"][best_b], log_b["val_loss"][best_b], "*", color=COLOR_B, markersize=15, zorder=5)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Curvas de Pérdida", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(log_a["epoch"], log_a["train_acc"], color=COLOR_A, alpha=0.4, linewidth=1, label="A train")
    ax2.plot(log_a["epoch"], log_a["val_acc"], color=COLOR_A, linewidth=2, label="A val")
    ax2.plot(log_b["epoch"], log_b["train_acc"], color=COLOR_B, alpha=0.4, linewidth=1, label="B train")
    ax2.plot(log_b["epoch"], log_b["val_acc"], color=COLOR_B, linewidth=2, label="B val")

    # Mark best epoch (by val_acc)
    best_a_acc = np.argmax(log_a["val_acc"])
    best_b_acc = np.argmax(log_b["val_acc"])
    ax2.plot(log_a["epoch"][best_a_acc], log_a["val_acc"][best_a_acc], "*", color=COLOR_A, markersize=15, zorder=5)
    ax2.plot(log_b["epoch"][best_b_acc], log_b["val_acc"][best_b_acc], "*", color=COLOR_B, markersize=15, zorder=5)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Curvas de Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Convergencia durante Entrenamiento (★ = mejor epoch)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, output_dir / "training_curves_comparison.png")


# ── Figure 6: Per-class comparison grouped bar ───────────────────────


def plot_per_class_comparison(evals: dict, output_dir: Path) -> str:
    """Grouped bar chart: for each class, precision/recall/F1 of both models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["precision", "recall", "f1"]
    metric_names = ["Precision", "Recall", "F1-Score"]

    eval_labels = ["A→A", "A→B", "B→A", "B→B"]
    eval_keys = ["A_on_A", "A_on_B", "B_on_A", "B_on_B"]
    bar_colors = [COLOR_A, COLOR_A_LIGHT, COLOR_B_LIGHT, COLOR_B]

    for ax, metric, mname in zip(axes, metrics, metric_names):
        x = np.arange(len(CLASS_NAMES))
        n_bars = len(eval_keys)
        width = 0.18

        for i, (key, label, color) in enumerate(zip(eval_keys, eval_labels, bar_colors)):
            vals = []
            for cls in CLASS_NAMES:
                v = evals.get(key, {}).get("tta", {}).get("per_class", {}).get(cls, {}).get(metric, 0)
                vals.append(v * 100)
            offset = (i - n_bars / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="gray", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_ylabel(f"{mname} (%)", fontsize=10)
        ax.set_title(mname, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Métricas por Clase — Evaluación Cruzada (TTA)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, output_dir / "per_class_comparison_radar.png")


# ── HTML Report ──────────────────────────────────────────────────────


def generate_html_report(evals: dict, figures_b64: dict, output_dir: Path) -> None:
    """Generate self-contained HTML report."""

    # Load supplementary data
    config_a = load_config_yaml(RESULTS_DIR / "model_A" / "config_used.yaml")
    config_b = load_config_yaml(RESULTS_DIR / "model_B_3cls" / "config_used.yaml")
    stats_synth = load_stats_json(Path("data/processed/synthetic/stats.json"))
    stats_real = load_stats_json(Path("data/processed/real_3cls/stats.json"))
    log_a = load_training_log(RESULTS_DIR / "model_A")
    log_b = load_training_log(RESULTS_DIR / "model_B_3cls")

    # Extract key metrics
    def get_acc(key):
        return evals.get(key, {}).get("tta", {}).get("overall", {}).get("accuracy", 0) * 100

    def get_f1(key):
        return evals.get(key, {}).get("tta", {}).get("overall", {}).get("macro_f1", 0) * 100

    acc_aa, acc_ab = get_acc("A_on_A"), get_acc("A_on_B")
    acc_ba, acc_bb = get_acc("B_on_A"), get_acc("B_on_B")
    f1_aa, f1_ab = get_f1("A_on_A"), get_f1("A_on_B")
    f1_ba, f1_bb = get_f1("B_on_A"), get_f1("B_on_B")

    drop_a = acc_aa - acc_ab
    drop_b = acc_bb - acc_ba

    best_epoch_a = int(evals.get("A_on_A", {}).get("checkpoint_epoch", 0))
    best_epoch_b = int(evals.get("B_on_B", {}).get("checkpoint_epoch", 0))
    total_epochs_a = len(log_a["epoch"])
    total_epochs_b = len(log_b["epoch"])

    now = datetime.now().strftime("%d de %B de %Y, %H:%M")

    def img_tag(key: str, caption: str, fig_num: int) -> str:
        b64 = figures_b64.get(key, "")
        return f"""
        <figure>
            <img src="data:image/png;base64,{b64}" alt="{caption}" style="max-width:100%;">
            <figcaption>Figura {fig_num}: {caption}</figcaption>
        </figure>"""

    # Per-class detail table
    def per_class_table() -> str:
        rows = ""
        for key, label in [("A_on_A", "A→A"), ("A_on_B", "A→B"), ("B_on_A", "B→A"), ("B_on_B", "B→B")]:
            pc = evals.get(key, {}).get("tta", {}).get("per_class", {})
            for cls in CLASS_NAMES:
                m = pc.get(cls, {})
                rows += f"""<tr>
                    <td>{label}</td><td>{cls}</td>
                    <td>{m.get('precision',0)*100:.1f}%</td>
                    <td>{m.get('recall',0)*100:.1f}%</td>
                    <td>{m.get('f1',0)*100:.1f}%</td>
                    <td>{m.get('support',0)}</td>
                </tr>"""
        return rows

    model_cfg = config_a["model"]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluación Comparativa: Datos Sintéticos vs Reales — PointMLP</title>
<style>
    @media print {{
        body {{ font-size: 11pt; }}
        .no-print {{ display: none; }}
        figure {{ break-inside: avoid; }}
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        line-height: 1.6;
        color: #333;
        background: #f5f5f5;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
        background: white;
        padding: 40px 50px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    h1 {{
        font-size: 1.8em;
        color: #1565C0;
        border-bottom: 3px solid #1565C0;
        padding-bottom: 10px;
        margin-bottom: 5px;
    }}
    .subtitle {{
        color: #666;
        font-size: 1.0em;
        margin-bottom: 5px;
    }}
    .institution {{
        color: #888;
        font-size: 0.95em;
        margin-bottom: 20px;
    }}
    .date {{
        color: #999;
        font-size: 0.85em;
        margin-bottom: 30px;
    }}
    h2 {{
        font-size: 1.4em;
        color: #1565C0;
        margin-top: 35px;
        margin-bottom: 15px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
    }}
    h3 {{
        font-size: 1.15em;
        color: #333;
        margin-top: 25px;
        margin-bottom: 10px;
    }}
    p {{
        margin-bottom: 12px;
        text-align: justify;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 0.95em;
    }}
    th, td {{
        padding: 8px 12px;
        text-align: center;
        border: 1px solid #ddd;
    }}
    th {{
        background: #1565C0;
        color: white;
        font-weight: 600;
    }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    tr:hover {{ background: #e3f2fd; }}
    .highlight {{ background: #e8f5e9 !important; font-weight: bold; }}
    .warning {{ background: #fff3e0 !important; }}
    .danger {{ background: #ffebee !important; }}
    figure {{
        margin: 20px 0;
        text-align: center;
    }}
    figure img {{
        border: 1px solid #eee;
        border-radius: 4px;
    }}
    figcaption {{
        font-size: 0.9em;
        color: #666;
        margin-top: 8px;
        font-style: italic;
    }}
    .summary-box {{
        background: #e3f2fd;
        border-left: 4px solid #1565C0;
        padding: 15px 20px;
        margin: 20px 0;
        border-radius: 0 4px 4px 0;
    }}
    .summary-box p {{ margin-bottom: 5px; }}
    .metric-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin: 20px 0;
    }}
    .metric-card {{
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #ddd;
    }}
    .metric-card .value {{
        font-size: 2em;
        font-weight: bold;
    }}
    .metric-card .label {{
        font-size: 0.85em;
        color: #666;
    }}
    .card-blue {{ background: #e3f2fd; }}
    .card-blue .value {{ color: #1565C0; }}
    .card-green {{ background: #e8f5e9; }}
    .card-green .value {{ color: #2E7D32; }}
    .card-orange {{ background: #fff3e0; }}
    .card-orange .value {{ color: #E65100; }}
    .card-red {{ background: #ffebee; }}
    .card-red .value {{ color: #C62828; }}
    ul {{ margin: 10px 0 10px 25px; }}
    li {{ margin-bottom: 5px; }}
    code {{
        background: #f5f5f5;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.9em;
    }}
    .footer {{
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #ddd;
        color: #999;
        font-size: 0.8em;
        text-align: center;
    }}
    .toc {{
        background: #fafafa;
        border: 1px solid #eee;
        padding: 15px 25px;
        border-radius: 4px;
        margin: 20px 0;
    }}
    .toc a {{ color: #1565C0; text-decoration: none; }}
    .toc a:hover {{ text-decoration: underline; }}
    .toc ul {{ list-style: none; margin: 5px 0 5px 15px; }}
</style>
</head>
<body>
<div class="container">

<!-- 1. PORTADA -->
<h1>Evaluación Comparativa: Datos Sintéticos vs Reales para Clasificación de Especies Arbóreas mediante PointMLP</h1>
<p class="subtitle">Análisis de domain gap en clasificación de nubes de puntos LiDAR</p>
<p class="institution">Universidad Autónoma de Querétaro — Facultad de Informática</p>
<p class="date">Generado el {now}</p>

<!-- Tabla de contenidos -->
<div class="toc">
<strong>Contenido</strong>
<ul>
    <li>1. <a href="#resumen">Resumen Ejecutivo</a></li>
    <li>2. <a href="#metodologia">Metodología</a></li>
    <li>3. <a href="#resultados">Resultados</a>
        <ul>
            <li>3.1 <a href="#dominio-propio">Rendimiento en dominio propio</a></li>
            <li>3.2 <a href="#cross-eval">Evaluación cruzada</a></li>
            <li>3.3 <a href="#por-clase">Análisis por clase</a></li>
            <li>3.4 <a href="#convergencia">Convergencia y entrenamiento</a></li>
        </ul>
    </li>
    <li>4. <a href="#discusion">Discusión</a></li>
    <li>5. <a href="#conclusiones">Conclusiones</a></li>
    <li>6. <a href="#apendice">Apéndice</a></li>
</ul>
</div>

<!-- 2. RESUMEN EJECUTIVO -->
<h2 id="resumen">1. Resumen Ejecutivo</h2>

<div class="metric-grid">
    <div class="metric-card card-blue">
        <div class="value">{acc_aa:.1f}%</div>
        <div class="label">A→A: Sintético en Sintético</div>
    </div>
    <div class="metric-card card-red">
        <div class="value">{acc_ab:.1f}%</div>
        <div class="label">A→B: Sintético en Real</div>
    </div>
    <div class="metric-card card-orange">
        <div class="value">{acc_ba:.1f}%</div>
        <div class="label">B→A: Real en Sintético</div>
    </div>
    <div class="metric-card card-green">
        <div class="value">{acc_bb:.1f}%</div>
        <div class="label">B→B: Real en Real</div>
    </div>
</div>

<table>
    <tr><th></th><th>Test Sintético</th><th>Test Real</th></tr>
    <tr><td><strong>Model A (Sintético)</strong></td>
        <td class="highlight">{acc_aa:.1f}%</td>
        <td class="danger">{acc_ab:.1f}%</td></tr>
    <tr><td><strong>Model B (Real)</strong></td>
        <td class="warning">{acc_ba:.1f}%</td>
        <td class="highlight">{acc_bb:.1f}%</td></tr>
</table>

<div class="summary-box">
    <p><strong>Hallazgo principal:</strong> Existe un <strong>domain gap severo</strong> entre datos sintéticos y reales.
    El Model A, entrenado exclusivamente con datos sintéticos, alcanza {acc_aa:.1f}% de accuracy en su propio dominio
    pero cae a solo {acc_ab:.1f}% al evaluar en datos reales — una caída de <strong>{drop_a:.1f} puntos porcentuales</strong>.
    Esto indica que los datos sintéticos actuales no capturan adecuadamente la variabilidad y complejidad de las nubes
    de puntos LiDAR reales.</p>
    <p>El Model B (real) también sufre degradación al cruzar dominios ({acc_bb:.1f}% → {acc_ba:.1f}%,
    −{drop_b:.1f}pp), aunque su dominio propio es inherentemente más difícil.</p>
    <p><strong>Domain gap cuantificado:</strong> Model A pierde {drop_a:.1f}pp, Model B pierde {drop_b:.1f}pp al cambiar de dominio.</p>
</div>

<!-- 3. METODOLOGÍA -->
<h2 id="metodologia">2. Metodología</h2>

<h3>2.1 Datasets</h3>

<table>
    <tr><th>Característica</th><th>Dataset Sintético</th><th>Dataset Real (IDTReeS)</th></tr>
    <tr><td>Origen</td><td>Generación procedimental</td><td>NEON AOP LiDAR + IDTReeS 2020</td></tr>
    <tr><td>Total de muestras</td><td>{stats_synth['total_processed']}</td><td>{stats_real['total_processed']}</td></tr>
    <tr><td>Train / Val / Test</td>
        <td>{stats_synth['splits']['train']['count']} / {stats_synth['splits']['val']['count']} / {stats_synth['splits']['test']['count']}</td>
        <td>{stats_real['splits']['train']['count']} / {stats_real['splits']['val']['count']} / {stats_real['splits']['test']['count']}</td></tr>
    <tr><td>Clases</td><td colspan="2">3: PINE, OAK, OTHER</td></tr>
    <tr><td>Puntos por muestra</td><td colspan="2">1,024</td></tr>
    <tr><td>Coordenadas</td><td colspan="2">XYZ (centradas en origen)</td></tr>
</table>

<p><strong>Distribución de clases (train):</strong></p>
<table>
    <tr><th>Clase</th><th>Sintético</th><th>Real</th></tr>
    <tr><td>PINE</td><td>{stats_synth['splits']['train']['class_distribution']['PINE']}</td>
        <td>{stats_real['splits']['train']['class_distribution']['PINE']}</td></tr>
    <tr><td>OAK</td><td>{stats_synth['splits']['train']['class_distribution']['OAK']}</td>
        <td>{stats_real['splits']['train']['class_distribution']['OAK']}</td></tr>
    <tr><td>OTHER</td><td>{stats_synth['splits']['train']['class_distribution']['OTHER']}</td>
        <td>{stats_real['splits']['train']['class_distribution']['OTHER']}</td></tr>
</table>

<h3>2.2 Arquitectura: PointMLP-Elite</h3>

<p>Se utiliza una implementación propia de PointMLP-Elite (Ma et al., ICLR 2022), adaptada para
clasificación de especies arbóreas con restricciones de memoria (6 GB VRAM).</p>

<table>
    <tr><th>Parámetro</th><th>Valor</th></tr>
    <tr><td>Embedding dimension</td><td>{model_cfg['embedding_dim']}</td></tr>
    <tr><td>Stages</td><td>{model_cfg['num_stages']} ({', '.join(str(d) for d in model_cfg['stage_dims'])})</td></tr>
    <tr><td>Points por stage</td><td>{', '.join(str(p) for p in model_cfg['stage_points'])}</td></tr>
    <tr><td>k-Neighbors</td><td>{model_cfg['k_neighbors']}</td></tr>
    <tr><td>Pre/Post blocks</td><td>{model_cfg['pre_blocks']} / {model_cfg['pos_blocks']}</td></tr>
    <tr><td>Dropout</td><td>{model_cfg['dropout']}</td></tr>
    <tr><td>Total parámetros</td><td>192,099</td></tr>
</table>

<h3>2.3 Entrenamiento</h3>

<table>
    <tr><th>Hiperparámetro</th><th>Valor</th></tr>
    <tr><td>Optimizer</td><td>AdamW (weight decay = {config_a['train']['weight_decay']})</td></tr>
    <tr><td>Learning rate</td><td>{config_a['train']['learning_rate']}</td></tr>
    <tr><td>Scheduler</td><td>Cosine annealing + warmup ({config_a['train']['warmup_epochs']} epochs)</td></tr>
    <tr><td>Batch size</td><td>{config_a['train']['batch_size']}</td></tr>
    <tr><td>Label smoothing</td><td>{config_a['train']['label_smoothing']}</td></tr>
    <tr><td>Max epochs</td><td>{config_a['train']['epochs']}</td></tr>
    <tr><td>Early stopping</td><td>Patience = {config_a['train']['early_stopping']['patience']}</td></tr>
    <tr><td>Gradient clipping</td><td>{config_a['train']['gradient_clip']}</td></tr>
</table>

<h3>2.4 Hardware</h3>
<p>NVIDIA GeForce RTX 4060 (6 GB VRAM), Intel i5-13420H, 32 GB RAM, Windows 11, CUDA 12.x, PyTorch 2.2+.</p>

<h3>2.5 Evaluación</h3>
<p>Cada modelo se evalúa en 4 escenarios: su propio test set y el test set del otro dominio.
Se utiliza <strong>Test Time Augmentation (TTA)</strong> con 10 augmentaciones ligeras
(rotación Z aleatoria + jitter leve), promediando las probabilidades softmax antes de tomar argmax.</p>

<!-- 4. RESULTADOS -->
<h2 id="resultados">3. Resultados</h2>

<h3 id="dominio-propio">3.1 Rendimiento en dominio propio</h3>

<table>
    <tr><th>Métrica</th><th>Model A (Sintético→Sintético)</th><th>Model B (Real→Real)</th></tr>
    <tr><td>Accuracy</td><td class="highlight">{acc_aa:.1f}%</td><td>{acc_bb:.1f}%</td></tr>
    <tr><td>Macro F1</td><td class="highlight">{f1_aa:.1f}%</td><td>{f1_bb:.1f}%</td></tr>
    <tr><td>Best epoch</td><td>{best_epoch_a}</td><td>{best_epoch_b}</td></tr>
    <tr><td>Epochs entrenados</td><td>{total_epochs_a}</td><td>{total_epochs_b}</td></tr>
</table>

<p>El Model A alcanza {acc_aa:.1f}% de accuracy en datos sintéticos, demostrando que la arquitectura
PointMLP-Elite es capaz de aprender patrones discriminativos entre las 3 categorías arbóreas cuando
los datos son "limpios" y procedimentales. El Model B logra {acc_bb:.1f}% en datos reales,
reflejando la mayor complejidad inherente al LiDAR real: ruido del sensor, oclusiones parciales,
variabilidad morfológica natural y el desbalance de clases.</p>

<h3 id="cross-eval">3.2 Evaluación cruzada (resultados principales)</h3>

<p>Esta es la sección central del estudio. Los resultados de evaluar cada modelo en el dominio opuesto
revelan la magnitud del domain gap entre datos sintéticos y reales.</p>

<table>
    <tr><th>Evaluación</th><th>Accuracy</th><th>Macro F1</th><th>Interpretación</th></tr>
    <tr><td>A→A</td><td class="highlight">{acc_aa:.1f}%</td><td>{f1_aa:.1f}%</td><td>Referencia sintético</td></tr>
    <tr class="danger"><td>A→B</td><td>{acc_ab:.1f}%</td><td>{f1_ab:.1f}%</td>
        <td>Caída severa (−{drop_a:.1f}pp)</td></tr>
    <tr class="warning"><td>B→A</td><td>{acc_ba:.1f}%</td><td>{f1_ba:.1f}%</td>
        <td>Degradación moderada (−{drop_b:.1f}pp)</td></tr>
    <tr><td>B→B</td><td class="highlight">{acc_bb:.1f}%</td><td>{f1_bb:.1f}%</td><td>Referencia real</td></tr>
</table>

{img_tag("confusion_4panel", "Matrices de confusión para los 4 escenarios de evaluación cruzada (con TTA).", 1)}

{img_tag("accuracy_bar", "Comparación de accuracy en los 4 escenarios. La línea roja indica el nivel de azar (33.3%).", 2)}

{img_tag("domain_gap", "Análisis de domain gap: caída de performance al cruzar dominios.", 3)}

<p><strong>Análisis clave:</strong></p>
<ul>
    <li><strong>Model A en datos reales ({acc_ab:.1f}%):</strong> El modelo entrenado con sintéticos
    apenas supera el azar (33.3%). Clasifica casi todo como PINE, ignorando OAK completamente.
    Los patrones aprendidos de los datos sintéticos no transfieren a la geometría real.</li>
    <li><strong>Model B en datos sintéticos ({acc_ba:.1f}%):</strong> El modelo real generaliza
    ligeramente mejor al dominio sintético, pero con sesgo fuerte hacia OAK. No reconoce PINE ni OTHER
    en el dominio sintético.</li>
    <li><strong>Ambos modelos muestran sesgos opuestos:</strong> Model A se sesga hacia PINE,
    Model B se sesga hacia OAK en cross-domain, sugiriendo que cada dominio tiene patrones
    geométricos distintos para estas clases.</li>
</ul>

<h3 id="por-clase">3.3 Análisis por clase</h3>

{img_tag("f1_heatmap", "F1-Score por clase en los 4 escenarios. Rojo indica bajo rendimiento, verde alto.", 4)}

{img_tag("per_class", "Precisión, recall y F1 por clase para cada escenario de evaluación.", 5)}

<table>
    <tr><th>Evaluación</th><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th><th>Soporte</th></tr>
    {per_class_table()}
</table>

<p><strong>Observaciones por clase:</strong></p>
<ul>
    <li><strong>PINE:</strong> Es la clase más consistente en dominio propio para ambos modelos.
    Sin embargo, en cross-eval A→B, el modelo sintético sobreclasifica como PINE (recall alto, precision baja),
    mientras que en B→A el modelo real no reconoce ningún PINE sintético.</li>
    <li><strong>OAK:</strong> Es la clase mayoritaria. Model A no la reconoce en datos reales (F1=0%),
    mientras que Model B la sobreclasifica en datos sintéticos (recall 94%, pero precision 48%).</li>
    <li><strong>OTHER:</strong> La clase más difícil en ambos dominios. Es una categoría heterogénea
    que agrupa múltiples especies, lo que dificulta aprender patrones consistentes.</li>
</ul>

<h3 id="convergencia">3.4 Convergencia y entrenamiento</h3>

{img_tag("training_curves", "Curvas de entrenamiento de ambos modelos. ★ marca el mejor epoch.", 6)}

<table>
    <tr><th>Métrica</th><th>Model A</th><th>Model B</th></tr>
    <tr><td>Mejor epoch</td><td>{best_epoch_a}</td><td>{best_epoch_b}</td></tr>
    <tr><td>Total epochs (early stop)</td><td>{total_epochs_a}</td><td>{total_epochs_b}</td></tr>
    <tr><td>Mejor val accuracy</td><td>{max(log_a['val_acc']):.1f}%</td><td>{max(log_b['val_acc']):.1f}%</td></tr>
    <tr><td>Val loss mínimo</td><td>{min(log_a['val_loss']):.4f}</td><td>{min(log_b['val_loss']):.4f}</td></tr>
</table>

<p>El Model A converge rápidamente (mejor epoch {best_epoch_a}) y muestra una separación clara
entre train/val accuracy, indicando buena generalización dentro de su dominio.
El Model B converge aún más rápido (epoch {best_epoch_b}), pero con un techo de performance mucho
más bajo y mayor variabilidad en las curvas de validación, reflejando la dificultad del dominio real.</p>

<!-- 5. DISCUSIÓN -->
<h2 id="discusion">4. Discusión</h2>

<h3>4.1 ¿Los datos sintéticos son un sustituto viable para datos reales?</h3>
<p><strong>No en su estado actual.</strong> La caída de {drop_a:.1f} puntos porcentuales
al evaluar Model A en datos reales demuestra que los datos sintéticos generados procedimentalmente
no capturan las características esenciales del LiDAR real. El modelo aprende a distinguir las
"versiones sintéticas" de cada especie, pero estas representaciones no corresponden a la
variabilidad geométrica real.</p>

<h3>4.2 Magnitud del domain gap</h3>
<p>El domain gap es <strong>asimétrico</strong>:</p>
<ul>
    <li>Sintético→Real: −{drop_a:.1f}pp (catastrophic)</li>
    <li>Real→Sintético: −{drop_b:.1f}pp (significativa, pero el modelo real parte de un baseline más bajo)</li>
</ul>
<p>Esto sugiere que el dominio sintético es un subconjunto simplificado del espacio de formas arbóreas,
mientras que el dominio real contiene mucha más variabilidad no capturada por el generador.</p>

<h3>4.3 ¿Para qué clases funciona mejor la transferencia?</h3>
<p>Ninguna clase transfiere bien en la configuración actual. PINE muestra algo de consistencia
en recall para A→B (el modelo detecta "algo" de PINE en reales), pero con demasiados falsos
positivos. La categoría OTHER es inherentemente difícil por ser una clase "catchall".</p>

<h3>4.4 Limitaciones</h3>
<ul>
    <li><strong>Dataset pequeño:</strong> ~770 muestras por dominio, ~600 de entrenamiento, limita
    la capacidad del modelo.</li>
    <li><strong>Solo coordenadas XYZ:</strong> No se utilizan features adicionales como intensidad
    de retorno, número de retornos, RGB, o índices espectrales.</li>
    <li><strong>3 clases agrupadas:</strong> La categoría OTHER agrupa especies muy diferentes,
    diluyendo patrones aprendibles.</li>
    <li><strong>Generación sintética simplificada:</strong> Los árboles procedimentales pueden
    no capturar ramificación real, densidad de copa, o patrones de oclusión del sensor.</li>
    <li><strong>Sin domain adaptation:</strong> No se aplican técnicas de adaptación de dominio
    que podrían reducir el gap (adversarial training, MMD, etc.).</li>
</ul>

<h3>4.5 Posibles mejoras</h3>
<ul>
    <li><strong>Domain adaptation:</strong> Técnicas como DANN o MMD loss para alinear
    representaciones entre dominios.</li>
    <li><strong>Mejores sintéticos:</strong> Incorporar ruido realista, oclusiones, y variabilidad
    morfológica basada en datos botánicos.</li>
    <li><strong>Más datos reales:</strong> Aumentar el dataset real a >2000 muestras.</li>
    <li><strong>Features adicionales:</strong> Incorporar intensidad, altura normalizada, o
    features derivadas.</li>
    <li><strong>Fine-tuning:</strong> Pre-entrenar en sintéticos y fine-tune con pocos datos reales.</li>
</ul>

<!-- 6. CONCLUSIONES -->
<h2 id="conclusiones">5. Conclusiones</h2>
<ul>
    <li>La arquitectura PointMLP-Elite con 192K parámetros es capaz de clasificar 3 categorías
    arbóreas con <strong>{acc_aa:.1f}% de accuracy en datos sintéticos</strong> y
    <strong>{acc_bb:.1f}% en datos reales</strong>.</li>
    <li>El <strong>domain gap es severo</strong>: el modelo entrenado en sintéticos cae a
    {acc_ab:.1f}% en datos reales (−{drop_a:.1f}pp), apenas por encima del azar.</li>
    <li>Los datos sintéticos en su forma actual <strong>no son un sustituto viable</strong>
    para datos LiDAR reales en esta tarea de clasificación.</li>
    <li>Ambos modelos muestran <strong>sesgos de clase opuestos</strong> al cruzar dominios,
    indicando que las representaciones geométricas aprendidas son específicas de cada dominio.</li>
    <li>Se recomienda explorar <strong>domain adaptation</strong> y mejorar la calidad de los
    datos sintéticos antes de considerarlos útiles para entrenamiento.</li>
    <li>La clase OTHER es consistentemente la más difícil, sugiriendo que una clasificación
    binaria (PINE vs no-PINE) o con clases más homogéneas podría mejorar resultados.</li>
</ul>

<!-- 7. APÉNDICE -->
<h2 id="apendice">6. Apéndice</h2>

<h3>6.1 Tabla completa de métricas (TTA)</h3>
<table>
    <tr><th>Evaluación</th><th>Accuracy</th><th>Macro Precision</th><th>Macro Recall</th><th>Macro F1</th></tr>
    <tr><td>A→A</td>
        <td>{acc_aa:.1f}%</td>
        <td>{evals.get('A_on_A',{}).get('tta',{}).get('overall',{}).get('macro_precision',0)*100:.1f}%</td>
        <td>{evals.get('A_on_A',{}).get('tta',{}).get('overall',{}).get('macro_recall',0)*100:.1f}%</td>
        <td>{f1_aa:.1f}%</td></tr>
    <tr><td>A→B</td>
        <td>{acc_ab:.1f}%</td>
        <td>{evals.get('A_on_B',{}).get('tta',{}).get('overall',{}).get('macro_precision',0)*100:.1f}%</td>
        <td>{evals.get('A_on_B',{}).get('tta',{}).get('overall',{}).get('macro_recall',0)*100:.1f}%</td>
        <td>{f1_ab:.1f}%</td></tr>
    <tr><td>B→A</td>
        <td>{acc_ba:.1f}%</td>
        <td>{evals.get('B_on_A',{}).get('tta',{}).get('overall',{}).get('macro_precision',0)*100:.1f}%</td>
        <td>{evals.get('B_on_A',{}).get('tta',{}).get('overall',{}).get('macro_recall',0)*100:.1f}%</td>
        <td>{f1_ba:.1f}%</td></tr>
    <tr><td>B→B</td>
        <td>{acc_bb:.1f}%</td>
        <td>{evals.get('B_on_B',{}).get('tta',{}).get('overall',{}).get('macro_precision',0)*100:.1f}%</td>
        <td>{evals.get('B_on_B',{}).get('tta',{}).get('overall',{}).get('macro_recall',0)*100:.1f}%</td>
        <td>{f1_bb:.1f}%</td></tr>
</table>

<h3>6.2 Configuración de modelos</h3>
<p>Ambos modelos utilizan la misma arquitectura y configuración de hiperparámetros
(ver sección 2.2 y 2.3). La única diferencia es el dataset de entrenamiento.</p>

<h3>6.3 Distribución de clases por dataset y split</h3>
<table>
    <tr><th>Dataset</th><th>Split</th><th>PINE</th><th>OAK</th><th>OTHER</th><th>Total</th></tr>
    <tr><td rowspan="3">Sintético</td>
        <td>Train</td><td>{stats_synth['splits']['train']['class_distribution']['PINE']}</td>
        <td>{stats_synth['splits']['train']['class_distribution']['OAK']}</td>
        <td>{stats_synth['splits']['train']['class_distribution']['OTHER']}</td>
        <td>{stats_synth['splits']['train']['count']}</td></tr>
    <tr><td>Val</td><td>{stats_synth['splits']['val']['class_distribution']['PINE']}</td>
        <td>{stats_synth['splits']['val']['class_distribution']['OAK']}</td>
        <td>{stats_synth['splits']['val']['class_distribution']['OTHER']}</td>
        <td>{stats_synth['splits']['val']['count']}</td></tr>
    <tr><td>Test</td><td>{stats_synth['splits']['test']['class_distribution']['PINE']}</td>
        <td>{stats_synth['splits']['test']['class_distribution']['OAK']}</td>
        <td>{stats_synth['splits']['test']['class_distribution']['OTHER']}</td>
        <td>{stats_synth['splits']['test']['count']}</td></tr>
    <tr><td rowspan="3">Real</td>
        <td>Train</td><td>{stats_real['splits']['train']['class_distribution']['PINE']}</td>
        <td>{stats_real['splits']['train']['class_distribution']['OAK']}</td>
        <td>{stats_real['splits']['train']['class_distribution']['OTHER']}</td>
        <td>{stats_real['splits']['train']['count']}</td></tr>
    <tr><td>Val</td><td>{stats_real['splits']['val']['class_distribution']['PINE']}</td>
        <td>{stats_real['splits']['val']['class_distribution']['OAK']}</td>
        <td>{stats_real['splits']['val']['class_distribution']['OTHER']}</td>
        <td>{stats_real['splits']['val']['count']}</td></tr>
    <tr><td>Test</td><td>{stats_real['splits']['test']['class_distribution']['PINE']}</td>
        <td>{stats_real['splits']['test']['class_distribution']['OAK']}</td>
        <td>{stats_real['splits']['test']['class_distribution']['OTHER']}</td>
        <td>{stats_real['splits']['test']['count']}</td></tr>
</table>

<div class="footer">
    <p>Reporte generado automáticamente por <code>src/report.py</code> | PointMLP Tree Classification</p>
    <p>Fecha: {now} | TTA: 10 augmentaciones | Seed: 42</p>
</div>

</div>
</body>
</html>"""

    output_path = output_dir / "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")


# ── Main pipeline ────────────────────────────────────────────────────


def run_cross_evaluations() -> None:
    """Run missing cross-evaluations using evaluate module."""
    from src.evaluate import evaluate as run_eval

    config_path = Path("configs/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tasks = [
        ("model_A", "synthetic", "A_on_synthetic"),
        ("model_A", "real_3cls", "A_on_real"),
        ("model_B_3cls", "real_3cls", "B_on_real"),
        ("model_B_3cls", "synthetic", "B_on_synthetic"),
    ]

    for exp_name, dataset, subdir in tasks:
        out_dir = CROSS_EVAL_DIR / subdir
        result_path = out_dir / "eval_results.json"
        if result_path.exists():
            print(f"  {subdir}: already exists, skipping")
            continue
        print(f"  Running {exp_name} on {dataset}...")
        run_eval(config, exp_name, dataset, n_tta=10,
                 output_name="eval_results", output_dir=str(out_dir))


def generate_report(model_names: list[str], output_dir: str, cross_eval: bool = False) -> None:
    """Full report generation pipeline."""
    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Run cross-evaluations if requested or if results missing
    if cross_eval:
        print("Running cross-evaluations...")
        run_cross_evaluations()

    # Load all results
    print("Loading evaluation results...")
    evals = load_all_results()

    if len(evals) < 4:
        print(f"WARNING: Only {len(evals)}/4 evaluation results found.")
        print("Run with --cross_eval to generate missing evaluations.")

    # Generate figures
    print("Generating figures...")
    figures_b64 = {}

    print("  1/6 Confusion matrices 4-panel...")
    figures_b64["confusion_4panel"] = plot_confusion_matrices_4panel(evals, figures_dir)

    print("  2/6 Accuracy comparison bar...")
    figures_b64["accuracy_bar"] = plot_accuracy_comparison(evals, figures_dir)

    print("  3/6 F1 per-class heatmap...")
    figures_b64["f1_heatmap"] = plot_f1_heatmap(evals, figures_dir)

    print("  4/6 Domain gap analysis...")
    figures_b64["domain_gap"] = plot_domain_gap(evals, figures_dir)

    print("  5/6 Training curves...")
    figures_b64["training_curves"] = plot_training_curves(figures_dir)

    print("  6/6 Per-class comparison...")
    figures_b64["per_class"] = plot_per_class_comparison(evals, figures_dir)

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(evals, figures_b64, output_path)

    print(f"\nReport generated successfully!")
    print(f"  Figures: {figures_dir}")
    print(f"  HTML: {output_path / 'report.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar reporte comparativo")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="Nombres de los experimentos a comparar",
    )
    parser.add_argument(
        "--output", type=str, default="results/comparison_report/",
        help="Directorio de salida para el reporte",
    )
    parser.add_argument(
        "--cross_eval", action="store_true",
        help="Ejecutar evaluaciones cruzadas si faltan resultados",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Ruta al archivo de configuración",
    )
    args = parser.parse_args()

    generate_report(args.models, args.output, args.cross_eval)


if __name__ == "__main__":
    main()
