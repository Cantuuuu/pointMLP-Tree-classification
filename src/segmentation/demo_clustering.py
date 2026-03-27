"""
Demo: Segmentation + Clustering on a real NeonTreeEvaluation tile.

Runs:
  1. PointNet++ segmentation (tree vs non-tree)
  2. HDBSCAN clustering (individual tree instances)
  3. Generates visualizations and appends section to HTML report
"""

import sys
import io
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.segmentation.pointnet2_model import build_model as build_seg_model
from src.segmentation.pipeline import segment_scene, cluster_trees


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def plot_segmentation_birdseye(xyz, predictions, save_path, title="Segmentation"):
    """Bird's-eye view: non-tree (gray) vs tree (green)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Original (color by height)
    ax = axes[0]
    sample = np.random.choice(len(xyz), min(200_000, len(xyz)), replace=False)
    sc = ax.scatter(xyz[sample, 0], xyz[sample, 1], c=xyz[sample, 2],
                    s=0.1, cmap="terrain", alpha=0.6)
    ax.set_title("Nube de puntos original (color = altura)", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Z (m)", shrink=0.7)

    # Segmentation
    ax = axes[1]
    colors = np.where(predictions == 1, "#2E7D32", "#9E9E9E")
    ax.scatter(xyz[sample, 0], xyz[sample, 1], c=colors[sample],
               s=0.1, alpha=0.6)
    ax.set_title("Segmentacion: arbol (verde) vs no-arbol (gris)", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2E7D32", label="Arbol"),
                       Patch(facecolor="#9E9E9E", label="No-arbol")]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_clustering_birdseye(xyz, tree_mask, instance_labels, save_path, title="Clustering"):
    """Bird's-eye view: each tree instance in a different color."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    tree_xyz = xyz[tree_mask]
    n_clusters = len(set(instance_labels)) - (1 if -1 in instance_labels else 0)

    # Left: segmentation result
    ax = axes[0]
    sample_all = np.random.choice(len(xyz), min(200_000, len(xyz)), replace=False)
    colors_all = np.where(np.isin(sample_all, np.where(tree_mask)[0]), "#2E7D32", "#D5D5D5")
    ax.scatter(xyz[sample_all, 0], xyz[sample_all, 1], c=colors_all,
               s=0.1, alpha=0.4)
    ax.set_title("Segmentacion binaria", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    # Right: clustering
    ax = axes[1]
    # Background: non-tree in light gray
    non_tree = xyz[~tree_mask]
    if len(non_tree) > 100_000:
        s_nt = np.random.choice(len(non_tree), 100_000, replace=False)
        non_tree = non_tree[s_nt]
    ax.scatter(non_tree[:, 0], non_tree[:, 1], c="#E8E8E8", s=0.05, alpha=0.3)

    # Noise points in dark gray
    noise_mask = instance_labels == -1
    if noise_mask.sum() > 0:
        noise_pts = tree_xyz[noise_mask]
        if len(noise_pts) > 50_000:
            s_n = np.random.choice(len(noise_pts), 50_000, replace=False)
            noise_pts = noise_pts[s_n]
        ax.scatter(noise_pts[:, 0], noise_pts[:, 1], c="#888888", s=0.1, alpha=0.3)

    # Each cluster in a unique color
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(min(n_clusters, 20))
    unique_labels = sorted(set(instance_labels) - {-1})
    for i, label in enumerate(unique_labels):
        mask = instance_labels == label
        pts = tree_xyz[mask]
        color = cmap(i % 20)
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=0.3, alpha=0.7)

    ax.set_title(f"CHM-Watershed: {n_clusters} arboles detectados", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cluster_detail(xyz, tree_mask, instance_labels, save_path, n_examples=6):
    """Show a few individual tree instances in 3D-like side views."""
    tree_xyz = xyz[tree_mask]
    unique_labels = sorted(set(instance_labels) - {-1})

    if len(unique_labels) == 0:
        print(f"  Skipped (no clusters): {save_path}")
        return

    if len(unique_labels) < n_examples:
        n_examples = len(unique_labels)

    # Pick trees with varying sizes
    sizes = [(instance_labels == l).sum() for l in unique_labels]
    sorted_idx = np.argsort(sizes)
    pick_idx = np.linspace(0, len(sorted_idx) - 1, n_examples, dtype=int)
    selected = [unique_labels[sorted_idx[i]] for i in pick_idx]

    fig, axes = plt.subplots(2, n_examples, figsize=(3.5 * n_examples, 8))
    if n_examples == 1:
        axes = axes.reshape(2, 1)

    cmap = matplotlib.colormaps.get_cmap("Set2").resampled(n_examples)

    for col, label in enumerate(selected):
        mask = instance_labels == label
        pts = tree_xyz[mask].copy()
        centroid = pts.mean(axis=0)
        pts -= centroid

        n_pts = len(pts)
        color = cmap(col)

        # Top view (XY)
        ax = axes[0, col]
        ax.scatter(pts[:, 0], pts[:, 1], c=[color], s=1, alpha=0.6)
        ax.set_title(f"Arbol #{label}\n({n_pts:,} pts)", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Side view (XZ)
        ax = axes[1, col]
        ax.scatter(pts[:, 0], pts[:, 2], c=[color], s=1, alpha=0.6)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Y (m)\nVista cenital", fontsize=10)
    axes[1, 0].set_ylabel("Z (m)\nVista lateral", fontsize=10)

    fig.suptitle("Instancias individuales de arboles (centradas)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cluster_stats(instance_labels, save_path):
    """Histogram of points per cluster and cluster count."""
    unique_labels = sorted(set(instance_labels) - {-1})
    sizes = [(instance_labels == l).sum() for l in unique_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of points per tree
    ax = axes[0]
    ax.hist(sizes, bins=30, color="#1565C0", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Puntos por arbol")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribucion de tamano ({len(sizes)} arboles)")
    ax.axvline(np.median(sizes), color="red", linestyle="--", label=f"Mediana: {int(np.median(sizes))}")
    ax.axvline(np.mean(sizes), color="orange", linestyle="--", label=f"Media: {int(np.mean(sizes))}")
    ax.legend()

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(sizes, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#1565C0", alpha=0.6),
                    medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("Puntos por arbol")
    ax.set_title("Distribucion de tamano (boxplot)")

    n_noise = (instance_labels == -1).sum()
    total_tree = len(instance_labels)
    noise_pct = 100 * n_noise / total_tree if total_tree > 0 else 0
    fig.suptitle(f"Estadisticas de clustering — Ruido: {n_noise:,} pts ({noise_pct:.1f}%)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def append_to_html_report(report_path, output_dir, stats):
    """Append clustering section to the existing HTML report."""
    report_path = Path(report_path)
    if not report_path.exists():
        print(f"  Warning: report not found at {report_path}")
        return

    content = report_path.read_text(encoding="utf-8")

    # Remove previous section 9 if it exists (idempotent re-runs)
    marker = "<h2>9. Demo: Segmentacion + Clustering en Escena Real</h2>"
    if marker in content:
        idx = content.index(marker)
        # Find the next <footer> or end-of-content after the section
        footer_idx = content.find("<footer>", idx)
        if footer_idx > idx:
            content = content[:idx] + content[footer_idx:]

    # Build the new section HTML
    new_section = f"""

<h2>9. Demo: Segmentacion + Clustering en Escena Real</h2>

<p>
    Para validar el pipeline completo, se ejecuto la segmentacion PointNet++ seguida del
    clustering HDBSCAN sobre un tile real del dataset NeonTreeEvaluation.
</p>

<h3>9.1 Escena utilizada</h3>
<table>
    <tr><th>Parametro</th><th>Valor</th></tr>
    <tr><td>Tile</td><td><code>{stats['tile_name']}</code></td></tr>
    <tr><td>Total de puntos</td><td>{stats['total_points']:,}</td></tr>
    <tr><td>Puntos de arbol (predichos)</td><td>{stats['tree_points']:,} ({stats['tree_ratio']:.1%})</td></tr>
    <tr><td>Tiempo segmentacion</td><td>{stats['seg_time']:.1f}s</td></tr>
    <tr><td>Arboles detectados (clusters)</td><td>{stats['n_clusters']}</td></tr>
    <tr><td>Puntos de ruido</td><td>{stats['n_noise']:,} ({stats['noise_pct']:.1f}%)</td></tr>
    <tr><td>Tiempo clustering</td><td>{stats['cluster_time']:.1f}s</td></tr>
    <tr><td>Mediana pts/arbol</td><td>{stats['median_pts_per_tree']:,}</td></tr>
    <tr><td>Media pts/arbol</td><td>{stats['mean_pts_per_tree']:,}</td></tr>
</table>

<h3>9.2 Segmentacion: arbol vs no-arbol</h3>
<figure>
    <img src="demo_segmentation.png" width="100%">
    <figcaption>
        Izquierda: nube de puntos original coloreada por altura.
        Derecha: resultado de la segmentacion binaria (verde = arbol, gris = no-arbol).
    </figcaption>
</figure>

<h3>9.3 Clustering: instancias individuales</h3>
<figure>
    <img src="demo_clustering.png" width="100%">
    <figcaption>
        Izquierda: segmentacion binaria. Derecha: cada color representa un arbol individual
        detectado por HDBSCAN. Se identificaron <strong>{stats['n_clusters']} arboles</strong>.
    </figcaption>
</figure>

<h3>9.4 Instancias individuales de arboles</h3>
<figure>
    <img src="demo_tree_instances.png" width="100%">
    <figcaption>
        Seleccion de arboles individuales extraidos. Fila superior: vista cenital (XY).
        Fila inferior: vista lateral (XZ). Cada arbol esta centrado en su centroide.
    </figcaption>
</figure>

<h3>9.5 Estadisticas de los clusters</h3>
<figure>
    <img src="demo_cluster_stats.png" width="100%">
    <figcaption>
        Distribucion del numero de puntos por arbol detectado.
    </figcaption>
</figure>

<div class="highlight">
    <strong>Resultado:</strong> El pipeline segmenta correctamente la escena y el clustering HDBSCAN
    separa {stats['n_clusters']} arboles individuales. Cada instancia puede alimentarse al
    clasificador PointMLP para identificar la especie. El pipeline completo
    (segmentacion + clustering) tomo {stats['seg_time'] + stats['cluster_time']:.1f}s en esta escena
    de {stats['total_points']:,} puntos.
</div>
"""

    # Insert before </footer> or before closing tags
    insertion_point = content.rfind("<footer>")
    if insertion_point == -1:
        insertion_point = content.rfind("</div>")

    new_content = content[:insertion_point] + new_section + "\n\n" + content[insertion_point:]
    report_path.write_text(new_content, encoding="utf-8")
    print(f"  Updated report: {report_path}")


def main():
    cfg_path = ROOT / "configs" / "segmentation.yaml"
    cfg = load_config(str(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Pick tile ---
    tile_path = Path("C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_MLBS_3_541000_4140000_image_crop.laz")
    tile_name = tile_path.stem
    print(f"\n{'='*60}")
    print(f"  DEMO: Segmentation + Clustering")
    print(f"  Tile: {tile_name}")
    print(f"{'='*60}")

    # --- Load scene ---
    print(f"\n1. Loading scene...")
    las = laspy.read(str(tile_path))
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    print(f"   Points: {len(xyz):,}")

    # --- Segmentation ---
    print(f"\n2. Running PointNet++ segmentation...")
    t0 = time.time()
    seg_model = build_seg_model(cfg).to(device)
    ckpt_path = ROOT / "results" / "segmentation" / "best_model.pth"
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    seg_model.load_state_dict(checkpoint["model_state_dict"])

    predictions = segment_scene(xyz, classification, seg_model, device, cfg)
    tree_mask = predictions == 1
    seg_time = time.time() - t0

    n_tree = int(tree_mask.sum())
    n_total = len(xyz)
    print(f"   Tree points: {n_tree:,} / {n_total:,} ({n_tree/n_total:.1%})")
    print(f"   Time: {seg_time:.1f}s")

    # --- Clustering ---
    print(f"\n3. Running HDBSCAN clustering...")
    t0 = time.time()
    instance_labels = cluster_trees(xyz, tree_mask, cfg, classification)
    cluster_time = time.time() - t0

    n_clusters = len(set(instance_labels)) - (1 if -1 in instance_labels else 0)
    n_noise = int((instance_labels == -1).sum())
    n_tree_pts = len(instance_labels)
    noise_pct = 100 * n_noise / n_tree_pts if n_tree_pts > 0 else 0

    unique_labels = sorted(set(instance_labels) - {-1})
    sizes = [(instance_labels == l).sum() for l in unique_labels]

    print(f"   Clusters: {n_clusters}")
    print(f"   Noise: {n_noise:,} ({noise_pct:.1f}%)")
    print(f"   Time: {cluster_time:.1f}s")

    # --- Generate visualizations ---
    output_dir = ROOT / "results" / "segmentation" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n4. Generating visualizations...")
    plot_segmentation_birdseye(
        xyz, predictions,
        output_dir / "demo_segmentation.png",
        title=f"Segmentacion — {tile_name}"
    )
    plot_clustering_birdseye(
        xyz, tree_mask, instance_labels,
        output_dir / "demo_clustering.png",
        title=f"Clustering — {tile_name}"
    )
    plot_cluster_detail(
        xyz, tree_mask, instance_labels,
        output_dir / "demo_tree_instances.png"
    )
    plot_cluster_stats(
        instance_labels,
        output_dir / "demo_cluster_stats.png"
    )

    # --- Update HTML report ---
    print(f"\n5. Updating HTML report...")
    stats = {
        "tile_name": tile_name,
        "total_points": n_total,
        "tree_points": n_tree,
        "tree_ratio": n_tree / n_total,
        "seg_time": seg_time,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "cluster_time": cluster_time,
        "median_pts_per_tree": int(np.median(sizes)) if sizes else 0,
        "mean_pts_per_tree": int(np.mean(sizes)) if sizes else 0,
    }
    append_to_html_report(
        output_dir / "segmentation_report.html",
        output_dir,
        stats
    )

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Total time: {seg_time + cluster_time:.1f}s")
    print(f"  Trees detected: {n_clusters}")
    print(f"  Report updated: {output_dir / 'segmentation_report.html'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
