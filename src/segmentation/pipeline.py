"""
Full pipeline: Segment -> Cluster -> Classify -> Report

Takes a raw LiDAR scene (.laz), runs:
  1. PointNet++ segmentation (tree vs non-tree per point)
  2. HDBSCAN clustering (group tree points into individual trees)
  3. PointMLP classification (species per tree instance)
  4. Generate report
"""

import sys

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import laspy

from src.segmentation.pointnet2_model import build_model as build_seg_model
from src.segmentation.preprocess import resample_points


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step 1: Segment a full scene
# ---------------------------------------------------------------------------

def segment_scene(xyz: np.ndarray, classification: np.ndarray,
                  model: torch.nn.Module, device: torch.device,
                  cfg: dict) -> np.ndarray:
    """Run segmentation on a full LiDAR scene using sliding windows.

    Returns:
        predictions: (N,) — per-point binary prediction (0=non-tree, 1=tree)
    """
    block_size = cfg["data"]["block_size"]
    stride = cfg["data"]["block_stride"]
    num_points = cfg["data"]["num_points"]
    tree_classes = cfg["data"]["tree_classes"]
    ignore_classes = cfg["data"]["ignore_classes"]
    normalize_height = cfg["data"]["normalize_height"]

    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    N = len(xyz)

    # Accumulate predictions (vote counting)
    vote_tree = np.zeros(N, dtype=np.float64)
    vote_count = np.zeros(N, dtype=np.int32)

    # Ignore mask
    ignore_mask = np.zeros(N, dtype=bool)
    for cls in ignore_classes:
        ignore_mask |= (classification == cls)

    x_starts = np.arange(x_min, x_max - block_size / 2, stride)
    y_starts = np.arange(y_min, y_max - block_size / 2, stride)

    model.eval()
    n_blocks = 0

    with torch.no_grad():
        for x0 in x_starts:
            for y0 in y_starts:
                # Select points in block
                mask = (
                    (x >= x0) & (x < x0 + block_size) &
                    (y >= y0) & (y < y0 + block_size) &
                    (~ignore_mask)
                )
                indices = np.where(mask)[0]
                n_pts = len(indices)

                if n_pts < 64:
                    continue

                block_xyz = xyz[indices].copy()

                # Height normalization (estimate ground per block)
                if normalize_height:
                    ground_mask = classification[indices] == 2  # ASPRS ground
                    if ground_mask.sum() > 0:
                        ground_z = np.median(block_xyz[ground_mask, 2])
                    else:
                        ground_z = np.percentile(block_xyz[:, 2], 5)
                    block_xyz[:, 2] -= ground_z

                # Center and normalize
                block_xyz[:, 0] -= (x0 + block_size / 2)
                block_xyz[:, 1] -= (y0 + block_size / 2)
                block_xyz[:, 0] /= (block_size / 2)
                block_xyz[:, 1] /= (block_size / 2)

                z_range = block_xyz[:, 2].max() - block_xyz[:, 2].min()
                if z_range > 0:
                    block_xyz[:, 2] = (block_xyz[:, 2] - block_xyz[:, 2].min()) / z_range * 2 - 1
                else:
                    block_xyz[:, 2] = 0.0

                # Subsample for model input
                if n_pts > num_points:
                    sample_idx = np.random.choice(n_pts, num_points, replace=False)
                else:
                    sample_idx = np.arange(n_pts)
                    if n_pts < num_points:
                        pad = np.random.choice(n_pts, num_points - n_pts, replace=True)
                        sample_idx = np.concatenate([sample_idx, pad])

                block_input = block_xyz[sample_idx[:num_points]].astype(np.float32)
                tensor_input = torch.from_numpy(block_input).unsqueeze(0).to(device)

                # Forward pass
                logits = model(tensor_input)  # (1, num_points, 2)
                probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

                # Map predictions back to original indices
                # Only for the actual (non-padded) points
                actual_n = min(n_pts, num_points)
                original_indices = indices[sample_idx[:actual_n]]
                vote_tree[original_indices] += probs[:actual_n]
                vote_count[original_indices] += 1
                n_blocks += 1

    print(f"  Processed {n_blocks} blocks")

    # Average votes
    valid = vote_count > 0
    predictions = np.zeros(N, dtype=np.int32)
    predictions[valid] = (vote_tree[valid] / vote_count[valid] > 0.5).astype(np.int32)

    # Points that were never in any block → use ASPRS classification as fallback
    never_seen = ~valid & ~ignore_mask
    for tc in tree_classes:
        predictions[never_seen & (classification == tc)] = 1

    return predictions


# ---------------------------------------------------------------------------
# Step 2: Cluster tree points into instances
# ---------------------------------------------------------------------------

def cluster_trees(xyz: np.ndarray, tree_mask: np.ndarray,
                  cfg: dict, classification: np.ndarray = None) -> np.ndarray:
    """Cluster predicted tree points into individual tree instances.

    Uses CHM-watershed: the standard Individual Tree Detection (ITD)
    approach in forestry remote sensing.  Works across any point density
    because it operates on rasterised height, not point-density valleys.

    Steps:
      1. Rasterise tree points to a 2D Canopy Height Model (CHM).
      2. Smooth CHM with a Gaussian filter to suppress noise.
      3. Detect local maxima → tree-top candidates.
      4. Watershed segmentation from those markers → crown regions.
      5. Assign every 3D tree point to its watershed region.

    Args:
        xyz: (N, 3) full scene point cloud
        tree_mask: (N,) boolean mask for tree points
        cfg: config dict
        classification: (N,) ASPRS class codes (used for ground estimation)

    Returns:
        instance_labels: (N_tree_points,) — cluster IDs (-1 = noise)
    """
    from scipy import ndimage

    tree_xyz = xyz[tree_mask]
    cluster_cfg = cfg["clustering"]
    n_tree = len(tree_xyz)
    print(f"  Clustering {n_tree:,} tree points...")

    # --- Config ---------------------------------------------------------
    chm_resolution = cluster_cfg.get("chm_resolution", 0.5)   # m/pixel
    smooth_sigma = cluster_cfg.get("smooth_sigma", 1.5)        # Gaussian σ in pixels
    min_tree_height = cluster_cfg.get("min_tree_height", 3.0)  # metres
    min_crown_pixels = cluster_cfg.get("min_crown_pixels", 4)  # min pixels per crown
    local_max_window = cluster_cfg.get("local_max_window", 5)  # window for maxima (pixels, odd)

    # --- 1. Build CHM (Canopy Height Model) ----------------------------
    # Normalise Z to height-above-ground using ALL scene points
    # (not just tree points) so ground estimation is accurate.
    all_z = xyz[:, 2]
    if classification is not None:
        ground_mask_scene = (classification == 2)
        if ground_mask_scene.sum() > 0:
            ground_z = float(np.median(all_z[ground_mask_scene]))
        else:
            ground_z = float(np.percentile(all_z, 5))
    else:
        ground_z = float(np.percentile(all_z, 5))

    x, y = tree_xyz[:, 0], tree_xyz[:, 1]
    z = tree_xyz[:, 2] - ground_z  # height above ground
    x_min, y_min = x.min(), y.min()

    # Grid coordinates
    col = ((x - x_min) / chm_resolution).astype(np.int32)
    row = ((y - y_min) / chm_resolution).astype(np.int32)
    n_cols = col.max() + 1
    n_rows = row.max() + 1

    # Max height per pixel = CHM (init with -inf, then replace with 0)
    chm = np.full((n_rows, n_cols), -np.inf, dtype=np.float64)
    flat_idx = row * n_cols + col
    np.maximum.at(chm.ravel(), flat_idx, z)
    chm[chm == -np.inf] = 0.0

    print(f"  CHM: {n_rows}x{n_cols} pixels ({chm_resolution} m/px), "
          f"height range [{chm[chm > 0].min():.1f}, {chm.max():.1f}] m "
          f"(ground_z={ground_z:.1f})")
    chm_filled = chm

    # --- 2. Smooth CHM --------------------------------------------------
    chm_smooth = ndimage.gaussian_filter(chm_filled, sigma=smooth_sigma)

    # --- 3. Detect local maxima (tree tops) -----------------------------
    # Dilate = local max in window; where smooth == dilated → local max
    footprint = np.ones((local_max_window, local_max_window))
    local_max = ndimage.maximum_filter(chm_smooth, footprint=footprint)
    is_max = (chm_smooth == local_max) & (chm_smooth >= min_tree_height)

    # Label connected components of maxima (each becomes a seed)
    markers, n_seeds = ndimage.label(is_max)
    print(f"  Tree-top seeds: {n_seeds} "
          f"(window={local_max_window}px, min_height={min_tree_height}m)")

    if n_seeds == 0:
        print("  WARNING: no tree tops found — returning all as noise")
        return np.full(n_tree, -1, dtype=np.int32)

    # --- 4. Watershed segmentation --------------------------------------
    # Invert CHM so valleys become basins (watershed fills from markers)
    chm_inv = chm_smooth.max() - chm_smooth
    # Mask: only segment where there is canopy
    canopy_mask = chm_filled >= min_tree_height
    # watershed_ift needs uint8 or uint16 input — scale to [0, 65535]
    chm_inv_norm = chm_inv / (chm_inv.max() + 1e-8) * 65534
    watershed_labels = ndimage.watershed_ift(
        chm_inv_norm.astype(np.uint16),
        markers,
    )
    # Zero out regions outside canopy mask
    watershed_labels[~canopy_mask] = 0

    # Filter tiny crowns
    for lbl in range(1, n_seeds + 1):
        if (watershed_labels == lbl).sum() < min_crown_pixels:
            watershed_labels[watershed_labels == lbl] = 0

    unique_labels = set(watershed_labels.ravel()) - {0}
    n_crowns = len(unique_labels)
    print(f"  Watershed crowns: {n_crowns} (after filtering < {min_crown_pixels}px)")

    # --- 5. Assign 3D points to watershed regions -----------------------
    point_labels = watershed_labels[row, col]

    # Remap to contiguous IDs (0 → -1 = noise, rest → 0,1,2...)
    labels = np.full(n_tree, -1, dtype=np.int32)
    remap = {}
    next_id = 0
    for lbl in sorted(unique_labels):
        remap[lbl] = next_id
        next_id += 1
    for i in range(n_tree):
        wl = point_labels[i]
        if wl in remap:
            labels[i] = remap[wl]

    n_noise = int((labels == -1).sum())
    print(f"  Found {n_crowns} tree instances, {n_noise:,} noise points")

    return labels


# ---------------------------------------------------------------------------
# Step 3: Extract individual trees and classify with PointMLP
# ---------------------------------------------------------------------------

def extract_tree_instances(xyz: np.ndarray, tree_mask: np.ndarray,
                           instance_labels: np.ndarray,
                           num_points: int = 1024) -> list:
    """Extract individual tree point clouds from clustered results."""
    tree_xyz = xyz[tree_mask]
    unique_labels = sorted(set(instance_labels) - {-1})

    trees = []
    for label in unique_labels:
        mask = instance_labels == label
        tree_pts = tree_xyz[mask].copy()

        if len(tree_pts) < 20:
            continue

        # Center at XY centroid, keep Z relative
        centroid = tree_pts.mean(axis=0)
        tree_pts -= centroid

        # Resample to fixed count
        tree_pts, _ = resample_points(
            tree_pts.astype(np.float32),
            np.zeros(len(tree_pts), dtype=np.int64),
            num_points,
        )

        trees.append({
            "points": tree_pts,
            "centroid": centroid,
            "n_original_points": int(mask.sum()),
        })

    return trees


def classify_trees(trees: list, cls_model: torch.nn.Module,
                   device: torch.device, species_map: dict) -> list:
    """Classify each tree instance using PointMLP."""
    if not trees:
        return []

    cls_model.eval()
    results = []

    with torch.no_grad():
        for tree in trees:
            pts = torch.from_numpy(tree["points"]).unsqueeze(0).to(device)
            logits = cls_model(pts)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class])

            results.append({
                "species": species_map.get(str(pred_class), f"class_{pred_class}"),
                "confidence": confidence,
                "centroid": tree["centroid"].tolist(),
                "n_points": tree["n_original_points"],
                "probabilities": {species_map.get(str(i), f"class_{i}"): float(p)
                                  for i, p in enumerate(probs)},
            })

    return results


# ---------------------------------------------------------------------------
# Step 4: Generate report
# ---------------------------------------------------------------------------

def generate_report(results: list, scene_name: str, seg_stats: dict,
                    output_dir: Path):
    """Generate a summary report."""
    report = {
        "scene": scene_name,
        "segmentation": seg_stats,
        "total_trees_detected": len(results),
        "species_summary": {},
        "trees": results,
    }

    # Species summary
    from collections import Counter
    species_counts = Counter(r["species"] for r in results)
    for species, count in species_counts.most_common():
        avg_conf = np.mean([r["confidence"] for r in results
                           if r["species"] == species])
        report["species_summary"][species] = {
            "count": count,
            "percentage": f"{100 * count / len(results):.1f}%",
            "avg_confidence": f"{avg_conf:.3f}",
        }

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"report_{scene_name}.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  TREE DETECTION & CLASSIFICATION REPORT")
    print(f"  Scene: {scene_name}")
    print(f"{'='*60}")
    print(f"\n  Segmentation:")
    print(f"    Total points:      {seg_stats['total_points']:,}")
    print(f"    Tree points:       {seg_stats['tree_points']:,} ({seg_stats['tree_ratio']:.1%})")
    print(f"    Non-tree points:   {seg_stats['non_tree_points']:,}")
    print(f"\n  Instance Detection:")
    print(f"    Trees detected:    {len(results)}")
    print(f"\n  Species Classification:")
    for species, info in report["species_summary"].items():
        print(f"    {species:<12} {info['count']:>4} trees "
              f"({info['percentage']:>5}) "
              f"avg_conf={info['avg_confidence']}")

    print(f"\n  Report saved to {output_dir / f'report_{scene_name}.json'}")
    print(f"{'='*60}")

    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full tree detection pipeline")
    parser.add_argument("input", help="Path to input .laz file")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--seg-checkpoint", default=None)
    parser.add_argument("--cls-checkpoint", default=None)
    parser.add_argument("--cls-config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/pipeline")
    args = parser.parse_args()

    seg_cfg = load_config(args.seg_config)
    cls_cfg = load_config(args.cls_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seg_log_dir = Path(seg_cfg["experiment"]["log_dir"])
    seg_ckpt = args.seg_checkpoint or str(seg_log_dir / "best_model.pth")
    cls_ckpt = args.cls_checkpoint or seg_cfg["classification"]["model_path"]

    scene_name = Path(args.input).stem

    # --- Load scene ---
    print(f"\n1. Loading scene: {args.input}")
    las = laspy.read(args.input)
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    print(f"   Points: {len(xyz):,}")

    # --- Segmentation ---
    print(f"\n2. Running PointNet++ segmentation...")
    t0 = time.time()
    seg_model = build_seg_model(seg_cfg).to(device)
    seg_checkpoint = torch.load(seg_ckpt, map_location=device, weights_only=False)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])

    predictions = segment_scene(xyz, classification, seg_model, device, seg_cfg)
    tree_mask = predictions == 1
    seg_time = time.time() - t0

    seg_stats = {
        "total_points": int(len(xyz)),
        "tree_points": int(tree_mask.sum()),
        "non_tree_points": int((~tree_mask).sum()),
        "tree_ratio": float(tree_mask.sum() / len(xyz)),
        "time_seconds": round(seg_time, 1),
    }
    print(f"   Tree points: {seg_stats['tree_points']:,} / {seg_stats['total_points']:,}")
    print(f"   Time: {seg_time:.1f}s")

    # --- Clustering ---
    print(f"\n3. Clustering tree instances...")
    t0 = time.time()
    instance_labels = cluster_trees(xyz, tree_mask, seg_cfg, classification)
    cluster_time = time.time() - t0
    print(f"   Time: {cluster_time:.1f}s")

    # --- Extract individual trees ---
    print(f"\n4. Extracting individual trees...")
    cls_num_points = seg_cfg["classification"]["num_points"]
    trees = extract_tree_instances(xyz, tree_mask, instance_labels, cls_num_points)
    print(f"   Valid tree instances: {len(trees)}")

    # --- Classification ---
    print(f"\n5. Classifying tree species with PointMLP...")
    t0 = time.time()

    # Load PointMLP model
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.model import PointMLPElite

    cls_model = PointMLPElite(cls_cfg).to(device)
    cls_checkpoint = torch.load(cls_ckpt, map_location=device, weights_only=False)
    cls_model.load_state_dict(cls_checkpoint["model_state_dict"])

    # Load species map
    processed_dir = Path(cls_cfg["data"]["processed_dir"])
    species_map_path = processed_dir / "real_3cls" / "species_map.json"
    if not species_map_path.exists():
        species_map_path = processed_dir / "real" / "species_map.json"
    with open(species_map_path) as f:
        species_map = json.load(f)

    results = classify_trees(trees, cls_model, device, species_map)
    cls_time = time.time() - t0
    print(f"   Time: {cls_time:.1f}s")

    # --- Report ---
    print(f"\n6. Generating report...")
    output_dir = Path(args.output_dir)
    generate_report(results, scene_name, seg_stats, output_dir)

    print(f"\nTotal pipeline time: {seg_time + cluster_time + cls_time:.1f}s")


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
