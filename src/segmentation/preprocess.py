"""
Preprocess NeonTreeEvaluation LiDAR tiles into training blocks for segmentation.

Pipeline:
  1. Load .laz tiles (full scenes)
  2. Sliding window -> extract spatial blocks
  3. Per-point labels: tree (1) vs non-tree (0) from ASPRS classification
  4. Height normalization (subtract local ground elevation per block)
  5. Subsample to fixed point count
  6. Save as .npy arrays (points, labels) per split
"""

import sys
import os
import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_tile(laz_path: str) -> dict:
    """Load a LAZ/LAS file and return xyz + classification arrays."""
    las = laspy.read(laz_path)
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    return {"xyz": xyz, "classification": classification}


def estimate_ground_z(xyz: np.ndarray, classification: np.ndarray,
                      ground_class: int = 2) -> np.ndarray:
    """Estimate ground height for each point in a small block.

    Simple and fast: use median Z of ground points.
    If no ground points, use percentile-5 of all Z (approximate ground).
    """
    ground_mask = classification == ground_class
    n = len(xyz)

    if ground_mask.sum() > 10:
        # Enough ground points: use their median
        ground_z = np.median(xyz[ground_mask, 2])
    elif ground_mask.sum() > 0:
        ground_z = np.median(xyz[ground_mask, 2])
    else:
        # No ground points: use 5th percentile as proxy
        ground_z = np.percentile(xyz[:, 2], 5)

    return np.full(n, ground_z, dtype=np.float64)


def resample_points(xyz: np.ndarray, labels: np.ndarray,
                    target_n: int) -> tuple:
    """Resample point cloud to exactly target_n points."""
    n = len(xyz)
    if n == target_n:
        return xyz, labels
    elif n > target_n:
        idx = np.random.choice(n, target_n, replace=False)
        return xyz[idx], labels[idx]
    else:
        pad_n = target_n - n
        idx = np.random.choice(n, pad_n, replace=True)
        xyz_pad = np.concatenate([xyz, xyz[idx]], axis=0)
        labels_pad = np.concatenate([labels, labels[idx]], axis=0)
        return xyz_pad, labels_pad


def extract_blocks(xyz: np.ndarray, classification: np.ndarray,
                   block_size: float, stride: float,
                   num_points: int, min_points: int,
                   tree_classes: list, ignore_classes: list,
                   normalize_height: bool) -> list:
    """Extract fixed-size blocks from a tile using sliding window.

    Uses spatial grid index for O(1) point lookup per block (fast for large tiles).
    Ground height is estimated per block.
    """
    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Pre-filter ignored classes
    valid_idx = np.arange(len(xyz))
    valid_mask = np.ones(len(xyz), dtype=bool)
    for cls in ignore_classes:
        valid_mask &= (classification != cls)
    valid_idx = valid_idx[valid_mask]
    v_xyz = xyz[valid_mask]
    v_cls = classification[valid_mask]
    v_x, v_y = v_xyz[:, 0], v_xyz[:, 1]

    # Build spatial grid index: assign each point to a grid cell
    # Cell size = stride (so each block spans block_size/stride cells in each dim)
    cell_size = stride
    cx = ((v_x - x_min) / cell_size).astype(np.int32)
    cy = ((v_y - y_min) / cell_size).astype(np.int32)
    nx = cx.max() + 1
    ny = cy.max() + 1

    # Create cell -> point index
    grid = {}
    cell_ids = cx * ny + cy
    order = np.argsort(cell_ids)
    sorted_cells = cell_ids[order]
    unique_cells, starts = np.unique(sorted_cells, return_index=True)
    ends = np.append(starts[1:], len(order))
    for i, cell_id in enumerate(unique_cells):
        grid[int(cell_id)] = order[starts[i]:ends[i]]

    # How many cells does each block span?
    cells_per_block = int(np.ceil(block_size / cell_size))

    blocks = []
    x_starts = np.arange(x_min, x_max - block_size / 2, stride)
    y_starts = np.arange(y_min, y_max - block_size / 2, stride)

    for x0 in x_starts:
        for y0 in y_starts:
            # Gather points from relevant grid cells
            cx0 = int((x0 - x_min) / cell_size)
            cy0 = int((y0 - y_min) / cell_size)

            candidate_idx = []
            for dcx in range(cells_per_block + 1):
                for dcy in range(cells_per_block + 1):
                    cell_id = (cx0 + dcx) * ny + (cy0 + dcy)
                    if cell_id in grid:
                        candidate_idx.append(grid[cell_id])

            if not candidate_idx:
                continue
            candidate_idx = np.concatenate(candidate_idx)

            # Exact filter within block bounds
            cx_pts = v_x[candidate_idx]
            cy_pts = v_y[candidate_idx]
            in_block = (
                (cx_pts >= x0) & (cx_pts < x0 + block_size) &
                (cy_pts >= y0) & (cy_pts < y0 + block_size)
            )
            block_idx = candidate_idx[in_block]
            n_pts = len(block_idx)

            if n_pts < min_points:
                continue

            block_xyz = v_xyz[block_idx].copy()
            block_cls = v_cls[block_idx]

            # Binary labels
            labels = np.zeros(n_pts, dtype=np.int64)
            for tc in tree_classes:
                labels[block_cls == tc] = 1

            # Height normalization
            if normalize_height:
                ground_z = estimate_ground_z(block_xyz, block_cls)
                block_xyz[:, 2] -= ground_z

            # Center XY at block center, normalize to [-1, 1]
            block_xyz[:, 0] -= (x0 + block_size / 2)
            block_xyz[:, 1] -= (y0 + block_size / 2)
            block_xyz[:, 0] /= (block_size / 2)
            block_xyz[:, 1] /= (block_size / 2)

            # Normalize Z to [-1, 1]
            z_range = block_xyz[:, 2].max() - block_xyz[:, 2].min()
            if z_range > 0.5:
                block_xyz[:, 2] = (block_xyz[:, 2] - block_xyz[:, 2].min()) / z_range * 2 - 1
            else:
                block_xyz[:, 2] = 0.0

            # Subsample to fixed point count
            block_xyz, labels = resample_points(block_xyz, labels, num_points)

            blocks.append({
                "points": block_xyz.astype(np.float32),
                "labels": labels,
            })

    return blocks


def process_training_tiles(cfg: dict) -> list:
    """Process the 16 large training tiles."""
    neon_dir = Path(cfg["data"]["neon_dir"])
    lidar_dir = neon_dir / "training" / "LiDAR"

    block_size = cfg["data"]["block_size"]
    stride = cfg["data"]["block_stride"]
    num_points = cfg["data"]["num_points"]
    min_points = cfg["data"]["min_points"]
    tree_classes = cfg["data"]["tree_classes"]
    ignore_classes = cfg["data"]["ignore_classes"]
    normalize_height = cfg["data"]["normalize_height"]

    all_blocks = []
    laz_files = sorted(lidar_dir.glob("*.laz"))
    print(f"Found {len(laz_files)} training tiles", flush=True)

    for i, laz_path in enumerate(laz_files):
        print(f"  [{i+1}/{len(laz_files)}] {laz_path.name}...", end=" ", flush=True)
        tile = load_tile(str(laz_path))
        n_pts = len(tile["xyz"])
        print(f"{n_pts:,} pts", end=" ", flush=True)

        blocks = extract_blocks(
            tile["xyz"], tile["classification"],
            block_size, stride, num_points, min_points,
            tree_classes, ignore_classes, normalize_height,
        )
        print(f"-> {len(blocks)} blocks", flush=True)

        for b in blocks:
            b["source"] = laz_path.stem
        all_blocks.extend(blocks)

        # Free memory for large tiles
        del tile

    return all_blocks


def process_eval_crops(cfg: dict) -> list:
    """Process the 2186 small evaluation crops."""
    neon_dir = Path(cfg["data"]["neon_dir"])
    lidar_dir = neon_dir / "evaluation" / "evaluation" / "LiDAR"

    num_points = cfg["data"]["num_points"]
    min_points = cfg["data"]["min_points"]
    tree_classes = cfg["data"]["tree_classes"]
    ignore_classes = cfg["data"]["ignore_classes"]
    normalize_height = cfg["data"]["normalize_height"]

    all_blocks = []
    laz_files = sorted(lidar_dir.glob("*.laz"))
    print(f"Found {len(laz_files)} evaluation crops", flush=True)

    for i, laz_path in enumerate(laz_files):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(laz_files)}] ...", flush=True)

        tile = load_tile(str(laz_path))
        xyz = tile["xyz"]
        classification = tile["classification"]
        n_pts = len(xyz)

        # Remove ignored
        keep_mask = np.ones(n_pts, dtype=bool)
        for cls in ignore_classes:
            keep_mask &= (classification != cls)
        xyz = xyz[keep_mask]
        classification = classification[keep_mask]
        n_pts = len(xyz)

        if n_pts < min_points:
            continue

        # Binary labels
        labels = np.zeros(n_pts, dtype=np.int64)
        for tc in tree_classes:
            labels[classification == tc] = 1

        # Height normalization
        if normalize_height:
            ground_z = estimate_ground_z(xyz, classification)
            xyz = xyz.copy()
            xyz[:, 2] -= ground_z

        # Center and normalize
        x_center = (xyz[:, 0].max() + xyz[:, 0].min()) / 2
        y_center = (xyz[:, 1].max() + xyz[:, 1].min()) / 2
        extent = max(xyz[:, 0].max() - xyz[:, 0].min(),
                     xyz[:, 1].max() - xyz[:, 1].min()) / 2
        if extent > 0:
            xyz[:, 0] = (xyz[:, 0] - x_center) / extent
            xyz[:, 1] = (xyz[:, 1] - y_center) / extent

        z_range = xyz[:, 2].max() - xyz[:, 2].min()
        if z_range > 0.5:
            xyz[:, 2] = (xyz[:, 2] - xyz[:, 2].min()) / z_range * 2 - 1
        else:
            xyz[:, 2] = 0.0

        xyz, labels = resample_points(xyz.astype(np.float32), labels, num_points)

        all_blocks.append({
            "points": xyz,
            "labels": labels,
            "source": laz_path.stem,
        })

    print(f"  Valid eval crops: {len(all_blocks)}", flush=True)
    return all_blocks


def split_blocks(blocks: list, train_ratio: float, val_ratio: float,
                 seed: int = 42) -> dict:
    """Split blocks into train/val/test sets."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(blocks))

    n = len(blocks)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": [blocks[i] for i in indices[:n_train]],
        "val": [blocks[i] for i in indices[n_train:n_train + n_val]],
        "test": [blocks[i] for i in indices[n_train + n_val:]],
    }


def save_split(blocks: list, out_dir: Path, split_name: str):
    """Save a split as .npy arrays."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    points = np.stack([b["points"] for b in blocks], axis=0)
    labels = np.stack([b["labels"] for b in blocks], axis=0)

    np.save(split_dir / "points.npy", points)
    np.save(split_dir / "labels.npy", labels)

    n_tree = (labels == 1).sum()
    n_total = labels.size
    tree_ratio = n_tree / n_total if n_total > 0 else 0

    print(f"  {split_name}: {len(blocks)} blocks, "
          f"{points.shape[1]} pts/block, "
          f"tree ratio: {tree_ratio:.3f}", flush=True)

    return {
        "num_blocks": len(blocks),
        "points_shape": list(points.shape),
        "labels_shape": list(labels.shape),
        "tree_ratio": float(tree_ratio),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess NeonTreeEvaluation for segmentation")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--skip-eval-crops", action="store_true",
                        help="Skip processing eval crops (faster for testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.get("experiment", {}).get("seed", 42))

    # 1. Process training tiles
    print("=" * 60, flush=True)
    print("Processing training tiles...", flush=True)
    print("=" * 60, flush=True)
    train_blocks = process_training_tiles(cfg)
    print(f"\nTotal training blocks: {len(train_blocks)}", flush=True)

    # 2. Process eval crops
    eval_blocks = []
    if not args.skip_eval_crops:
        print("\n" + "=" * 60, flush=True)
        print("Processing evaluation crops...", flush=True)
        print("=" * 60, flush=True)
        eval_blocks = process_eval_crops(cfg)
        print(f"\nTotal eval crop blocks: {len(eval_blocks)}", flush=True)

    # 3. Split
    print("\n" + "=" * 60, flush=True)
    print("Splitting and saving...", flush=True)
    print("=" * 60, flush=True)

    splits = split_blocks(
        train_blocks,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
    )

    if eval_blocks:
        splits["test"].extend(eval_blocks)
        print(f"  Added {len(eval_blocks)} eval crops to test set", flush=True)

    # 4. Save
    stats = {}
    for split_name, blocks in splits.items():
        if blocks:
            stats[split_name] = save_split(blocks, out_dir, split_name)

    # Class balance
    all_train_labels = np.concatenate([b["labels"] for b in splits["train"]])
    tree_ratio = (all_train_labels == 1).sum() / len(all_train_labels)
    pos_weight = (1 - tree_ratio) / tree_ratio if tree_ratio > 0 else 1.0
    stats["recommended_pos_weight"] = float(pos_weight)
    print(f"\nTree ratio in train: {tree_ratio:.3f}", flush=True)
    print(f"Recommended pos_weight: {pos_weight:.2f}", flush=True)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {out_dir}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
