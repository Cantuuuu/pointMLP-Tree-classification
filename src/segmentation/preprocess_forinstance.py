"""
Preprocess FORinstance TLS dataset into training blocks for segmentation.

Pipeline:
  1. Load .las plots (TLS, ground-based, dense point clouds)
  2. Exclude class-3 points (Out-points: unannotated trees outside plot boundary)
  3. Binary label: tree = (treeID > 0), non-tree = (treeID == 0 AND class != 3)
  4. Sliding window -> extract spatial blocks (XY plane, 5m x 5m)
  5. Height normalization per block using ground points (class 2)
     If no class-2 points in block, fall back to 5th-percentile Z.
  6. Subsample to fixed point count (4096)
  7. Save .npy arrays split by official dev/test from data_split_metadata.csv

NOTE: NIBIO2 is in the CSV but the folder does not exist on disk — skipped silently.
"""

import sys
import os
import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import yaml


OUT_POINTS_CLASS = 3   # trees outside annotated plot — excluded entirely
GROUND_CLASS = 2


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_plot(las_path: str) -> dict | None:
    """Load a FORinstance .las file.

    Returns dict with:
        xyz            : float64 (N, 3)
        classification : int32   (N,)
        tree_id        : int32   (N,)  — treeID field; 0 = non-tree
    Returns None if file does not exist.
    """
    p = Path(las_path)
    if not p.exists():
        return None

    las = laspy.read(str(p))
    xyz = np.stack([np.array(las.x), np.array(las.y), np.array(las.z)], axis=-1).astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    tree_id = np.array(las["treeID"], dtype=np.int32)
    return {"xyz": xyz, "classification": classification, "tree_id": tree_id}


def estimate_ground_z_block(xyz: np.ndarray, classification: np.ndarray) -> float:
    """Estimate ground elevation for a block.

    Uses median Z of class-2 (Terrain) points.
    Falls back to 5th-percentile Z if no terrain points present.
    """
    ground_mask = classification == GROUND_CLASS
    if ground_mask.sum() > 0:
        return float(np.median(xyz[ground_mask, 2]))
    return float(np.percentile(xyz[:, 2], 5))


def resample_points(xyz: np.ndarray, labels: np.ndarray,
                    tree_id: np.ndarray, target_n: int) -> tuple:
    """Resample a block to exactly target_n points."""
    n = len(xyz)
    if n == target_n:
        return xyz, labels, tree_id
    elif n > target_n:
        idx = np.random.choice(n, target_n, replace=False)
    else:
        pad_idx = np.random.choice(n, target_n - n, replace=True)
        idx = np.concatenate([np.arange(n), pad_idx])
    return xyz[idx], labels[idx], tree_id[idx]


def extract_blocks(xyz: np.ndarray, classification: np.ndarray,
                   tree_id: np.ndarray, block_size: float, stride: float,
                   num_points: int, min_points: int) -> list:
    """Extract fixed-size XY blocks from a TLS plot via sliding window.

    Excludes Out-points (class 3) before block extraction.
    Labels: tree = (treeID > 0) = 1, non-tree = 0.
    Heights are normalised per-block relative to local ground.
    XY are centred at block centre and normalised to [-1, 1].
    Z is normalised to [0, 1] within the block (TLS trees grow upward).
    """
    # --- 1. Remove Out-points (class 3) ---
    valid = classification != OUT_POINTS_CLASS
    xyz = xyz[valid]
    clf = classification[valid]
    tid = tree_id[valid]

    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # --- 2. Spatial grid index for fast block lookup ---
    cell_size = stride
    cx_idx = ((x - x_min) / cell_size).astype(np.int32)
    cy_idx = ((y - y_min) / cell_size).astype(np.int32)
    ny = int((y_max - y_min) / cell_size) + 2

    cell_ids = cx_idx * ny + cy_idx
    order = np.argsort(cell_ids)
    sorted_cells = cell_ids[order]
    unique_cells, starts = np.unique(sorted_cells, return_index=True)
    ends = np.append(starts[1:], len(order))

    grid = {}
    for i, cell_id in enumerate(unique_cells):
        grid[int(cell_id)] = order[starts[i]:ends[i]]

    cells_per_block = int(np.ceil(block_size / cell_size))

    blocks = []
    x_starts = np.arange(x_min, x_max - block_size / 2, stride)
    y_starts = np.arange(y_min, y_max - block_size / 2, stride)

    for x0 in x_starts:
        for y0 in y_starts:
            cx0 = int((x0 - x_min) / cell_size)
            cy0 = int((y0 - y_min) / cell_size)

            candidate_idx = []
            for dcx in range(cells_per_block + 1):
                for dcy in range(cells_per_block + 1):
                    cid = (cx0 + dcx) * ny + (cy0 + dcy)
                    if cid in grid:
                        candidate_idx.append(grid[cid])

            if not candidate_idx:
                continue
            candidate_idx = np.concatenate(candidate_idx)

            px = x[candidate_idx]
            py = y[candidate_idx]
            in_block = (
                (px >= x0) & (px < x0 + block_size) &
                (py >= y0) & (py < y0 + block_size)
            )
            block_idx = candidate_idx[in_block]

            if len(block_idx) < min_points:
                continue

            b_xyz = xyz[block_idx].copy()
            b_clf = clf[block_idx]
            b_tid = tid[block_idx]

            # --- 3. Binary labels ---
            labels = (b_tid > 0).astype(np.int64)

            # --- 4. Height normalisation (per-block ground) ---
            ground_z = estimate_ground_z_block(b_xyz, b_clf)
            b_xyz[:, 2] -= ground_z  # Z is now height above local ground

            # --- 5. XY centre + normalise to [-1, 1] ---
            b_xyz[:, 0] -= (x0 + block_size / 2)
            b_xyz[:, 1] -= (y0 + block_size / 2)
            b_xyz[:, 0] /= (block_size / 2)
            b_xyz[:, 1] /= (block_size / 2)

            # --- 6. Z normalise to [0, 1] ---
            z_range = b_xyz[:, 2].max() - b_xyz[:, 2].min()
            if z_range > 0.1:
                b_xyz[:, 2] = (b_xyz[:, 2] - b_xyz[:, 2].min()) / z_range
            else:
                b_xyz[:, 2] = 0.0

            # --- 7. Subsample ---
            b_xyz, labels, b_tid = resample_points(
                b_xyz.astype(np.float32), labels, b_tid, num_points
            )

            blocks.append({
                "points": b_xyz,
                "labels": labels,
                "tree_id": b_tid,
            })

    return blocks


def process_dataset(cfg: dict) -> dict:
    """Process all FORinstance plots and return blocks split by train/test."""
    dataset_dir = Path(cfg["data"]["forinstance_dir"])
    meta_path = dataset_dir / "data_split_metadata.csv"
    meta = pd.read_csv(meta_path)

    block_size  = cfg["data"]["block_size"]
    stride      = cfg["data"]["block_stride"]
    num_points  = cfg["data"]["num_points"]
    min_points  = cfg["data"]["min_points"]

    splits = {"train": [], "test": []}
    site_stats = {}

    for _, row in meta.iterrows():
        folder = row["folder"]
        rel_path = row["path"]
        official_split = row["split"]     # "dev" -> train, "test" -> test

        las_path = dataset_dir / rel_path
        if not las_path.exists():
            print(f"  [SKIP] {rel_path} not found", flush=True)
            continue

        plot_name = Path(rel_path).stem
        print(f"  [{official_split.upper()}] {rel_path} ...", end=" ", flush=True)

        plot = load_plot(str(las_path))
        if plot is None:
            print("FAILED", flush=True)
            continue

        blocks = extract_blocks(
            plot["xyz"], plot["classification"], plot["tree_id"],
            block_size, stride, num_points, min_points,
        )

        dest_split = "train" if official_split == "dev" else "test"
        for b in blocks:
            b["source"] = plot_name
            b["site"]   = folder
        splits[dest_split].extend(blocks)

        n_tree = sum((b["labels"] == 1).sum() for b in blocks)
        n_total = sum(len(b["labels"]) for b in blocks)
        tree_ratio = n_tree / n_total if n_total else 0.0

        key = f"{folder}/{plot_name}"
        site_stats[key] = {
            "split": dest_split,
            "blocks": len(blocks),
            "tree_ratio": float(tree_ratio),
        }
        print(f"{len(plot['xyz']):,} pts -> {len(blocks)} blocks "
              f"(tree {tree_ratio:.2f})", flush=True)
        del plot

    return splits, site_stats


def save_split(blocks: list, out_dir: Path, split_name: str) -> dict:
    """Save blocks as .npy arrays and return summary stats."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    points = np.stack([b["points"] for b in blocks], axis=0)
    labels = np.stack([b["labels"] for b in blocks], axis=0)

    # Save site/source metadata as JSON list for per-site evaluation
    meta = [{"source": b["source"], "site": b["site"]} for b in blocks]

    np.save(split_dir / "points.npy", points)
    np.save(split_dir / "labels.npy", labels)
    with open(split_dir / "block_meta.json", "w") as f:
        json.dump(meta, f)

    n_tree = int((labels == 1).sum())
    n_total = int(labels.size)
    tree_ratio = n_tree / n_total if n_total else 0.0

    print(f"  {split_name}: {len(blocks)} blocks | "
          f"pts/block={points.shape[1]} | tree_ratio={tree_ratio:.3f}", flush=True)

    return {
        "num_blocks": len(blocks),
        "points_shape": list(points.shape),
        "labels_shape": list(labels.shape),
        "tree_ratio": float(tree_ratio),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FORinstance TLS dataset for segmentation"
    )
    parser.add_argument("--config", default="configs/segmentation_forinstance.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.get("experiment", {}).get("seed", 42))

    print("=" * 60, flush=True)
    print("Processing FORinstance dataset...", flush=True)
    print("=" * 60, flush=True)

    splits, site_stats = process_dataset(cfg)

    print(f"\nTotal train blocks: {len(splits['train'])}", flush=True)
    print(f"Total test  blocks: {len(splits['test'])}", flush=True)

    # Val split: 15% of train blocks (random, stratified by site not needed at this scale)
    rng = np.random.RandomState(cfg.get("experiment", {}).get("seed", 42))
    train_blocks = splits["train"]
    rng.shuffle(train_blocks)
    n_val = max(1, int(len(train_blocks) * 0.15))
    val_blocks   = train_blocks[:n_val]
    train_blocks = train_blocks[n_val:]

    print("\n" + "=" * 60, flush=True)
    print("Saving splits...", flush=True)
    print("=" * 60, flush=True)

    all_stats = {}
    all_stats["train"] = save_split(train_blocks, out_dir, "train")
    all_stats["val"]   = save_split(val_blocks,   out_dir, "val")
    all_stats["test"]  = save_split(splits["test"], out_dir, "test")
    all_stats["per_plot"] = site_stats

    # Compute recommended pos_weight from train set
    train_labels = np.concatenate([b["labels"] for b in train_blocks])
    tree_ratio = float((train_labels == 1).sum() / len(train_labels))
    pos_weight = (1 - tree_ratio) / tree_ratio if tree_ratio > 0 else 1.0
    all_stats["recommended_pos_weight"] = pos_weight

    print(f"\nTree ratio (train): {tree_ratio:.3f}", flush=True)
    print(f"Recommended pos_weight: {pos_weight:.2f}", flush=True)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved to {out_dir}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
