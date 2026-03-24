"""Filter processed dataset to keep only top-N classes with most samples.

Reads existing .npy files, filters to selected classes, remaps labels 0..N-1,
and saves to a new processed directory.

Usage:
    python scripts/filter_classes.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────────────

# Keep only these 5 classes (species with >= 50 samples)
KEEP_SPECIES = ["PIPA2", "QURU", "ACRU", "QUAL", "QUCO2"]

SRC_DIR = Path("data/processed/real")
DST_DIR = Path("data/processed/real_5cls")

SPLITS = ["train", "val", "test"]


def main() -> None:
    # Load original species map
    with open(SRC_DIR / "species_map.json") as f:
        old_map: dict = json.load(f)

    # Build old_index -> species_code lookup
    idx_to_species = {v["index"]: k for k, v in old_map.items()}

    # Build new species map (sorted by new index)
    new_map = {}
    for new_idx, species in enumerate(KEEP_SPECIES):
        new_map[species] = {
            "index": new_idx,
            "count": 0,  # will be updated
            "scientific_name": old_map[species]["scientific_name"],
        }

    old_to_new = {}
    for species in KEEP_SPECIES:
        old_to_new[old_map[species]["index"]] = new_map[species]["index"]

    keep_old_indices = set(old_to_new.keys())

    print(f"Filtering {len(old_map)} classes -> {len(KEEP_SPECIES)} classes")
    print(f"Keeping: {KEEP_SPECIES}")
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    print()

    # Process each split
    total_stats = {"total_processed": 0, "n_points": 0, "n_classes": len(KEEP_SPECIES), "splits": {}}

    for split in SPLITS:
        src_split = SRC_DIR / split
        dst_split = DST_DIR / split
        dst_split.mkdir(parents=True, exist_ok=True)

        points = np.load(src_split / "points.npy")   # (N, 1024, 3)
        labels = np.load(src_split / "labels.npy")    # (N,)
        tree_ids = np.load(src_split / "tree_ids.npy") if (src_split / "tree_ids.npy").exists() else None

        # Filter: keep only samples whose label is in keep set
        mask = np.isin(labels, list(keep_old_indices))
        points_f = points[mask]
        labels_f = labels[mask]
        tree_ids_f = tree_ids[mask] if tree_ids is not None else None

        # Remap labels
        new_labels = np.array([old_to_new[int(l)] for l in labels_f], dtype=np.int64)

        # Save
        np.save(dst_split / "points.npy", points_f)
        np.save(dst_split / "labels.npy", new_labels)
        if tree_ids_f is not None:
            np.save(dst_split / "tree_ids.npy", tree_ids_f)

        # Count per class
        class_dist = {}
        for species in KEEP_SPECIES:
            idx = new_map[species]["index"]
            count = int((new_labels == idx).sum())
            class_dist[species] = count
            new_map[species]["count"] += count

        total_stats["n_points"] = points_f.shape[1]
        total_stats["total_processed"] += len(new_labels)
        total_stats["splits"][split] = {
            "count": len(new_labels),
            "class_distribution": class_dist,
        }

        print(f"  {split}: {len(labels)} -> {len(new_labels)} samples")
        for sp, cnt in class_dist.items():
            print(f"    {sp}: {cnt}")

    # Save new species map and stats
    with open(DST_DIR / "species_map.json", "w") as f:
        json.dump(new_map, f, indent=2)

    with open(DST_DIR / "stats.json", "w") as f:
        json.dump(total_stats, f, indent=2)

    print(f"\nTotal: {total_stats['total_processed']} samples across {len(KEEP_SPECIES)} classes")
    print(f"Saved to {DST_DIR}")


if __name__ == "__main__":
    main()
