"""Filter processed dataset to 3 macro-classes.

Groups:
  0 - PINE:  PIPA2 (190 samples)
  1 - OAK:   QUAL + QUCO2 + QULA2 + QURU (353 samples)
  2 - OTHER:  ACRU + AMLA + NYSY (224 samples)

Usage:
    python scripts/filter_3cls.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC_DIR = Path("data/processed/real")
DST_DIR = Path("data/processed/real_3cls")

SPLITS = ["train", "val", "test"]

# Mapping: old species code -> new class index
# 0=PINE, 1=OAK (all Quercus), 2=OTHER (broadleaf non-oak)
GROUPS = {
    "PIPA2": 0,   # Pine
    "QUAL":  1,   # Oak
    "QUCO2": 1,   # Oak
    "QULA2": 1,   # Oak
    "QURU":  1,   # Oak
    "ACRU":  2,   # Other broadleaf
    "AMLA":  2,   # Other broadleaf
    "NYSY":  2,   # Other broadleaf
}

CLASS_NAMES = ["PINE", "OAK", "OTHER"]


def main() -> None:
    with open(SRC_DIR / "species_map.json") as f:
        old_map: dict = json.load(f)

    # Build old_index -> new_class mapping
    old_idx_to_new = {}
    for species, group_idx in GROUPS.items():
        old_idx = old_map[species]["index"]
        old_idx_to_new[old_idx] = group_idx

    print(f"Grouping {len(old_map)} species -> {len(CLASS_NAMES)} classes")
    print(f"  PINE:  PIPA2")
    print(f"  OAK:   QUAL, QUCO2, QULA2, QURU")
    print(f"  OTHER: ACRU, AMLA, NYSY")
    print()

    new_map = {}
    for i, name in enumerate(CLASS_NAMES):
        new_map[name] = {"index": i, "count": 0}

    total_stats = {"total_processed": 0, "n_points": 0, "n_classes": 3, "splits": {}}

    for split in SPLITS:
        src_split = SRC_DIR / split
        dst_split = DST_DIR / split
        dst_split.mkdir(parents=True, exist_ok=True)

        points = np.load(src_split / "points.npy")
        labels = np.load(src_split / "labels.npy")
        tree_ids = np.load(src_split / "tree_ids.npy") if (src_split / "tree_ids.npy").exists() else None

        # Remap all labels (no filtering, all samples used)
        new_labels = np.array([old_idx_to_new[int(l)] for l in labels], dtype=np.int64)

        np.save(dst_split / "points.npy", points)
        np.save(dst_split / "labels.npy", new_labels)
        if tree_ids is not None:
            np.save(dst_split / "tree_ids.npy", tree_ids)

        class_dist = {}
        for name in CLASS_NAMES:
            idx = new_map[name]["index"]
            count = int((new_labels == idx).sum())
            class_dist[name] = count
            new_map[name]["count"] += count

        total_stats["n_points"] = points.shape[1]
        total_stats["total_processed"] += len(new_labels)
        total_stats["splits"][split] = {
            "count": len(new_labels),
            "class_distribution": class_dist,
        }

        print(f"  {split}: {len(new_labels)} samples")
        for name, cnt in class_dist.items():
            print(f"    {name}: {cnt}")

    with open(DST_DIR / "species_map.json", "w") as f:
        json.dump(new_map, f, indent=2)
    with open(DST_DIR / "stats.json", "w") as f:
        json.dump(total_stats, f, indent=2)

    print(f"\nTotal: {total_stats['total_processed']} samples across {len(CLASS_NAMES)} classes")
    print(f"Saved to {DST_DIR}")


if __name__ == "__main__":
    main()
