"""
Dataset and data loading for point cloud segmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SegmentationDataset(Dataset):
    """Dataset of point cloud blocks with per-point segmentation labels."""

    def __init__(self, data_dir: str, split: str = "train",
                 augment: bool = False, max_samples: int = 0):
        self.data_dir = Path(data_dir) / split
        self.augment = augment

        self.points = np.load(self.data_dir / "points.npy")  # (N, num_points, 3)
        self.labels = np.load(self.data_dir / "labels.npy")  # (N, num_points)

        # Subsample if dataset is too large
        if max_samples > 0 and len(self.points) > max_samples:
            idx = np.random.choice(len(self.points), max_samples, replace=False)
            self.points = self.points[idx]
            self.labels = self.labels[idx]

        print(f"  {split}: {len(self.points)} blocks, "
              f"tree ratio: {(self.labels == 1).sum() / self.labels.size:.3f}")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx].copy()  # (num_points, 3)
        lbl = self.labels[idx].copy()  # (num_points,)

        if self.augment:
            pts, lbl = self._augment(pts, lbl)

        return torch.from_numpy(pts).float(), torch.from_numpy(lbl).long()

    def _augment(self, pts: np.ndarray, lbl: np.ndarray):
        """Apply augmentations to a point cloud block."""
        # Random rotation around Z axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0],
                        [sin_t, cos_t, 0],
                        [0, 0, 1]], dtype=np.float32)
        pts = pts @ rot.T

        # Random flip X and Y
        if np.random.random() > 0.5:
            pts[:, 0] *= -1
        if np.random.random() > 0.5:
            pts[:, 1] *= -1

        # Random jitter
        pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        pts = np.clip(pts, -2, 2)

        # Random anisotropic scale
        scale = np.random.uniform(0.9, 1.1, size=3).astype(np.float32)
        pts *= scale

        # Random point dropout + reshuffle (shuffling breaks spatial ordering bias)
        n = len(pts)
        drop_ratio = np.random.uniform(0, 0.1)
        keep_n = max(int(n * (1 - drop_ratio)), n // 2)
        idx = np.random.choice(n, keep_n, replace=False)
        # Pad back to original count
        if keep_n < n:
            pad_idx = np.random.choice(keep_n, n - keep_n, replace=True)
            idx = np.concatenate([idx, idx[pad_idx]])
        np.random.shuffle(idx)
        pts = pts[idx]
        lbl = lbl[idx]

        return pts, lbl


def build_dataloaders(cfg: dict) -> dict:
    """Build train/val/test dataloaders."""
    data_dir = cfg["data"]["processed_dir"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 0)
    pin_memory = cfg["train"].get("pin_memory", True)

    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            print(f"  Warning: {split} split not found at {split_dir}")
            continue

        augment = (split == "train")
        max_samples = cfg["data"].get("max_train_samples", 0) if split == "train" else 0
        ds = SegmentationDataset(data_dir, split, augment=augment,
                                 max_samples=max_samples)

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return loaders
