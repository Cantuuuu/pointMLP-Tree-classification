"""TreeDataset y DataLoader para nubes de puntos preprocesadas.

Carga los .npy generados por preprocess.py, aplica data augmentation
(rotacion, jitter, escala) y entrega batches listos para PointMLP.

Uso:
    from src.dataset import get_dataloaders
    loaders = get_dataloaders(Path("data/processed/real"), batch_size=8)
    for points, labels in loaders["train"]:
        ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ── Data augmentation ─────────────────────────────────────────────────


def random_rotate_z(points: np.ndarray) -> np.ndarray:
    """Rotate point cloud randomly around Z axis (0-360 degrees)."""
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1],
    ], dtype=np.float32)
    return points @ rotation.T


def random_jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Add Gaussian noise clipped to [-clip, clip]."""
    noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip)
    return points + noise.astype(np.float32)


def random_scale(points: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    """Scale uniformly by a random factor."""
    factor = np.random.uniform(low, high)
    return points * factor


def random_translate(points: np.ndarray, shift: float = 0.1) -> np.ndarray:
    """Translate by a random vector in [-shift, shift]."""
    offset = np.random.uniform(-shift, shift, size=(1, 3)).astype(np.float32)
    return points + offset


def random_point_dropout(points: np.ndarray, max_dropout: float = 0.15) -> np.ndarray:
    """Randomly drop points and duplicate remaining to keep shape.

    Simulates occlusion / sensor noise.
    """
    N = points.shape[0]
    drop_ratio = np.random.uniform(0, max_dropout)
    n_drop = int(N * drop_ratio)
    if n_drop == 0:
        return points
    keep_mask = np.ones(N, dtype=bool)
    drop_idx = np.random.choice(N, n_drop, replace=False)
    keep_mask[drop_idx] = False
    kept = points[keep_mask]
    # Duplicate random kept points to fill back to N
    fill_idx = np.random.choice(len(kept), n_drop, replace=True)
    return np.concatenate([kept, kept[fill_idx]], axis=0)


def random_anisotropic_scale(points: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    """Scale each axis independently for more diversity."""
    factors = np.random.uniform(low, high, size=(1, 3)).astype(np.float32)
    return points * factors


def augment_point_cloud(points: np.ndarray) -> np.ndarray:
    """Apply all augmentations to a point cloud (training)."""
    points = random_rotate_z(points)
    points = random_jitter(points, sigma=0.02, clip=0.05)
    points = random_anisotropic_scale(points, low=0.85, high=1.15)
    points = random_translate(points, shift=0.15)
    points = random_point_dropout(points, max_dropout=0.15)
    return points


def augment_point_cloud_light(points: np.ndarray) -> np.ndarray:
    """Light augmentation for TTA (rotation + mild jitter only)."""
    points = random_rotate_z(points)
    points = random_jitter(points, sigma=0.005, clip=0.02)
    return points


# ── Mixup (batch-level, called from training loop) ───────────────────


def mixup_point_clouds(
    points: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup augmentation for point clouds.

    Args:
        points: (B, N, 3)
        labels: (B,) integer class labels
        n_classes: number of classes (for one-hot)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_points: (B, N, 3)
        mixed_labels: (B, n_classes) soft labels
    """
    B = points.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5 so original dominates

    perm = torch.randperm(B, device=points.device)
    mixed_points = lam * points + (1.0 - lam) * points[perm]

    # One-hot encode and mix labels
    labels_onehot = torch.zeros(B, n_classes, device=labels.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    labels_perm = labels_onehot[perm]
    mixed_labels = lam * labels_onehot + (1.0 - lam) * labels_perm

    return mixed_points, mixed_labels


# ── Dataset ───────────────────────────────────────────────────────────


class TreeDataset(Dataset):
    """Dataset of preprocessed tree point clouds.

    Each sample is a (num_points, 3) float32 tensor with an int64 label.

    Args:
        data_dir: Path to split directory (e.g. data/processed/real/train/)
        augment: Whether to apply data augmentation.
        n_points: Expected number of points per sample (for validation).
    """

    def __init__(
        self,
        data_dir: Path,
        augment: bool = False,
        n_points: int = 1024,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment = augment

        self.points = np.load(self.data_dir / "points.npy")  # (N, n_points, 3)
        self.labels = np.load(self.data_dir / "labels.npy")   # (N,)

        assert self.points.shape[0] == self.labels.shape[0], (
            f"Mismatch: {self.points.shape[0]} points vs {self.labels.shape[0]} labels"
        )
        assert self.points.shape[1] == n_points, (
            f"Expected {n_points} points per sample, got {self.points.shape[1]}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        points = self.points[idx].copy()  # (n_points, 3)
        label = int(self.labels[idx])

        # Center at origin (preserve scale — tree size is discriminative)
        centroid = points.mean(axis=0, keepdims=True)
        points = points - centroid

        if self.augment:
            points = augment_point_cloud(points)

        return (
            torch.from_numpy(points).float(),   # (n_points, 3)
            torch.tensor(label, dtype=torch.long),
        )

    def get_class_counts(self) -> np.ndarray:
        """Return array of sample counts per class."""
        n_classes = int(self.labels.max()) + 1
        counts = np.zeros(n_classes, dtype=np.int64)
        for c in range(n_classes):
            counts[c] = (self.labels == c).sum()
        return counts


# ── DataLoaders ───────────────────────────────────────────────────────


def get_dataloaders(
    processed_dir: Path,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment_train: bool = True,
    n_points: int = 1024,
) -> dict:
    """Create DataLoaders for train, val, test splits.

    Returns dict with keys: 'train', 'val', 'test', 'class_weights', 'n_classes'.
    Train loader uses WeightedRandomSampler for class balancing.
    """
    processed_dir = Path(processed_dir)

    train_ds = TreeDataset(processed_dir / "train", augment=augment_train, n_points=n_points)
    val_ds = TreeDataset(processed_dir / "val", augment=False, n_points=n_points)
    test_ds = TreeDataset(processed_dir / "test", augment=False, n_points=n_points)

    # ── Weighted sampler for train (balance classes) ──
    class_counts = train_ds.get_class_counts()
    n_classes = len(class_counts)

    # Per-sample weight = 1 / freq(class)
    class_freq = class_counts.astype(np.float64)
    class_freq[class_freq == 0] = 1.0  # avoid div by zero
    sample_weights = 1.0 / class_freq[train_ds.labels.astype(int)]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    # ── Class weights for loss function: 1/sqrt(freq), normalized ──
    inv_sqrt = 1.0 / np.sqrt(class_freq)
    inv_sqrt /= inv_sqrt.sum()
    inv_sqrt *= n_classes  # scale so mean weight = 1
    class_weights = torch.from_numpy(inv_sqrt).float()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "n_classes": n_classes,
    }


# ── Quick test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/real")
    loaders = get_dataloaders(data_dir, batch_size=8)

    print(f"Classes: {loaders['n_classes']}")
    print(f"Class weights: {loaders['class_weights']}")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")

    for points, labels in loaders["train"]:
        print(f"Batch: points={points.shape} labels={labels.shape}")
        print(f"  Points range: [{points.min():.4f}, {points.max():.4f}]")
        print(f"  Labels: {labels.tolist()}")
        break
