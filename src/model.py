"""PointMLP-Elite para clasificacion de arboles.

Implementacion from scratch basada en:
"Rethinking Network Design and Local Geometry in Point Cloud:
 A Simple Residual MLP Framework" (Ma et al., ICLR 2022).

Simplificada para caber en 6GB VRAM con batch_size=8, 1024 puntos.

Uso:
    from src.model import get_model, count_parameters
    model = get_model(config)
    print(f"{count_parameters(model):,} parameters")
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ── kNN ───────────────────────────────────────────────────────────────


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-Nearest Neighbors using pairwise distances.

    Args:
        x: (B, N, C) point coordinates or features.
        k: number of neighbors.

    Returns:
        idx: (B, N, k) indices of k nearest neighbors.
    """
    # (B, N, N) pairwise squared distances
    dists = torch.cdist(x, x)  # caller ensures FP32
    _, idx = dists.topk(k, dim=-1, largest=False)  # smallest k distances
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C)
        idx: (B, N, k) or (B, M)

    Returns:
        gathered: (B, N, k, C) or (B, M, C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


# ── Farthest Point Sampling (GPU) ────────────────────────────────────


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """Random sampling as fast alternative to FPS for downsampling.

    Uses random subset selection which is O(1) on GPU vs O(n_points*N) for FPS.
    Quality is close enough for training; FPS matters more for very sparse clouds.

    Args:
        xyz: (B, N, 3) input points.
        n_points: number of points to sample.

    Returns:
        idx: (B, n_points) indices of sampled points.
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Random sampling: generate random indices per batch element
    idx = torch.stack([
        torch.randperm(N, device=device)[:n_points] for _ in range(B)
    ])
    return idx


# ── Geometric Affine Module ──────────────────────────────────────────


class GeometricAffine(nn.Module):
    """Learnable affine transformation for geometric normalization.

    Normalizes features with learned alpha (scale) and beta (bias).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, k, C) -> (B, N, k, C)"""
        mean = x.mean(dim=-2, keepdim=True)  # (B, N, 1, C)
        std = x.std(dim=-2, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        return x * self.alpha + self.beta


# ── Residual MLP Block ───────────────────────────────────────────────


class ResidualMLP(nn.Module):
    """Residual block: Linear -> LN -> ReLU -> Linear -> LN + skip.

    Uses LayerNorm instead of BatchNorm for stability with mixed precision
    and variable effective batch sizes (B*N*k reshapes).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, C) -> (B*N, C)"""
        return self.relu(self.net(x) + self.skip(x))


# ── PointMLP Stage ───────────────────────────────────────────────────


class PointMLPStage(nn.Module):
    """One stage of PointMLP feature extraction.

    1. kNN local grouping
    2. Geometric affine normalization
    3. Pre-norm residual MLP (on grouped features)
    4. Aggregate (max pool over neighbors)
    5. Post-norm residual MLP
    6. Farthest point sampling to reduce points

    Args:
        in_dim: input feature dimension.
        out_dim: output feature dimension.
        n_points_out: number of points after downsampling.
        k: number of nearest neighbors.
        pre_blocks: number of pre-norm residual blocks.
        pos_blocks: number of post-norm residual blocks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_points_out: int,
        k: int = 20,
        pre_blocks: int = 2,
        pos_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.k = k
        self.n_points_out = n_points_out

        # Geometric affine on grouped features (in_dim * 2 because we concat relative + absolute)
        self.affine = GeometricAffine(in_dim)

        # Pre-norm MLPs: process concatenated features [center || neighbor_relative]
        pre_layers = []
        cur_dim = in_dim * 2
        for i in range(pre_blocks):
            next_dim = out_dim if i == pre_blocks - 1 else cur_dim
            pre_layers.append(ResidualMLP(cur_dim, next_dim))
            cur_dim = next_dim
        self.pre_mlp = nn.Sequential(*pre_layers)

        # Post-norm MLPs
        pos_layers = []
        for _ in range(pos_blocks):
            pos_layers.append(ResidualMLP(out_dim, out_dim))
        self.pos_mlp = nn.Sequential(*pos_layers)

    def forward(
        self, features: torch.Tensor, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, N, C_in) point features.
            xyz: (B, N, 3) point coordinates.

        Returns:
            new_features: (B, M, C_out) downsampled features.
            new_xyz: (B, M, 3) downsampled coordinates.
        """
        # Disable AMP for entire stage — B*N*k reshapes + cdist + std cause
        # FP16 overflow/underflow that produces NaN gradients in backward pass.
        with torch.amp.autocast("cuda", enabled=False):
            features = features.float()
            xyz = xyz.float()
            B, N, C = features.shape

            # 1. kNN grouping
            idx = knn(xyz, self.k)                          # (B, N, k)
            grouped = index_points(features, idx)           # (B, N, k, C)

            # 2. Geometric affine normalization
            grouped = self.affine(grouped)                  # (B, N, k, C)

            # 3. Build local features: center expanded + grouped relative
            center = features.unsqueeze(2).expand_as(grouped)  # (B, N, k, C)
            local_feat = torch.cat([center, grouped - center], dim=-1)  # (B, N, k, 2C)

            # 4. Pre-norm MLP
            BNk = B * N * self.k
            local_feat = local_feat.reshape(BNk, -1)
            local_feat = self.pre_mlp(local_feat)           # (B*N*k, C_out)
            local_feat = local_feat.reshape(B, N, self.k, -1)

            # 5. Max pool over neighbors
            features = local_feat.max(dim=2).values         # (B, N, C_out)

            # 6. Post-norm MLP
            BN = B * N
            features = features.reshape(BN, -1)
            features = self.pos_mlp(features)               # (B*N, C_out)
            features = features.reshape(B, N, -1)

            # 7. FPS downsample
            if self.n_points_out < N:
                fps_idx = farthest_point_sample(xyz, self.n_points_out)  # (B, M)
                new_xyz = index_points(xyz, fps_idx)           # (B, M, 3)
                new_features = index_points(features, fps_idx) # (B, M, C_out)
            else:
                new_xyz = xyz
                new_features = features

        return new_features, new_xyz


# ── PointMLP Classifier ──────────────────────────────────────────────


class PointMLPClassifier(nn.Module):
    """PointMLP-Elite for tree species classification.

    Args:
        n_classes: Number of species to classify.
        embedding_dim: Initial embedding dimension.
        stage_dims: Feature dimensions for each stage.
        stage_points: Number of points at each stage.
        k: Number of nearest neighbors.
        pre_blocks: Residual blocks before aggregation.
        pos_blocks: Residual blocks after aggregation.
        dropout: Dropout rate in classifier head.
    """

    def __init__(
        self,
        n_classes: int = 8,
        embedding_dim: int = 64,
        stage_dims: list[int] | None = None,
        stage_points: list[int] | None = None,
        k: int = 20,
        pre_blocks: int = 2,
        pos_blocks: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if stage_dims is None:
            stage_dims = [64, 128, 256, 512]
        if stage_points is None:
            stage_points = [1024, 512, 256, 64]

        assert len(stage_dims) == len(stage_points)
        self.n_stages = len(stage_dims)

        # Initial embedding: (B, N, 3) -> (B, N, embedding_dim)
        self.embedding = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Stages
        self.stages = nn.ModuleList()
        in_dim = embedding_dim
        for i in range(self.n_stages):
            self.stages.append(
                PointMLPStage(
                    in_dim=in_dim,
                    out_dim=stage_dims[i],
                    n_points_out=stage_points[i],
                    k=k,
                    pre_blocks=pre_blocks,
                    pos_blocks=pos_blocks,
                )
            )
            in_dim = stage_dims[i]

        # Classifier head: global max+avg pool -> MLP
        final_dim = stage_dims[-1] * 2  # concat max and avg pooling
        head_dim = min(256, final_dim)  # scale head to match feature dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, head_dim),
            nn.BatchNorm1d(head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.BatchNorm1d(head_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, N, 3) batch of point clouds.

        Returns:
            logits: (B, n_classes).
        """
        # Disable AMP for entire model — FP16 causes NaN gradients in
        # embedding (BN1d with B*N reshape), stages (cdist, affine, B*N*k),
        # and classifier head (large linear layers overflow).
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            B, N, _ = x.shape
            xyz = x  # keep coordinates for kNN

            # Initial embedding
            features = x.reshape(B * N, 3)
            features = self.embedding(features)
            features = features.reshape(B, N, -1)

            # Stages
            for stage in self.stages:
                features, xyz = stage(features, xyz)

            # Global pooling: max + avg
            max_pool = features.max(dim=1).values   # (B, C)
            avg_pool = features.mean(dim=1)         # (B, C)
            global_feat = torch.cat([max_pool, avg_pool], dim=-1)  # (B, 2C)

            # Classifier
            logits = self.classifier(global_feat)
        return logits


# ── Factory functions ─────────────────────────────────────────────────


def get_model(config: dict) -> PointMLPClassifier:
    """Instantiate PointMLP from config dict."""
    model_cfg = config["model"]
    data_cfg = config["data"]

    model = PointMLPClassifier(
        n_classes=data_cfg["num_classes"],
        embedding_dim=model_cfg["embedding_dim"],
        stage_dims=model_cfg.get("stage_dims", [64, 128, 256, 512]),
        stage_points=model_cfg.get("stage_points", [1024, 512, 256, 64]),
        k=model_cfg["k_neighbors"],
        pre_blocks=model_cfg["pre_blocks"],
        pos_blocks=model_cfg["pos_blocks"],
        dropout=model_cfg["dropout"],
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── VRAM test ─────────────────────────────────────────────────────────


def test_model_memory(batch_size: int = 8, n_points: int = 1024) -> None:
    """Run a dummy forward pass and report VRAM usage."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping VRAM test.")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    config = {
        "model": {
            "embedding_dim": 64,
            "stage_dims": [64, 128, 256, 512],
            "stage_points": [1024, 512, 256, 64],
            "k_neighbors": 20,
            "pre_blocks": 2,
            "pos_blocks": 2,
            "dropout": 0.5,
        },
        "data": {"num_classes": 8},
    }

    model = get_model(config).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    dummy = torch.randn(batch_size, n_points, 3, device=device)

    # Forward + backward to measure peak
    with torch.amp.autocast("cuda"):
        logits = model(dummy)
        loss = logits.sum()
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"Peak VRAM (batch_size={batch_size}): {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")

    if peak_mb > 5120:
        print(f"WARNING: peak VRAM > 5GB. Reduce batch_size or k_neighbors.")
    else:
        print(f"OK: fits within 6GB VRAM.")

    del model, dummy
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = get_model(config)
    print(f"PointMLP-Elite: {count_parameters(model):,} parameters")
    print(f"Input: (B, {config['data']['num_points']}, 3)")
    print(f"Output: (B, {config['data']['num_classes']})")

    # Quick forward test on CPU
    dummy = torch.randn(2, config["data"]["num_points"], 3)
    logits = model(dummy)
    print(f"Forward pass OK: {dummy.shape} -> {logits.shape}")

    # VRAM test if CUDA available
    test_model_memory(batch_size=config["train"]["batch_size"])
