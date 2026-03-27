"""
PointNet++ for semantic segmentation — pure PyTorch implementation.

No CUDA extensions required. Uses kNN instead of ball query for
Windows compatibility.

Architecture:
  Encoder: Set Abstraction (SA) layers — FPS + kNN + shared MLP + maxpool
  Decoder: Feature Propagation (FP) layers — interpolation + MLP
  Head: Per-point MLP → binary segmentation (tree vs non-tree)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared distances between two point sets.

    Args:
        src: (B, N, 3)
        dst: (B, M, 3)

    Returns:
        dist: (B, N, M)
    """
    return torch.cdist(src, dst, p=2.0).pow(2)


def random_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Random point sampling (fast replacement for FPS).

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample

    Returns:
        indices: (B, npoint) — indices into xyz
    """
    B, N, _ = xyz.shape
    # Generate random indices per batch element
    idx = torch.stack([torch.randperm(N, device=xyz.device)[:npoint]
                       for _ in range(B)], dim=0)
    return idx


def knn(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """k-Nearest Neighbors.

    Args:
        xyz: (B, N, 3) — all points
        new_xyz: (B, S, 3) — query points
        k: number of neighbors

    Returns:
        idx: (B, S, k) — indices into xyz
    """
    dist = square_distance(new_xyz, xyz)  # (B, S, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, S, k)
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index into points using idx.

    Args:
        points: (B, N, C)
        idx: (B, ...) — indices

    Returns:
        indexed: (B, ..., C)
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device)
    batch_indices = batch_indices.view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


# ---------------------------------------------------------------------------
# PointNet++ Layers
# ---------------------------------------------------------------------------

class SharedMLP(nn.Module):
    """Shared MLP applied per-point (1D convolutions)."""

    def __init__(self, channels: list, bn: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N) → (B, C_out, N)"""
        return self.mlp(x)


class SetAbstraction(nn.Module):
    """Set Abstraction layer: sample + group + PointNet."""

    def __init__(self, npoint: int, nsample: int, in_channel: int,
                 mlp_channels: list):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = SharedMLP([in_channel + 3] + mlp_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3) — point positions
            features: (B, N, C) — per-point features (or None)

        Returns:
            new_xyz: (B, npoint, 3) — sampled positions
            new_features: (B, npoint, C') — aggregated features
        """
        B, N, _ = xyz.shape

        # 1. Farthest Point Sampling
        fps_idx = random_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        # 2. kNN Grouping
        idx = knn(xyz, new_xyz, self.nsample)  # (B, npoint, nsample)
        grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # relative coordinates

        # 3. Concatenate features
        if features is not None:
            grouped_features = index_points(features, idx)  # (B, npoint, nsample, C)
            grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped = grouped_xyz  # (B, npoint, nsample, 3)

        # 4. Shared MLP + max pool
        # Reshape: (B, npoint, nsample, C) → (B*npoint, C, nsample)
        B_np = B * self.npoint
        grouped = grouped.view(B_np, self.nsample, -1).transpose(1, 2)  # (B*npoint, C, nsample)
        grouped = self.mlp(grouped)  # (B*npoint, C', nsample)
        new_features = grouped.max(dim=-1)[0]  # (B*npoint, C')
        new_features = new_features.view(B, self.npoint, -1)  # (B, npoint, C')

        return new_xyz, new_features


class SetAbstractionAll(nn.Module):
    """Set Abstraction that groups ALL points (global feature)."""

    def __init__(self, in_channel: int, mlp_channels: list):
        super().__init__()
        self.mlp = SharedMLP([in_channel + 3] + mlp_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Returns:
            new_xyz: (B, 1, 3) — centroid
            new_features: (B, 1, C') — global features
        """
        B, N, _ = xyz.shape
        new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
        relative_xyz = xyz - new_xyz  # (B, N, 3)

        if features is not None:
            grouped = torch.cat([relative_xyz, features], dim=-1)
        else:
            grouped = relative_xyz

        # (B, N, C) → (B, C, N)
        grouped = grouped.transpose(1, 2)
        grouped = self.mlp(grouped)  # (B, C', N)
        new_features = grouped.max(dim=-1)[0].unsqueeze(1)  # (B, 1, C')

        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    """Feature Propagation layer: interpolate + concatenate + MLP."""

    def __init__(self, in_channel: int, mlp_channels: list):
        super().__init__()
        self.mlp = SharedMLP([in_channel] + mlp_channels)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Propagate features from xyz2 (fewer points) to xyz1 (more points).

        Args:
            xyz1: (B, N, 3) — target points (higher resolution)
            xyz2: (B, S, 3) — source points (lower resolution)
            features1: (B, N, C1) — skip-connection features at xyz1 (or None)
            features2: (B, S, C2) — features at xyz2

        Returns:
            new_features: (B, N, C') — propagated features
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Global feature: broadcast to all points
            interpolated = features2.repeat(1, N, 1)  # (B, N, C2)
        else:
            # 3-nearest neighbor interpolation
            dist = square_distance(xyz1, xyz2)  # (B, N, S)
            dist, idx = dist.topk(3, dim=-1, largest=False)  # (B, N, 3)
            dist_recip = 1.0 / (dist + 1e-8)  # (B, N, 3)
            weights = dist_recip / dist_recip.sum(dim=-1, keepdim=True)  # (B, N, 3)

            interpolated_features = index_points(features2, idx)  # (B, N, 3, C2)
            interpolated = (interpolated_features * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, C2)

        # Concatenate with skip features
        if features1 is not None:
            concatenated = torch.cat([interpolated, features1], dim=-1)  # (B, N, C1+C2)
        else:
            concatenated = interpolated

        # MLP: (B, N, C) → (B, C, N) → MLP → (B, C', N) → (B, N, C')
        concatenated = concatenated.transpose(1, 2)
        new_features = self.mlp(concatenated)
        new_features = new_features.transpose(1, 2)

        return new_features


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class PointNet2Segmentation(nn.Module):
    """PointNet++ for binary semantic segmentation.

    Encoder: 3 Set Abstraction layers + 1 global SA
    Decoder: 3 Feature Propagation layers
    Head: Per-point MLP → 2-class logits
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        sa_npoints = model_cfg["sa_npoints"]
        sa_nsamples = model_cfg["sa_nsamples"]
        sa_mlps = model_cfg["sa_mlps"]
        fp_mlps = model_cfg["fp_mlps"]
        seg_mlp = model_cfg["seg_mlp"]
        num_classes = model_cfg["num_classes"]
        dropout = model_cfg.get("dropout", 0.3)

        # --- Encoder ---
        # SA1: N → sa_npoints[0]
        self.sa1 = SetAbstraction(sa_npoints[0], sa_nsamples[0],
                                  in_channel=0, mlp_channels=sa_mlps[0])
        # SA2: sa_npoints[0] → sa_npoints[1]
        self.sa2 = SetAbstraction(sa_npoints[1], sa_nsamples[1],
                                  in_channel=sa_mlps[0][-1],
                                  mlp_channels=sa_mlps[1])
        # SA3: sa_npoints[1] → sa_npoints[2]
        self.sa3 = SetAbstraction(sa_npoints[2], sa_nsamples[2],
                                  in_channel=sa_mlps[1][-1],
                                  mlp_channels=sa_mlps[2])
        # SA4: global
        self.sa4 = SetAbstractionAll(in_channel=sa_mlps[2][-1],
                                     mlp_channels=[256, 512])

        # --- Decoder ---
        # FP4: from SA4 (512) + SA3 (256)
        self.fp4 = FeaturePropagation(512 + sa_mlps[2][-1], fp_mlps[0])
        # FP3: from FP4 (fp_mlps[0][-1]) + SA2 (128)
        self.fp3 = FeaturePropagation(fp_mlps[0][-1] + sa_mlps[1][-1], fp_mlps[1])
        # FP2: from FP3 (fp_mlps[1][-1]) + SA1 (64)
        self.fp2 = FeaturePropagation(fp_mlps[1][-1] + sa_mlps[0][-1], fp_mlps[2])
        # FP1: from FP2 (fp_mlps[2][-1]) + raw input (no features, just xyz=3)
        self.fp1 = FeaturePropagation(fp_mlps[2][-1] + 3, [128, 128])

        # --- Segmentation head ---
        head_layers = []
        in_ch = 128
        for out_ch in seg_mlp:
            head_layers.extend([
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        head_layers.append(nn.Conv1d(in_ch, num_classes, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) — input point cloud

        Returns:
            logits: (B, N, num_classes) — per-point predictions
        """
        B, N, _ = xyz.shape

        # Keep raw xyz for FP1 skip connection
        l0_xyz = xyz
        l0_features = xyz  # use xyz as initial features for FP1 skip

        # --- Encoder ---
        l1_xyz, l1_features = self.sa1(l0_xyz, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)

        # --- Decoder ---
        l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(l0_xyz, l1_xyz, l0_features, l1_features)

        # --- Head ---
        # (B, N, C) → (B, C, N)
        x = l0_features.transpose(1, 2)
        logits = self.head(x)  # (B, num_classes, N)
        logits = logits.transpose(1, 2)  # (B, N, num_classes)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> PointNet2Segmentation:
    """Build PointNet++ segmentation model from config."""
    model = PointNet2Segmentation(cfg)
    print(f"PointNet++ Segmentation: {model.count_parameters():,} parameters")
    return model
