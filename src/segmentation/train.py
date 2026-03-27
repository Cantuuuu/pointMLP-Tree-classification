"""
Training script for PointNet++ tree segmentation.

Loss: BCE + Dice (configurable)
Metrics: IoU, Accuracy, Precision, Recall per class
"""

import sys
import os

import argparse
import json
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.segmentation.pointnet2_model import build_model
from src.segmentation.dataset import build_dataloaders


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, N, 2) — raw model output
            targets: (B, N) — binary labels
        """
        probs = F.softmax(logits, dim=-1)[:, :, 1]  # P(tree)
        targets_f = targets.float()

        intersection = (probs * targets_f).sum()
        union = probs.sum() + targets_f.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class SegmentationLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, pos_weight: float = 1.0, use_dice: bool = True,
                 dice_weight: float = 0.5):
        super().__init__()
        weight = torch.tensor([1.0, pos_weight])
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        if use_dice:
            self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, N, 2)
            targets: (B, N)
        """
        B, N, C = logits.shape
        # CE expects (B*N, C) and (B*N,)
        ce = self.ce_loss(logits.reshape(-1, C), targets.reshape(-1))

        if self.use_dice:
            dice = self.dice_loss(logits, targets)
            return ce + self.dice_weight * dice

        return ce


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute segmentation metrics.

    Args:
        logits: (B, N, 2)
        targets: (B, N)

    Returns:
        dict with accuracy, iou, precision, recall per class
    """
    preds = logits.argmax(dim=-1)  # (B, N)

    # Flatten
    preds_flat = preds.reshape(-1).cpu().numpy()
    targets_flat = targets.reshape(-1).cpu().numpy()

    # Overall accuracy
    correct = (preds_flat == targets_flat).sum()
    total = len(targets_flat)
    accuracy = correct / total

    metrics = {"accuracy": accuracy}

    # Per-class IoU, precision, recall
    for cls, name in [(0, "non_tree"), (1, "tree")]:
        tp = ((preds_flat == cls) & (targets_flat == cls)).sum()
        fp = ((preds_flat == cls) & (targets_flat != cls)).sum()
        fn = ((preds_flat != cls) & (targets_flat == cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_iou"] = iou

    # Mean IoU
    metrics["mean_iou"] = (metrics["non_tree_iou"] + metrics["tree_iou"]) / 2

    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0
    all_logits = []
    all_targets = []

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(points)  # (B, N, 2)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * points.size(0)
        all_logits.append(logits.detach())
        all_targets.append(labels.detach())

    avg_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_logits = []
    all_targets = []

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)

        logits = model(points)
        loss = criterion(logits, labels)

        total_loss += loss.item() * points.size(0)
        all_logits.append(logits)
        all_targets.append(labels)

    avg_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = avg_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ segmentation")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lr:
        cfg["train"]["learning_rate"] = args.lr

    # Seed
    seed = cfg.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device(cfg["hardware"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cfg["hardware"].get("cudnn_benchmark", True)

    # Data
    print("\nLoading data...")
    loaders = build_dataloaders(cfg)

    if "train" not in loaders:
        print("ERROR: No training data found!")
        return

    # Model
    print("\nBuilding model...")
    model = build_model(cfg)
    model = model.to(device)

    # Loss
    pos_weight = cfg["train"].get("pos_weight", 1.0)
    use_dice = cfg["train"].get("use_dice_loss", True)
    dice_weight = cfg["train"].get("dice_weight", 0.5)
    criterion = SegmentationLoss(pos_weight, use_dice, dice_weight)
    criterion.ce_loss.weight = criterion.ce_loss.weight.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Scheduler
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 5)
    scheduler = CosineWarmupScheduler(optimizer, warmup, epochs)

    # Early stopping
    es_cfg = cfg["train"]["early_stopping"]
    best_metric = -float("inf") if es_cfg["mode"] == "max" else float("inf")
    patience_counter = 0
    best_epoch = 0

    # Output directory
    log_dir = Path(cfg["experiment"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    csv_path = log_dir / "training_log.csv"
    csv_fields = ["epoch", "lr", "train_loss", "train_acc", "train_iou",
                  "val_loss", "val_acc", "val_iou", "val_tree_iou",
                  "val_tree_prec", "val_tree_rec"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    grad_clip = cfg["train"].get("gradient_clip", 5.0)

    print(f"\nTraining for up to {epochs} epochs...")
    print(f"  Batch size: {cfg['train']['batch_size']}")
    print(f"  LR: {cfg['train']['learning_rate']}")
    print(f"  Pos weight: {pos_weight}")
    print(f"  Dice loss: {use_dice} (weight={dice_weight})")
    print(f"  Early stopping: patience={es_cfg['patience']} on {es_cfg['metric']}")
    print()

    for epoch in range(epochs):
        t0 = time.time()
        lr = scheduler.step(epoch)

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, grad_clip)

        # Validate
        val_metrics = {}
        if "val" in loaders:
            val_metrics = evaluate(model, loaders["val"], criterion, device)

        dt = time.time() - t0

        # Log
        val_iou = val_metrics.get("mean_iou", 0)
        tree_iou = val_metrics.get("tree_iou", 0)
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"lr={lr:.6f} | "
              f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} iou={train_metrics['mean_iou']:.4f} | "
              f"val_loss={val_metrics.get('loss', 0):.4f} acc={val_metrics.get('accuracy', 0):.4f} "
              f"iou={val_iou:.4f} tree_iou={tree_iou:.4f} | "
              f"{dt:.1f}s")

        # CSV
        row = {
            "epoch": epoch + 1,
            "lr": f"{lr:.8f}",
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_acc": f"{train_metrics['accuracy']:.6f}",
            "train_iou": f"{train_metrics['mean_iou']:.6f}",
            "val_loss": f"{val_metrics.get('loss', 0):.6f}",
            "val_acc": f"{val_metrics.get('accuracy', 0):.6f}",
            "val_iou": f"{val_iou:.6f}",
            "val_tree_iou": f"{tree_iou:.6f}",
            "val_tree_prec": f"{val_metrics.get('tree_precision', 0):.6f}",
            "val_tree_rec": f"{val_metrics.get('tree_recall', 0):.6f}",
        }
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        # Early stopping
        metric_key = es_cfg["metric"].replace("val_", "")
        current_metric = val_metrics.get(metric_key, val_metrics.get("mean_iou", 0))

        improved = (current_metric > best_metric if es_cfg["mode"] == "max"
                    else current_metric < best_metric)

        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": cfg,
            }, log_dir / "best_model.pth")
            print(f"  >> New best {es_cfg['metric']}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= es_cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(best: epoch {best_epoch}, {es_cfg['metric']}={best_metric:.4f})")
                break

    # Save last model
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, log_dir / "last_model.pth")

    # Final evaluation on test set
    if "test" in loaders:
        print("\n" + "=" * 60)
        print("Test set evaluation (best model)")
        print("=" * 60)
        checkpoint = torch.load(log_dir / "best_model.pth", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], criterion, device)

        print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
        print(f"  Mean IoU:       {test_metrics['mean_iou']:.4f}")
        print(f"  Tree IoU:       {test_metrics['tree_iou']:.4f}")
        print(f"  Tree Precision: {test_metrics['tree_precision']:.4f}")
        print(f"  Tree Recall:    {test_metrics['tree_recall']:.4f}")
        print(f"  Non-tree IoU:   {test_metrics['non_tree_iou']:.4f}")

        with open(log_dir / "test_results.json", "w") as f:
            json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    print(f"\nResults saved to {log_dir}")
    print("Done!")


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
