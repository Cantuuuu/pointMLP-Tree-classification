"""Training loop para PointMLP con mixed precision y early stopping.

Entrena el modelo sobre un dataset (real o sintetico), guarda checkpoints,
loguea metricas a TensorBoard y CSV, y aplica early stopping.

Uso:
    python src/train.py --dataset real --exp_name model_B
    python src/train.py --dataset synthetic --exp_name model_A
    python src/train.py --dataset real --exp_name model_B --batch_size 4 --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# UTF-8 on Windows
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from src.dataset import get_dataloaders, mixup_point_clouds
    from src.model import PointMLPClassifier, count_parameters, get_model
except ModuleNotFoundError:
    # Allow running as `python src/train.py` from project root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.dataset import get_dataloaders, mixup_point_clouds
    from src.model import PointMLPClassifier, count_parameters, get_model


# ── Reproducibility ───────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── LR Scheduler with Warmup ─────────────────────────────────────────


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup.

    LR ramps linearly from 0 to base_lr over warmup_epochs,
    then decays via cosine to lr_min.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        lr_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.lr_min + (base_lr - self.lr_min) * factor
                for base_lr in self.base_lrs
            ]


# ── Early Stopping ────────────────────────────────────────────────────


class EarlyStopping:
    """Stop training when a metric stops improving."""

    def __init__(self, patience: int = 15, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_value = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0

    def step(self, value: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        improved = (
            value > self.best_value if self.mode == "max"
            else value < self.best_value
        )
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ── Training & Validation ─────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float = 10.0,
    use_amp: bool = True,
    use_mixup: bool = False,
    n_classes: int = 5,
    mixup_alpha: float = 0.4,
) -> dict[str, float]:
    """Train one epoch. Returns dict with loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for points, labels in loader:
        points = points.to(device, non_blocking=True)  # (B, N, 3)
        labels = labels.to(device, non_blocking=True)   # (B,)

        # Apply mixup at batch level
        mixed_labels = None
        if use_mixup:
            points, mixed_labels = mixup_point_clouds(
                points, labels, n_classes=n_classes, alpha=mixup_alpha
            )

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(points)
                if mixed_labels is not None:
                    # Soft cross-entropy for mixup
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    loss = -(mixed_labels * log_probs).sum(dim=-1).mean()
                else:
                    loss = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(points)
            if mixed_labels is not None:
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = -(mixed_labels * log_probs).sum(dim=-1).mean()
            else:
                loss = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected, skipping batch")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        n_batches += 1

    if total == 0:
        return {"loss": float("nan"), "accuracy": 0.0}

    return {
        "loss": total_loss / total,
        "accuracy": correct / total * 100,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Validate on a dataset. Returns dict with loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in loader:
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(points)
                loss = criterion(logits, labels)
        else:
            logits = model(points)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return {"loss": float("nan"), "accuracy": 0.0}

    return {
        "loss": total_loss / total,
        "accuracy": correct / total * 100,
    }


# ── Main Training Pipeline ───────────────────────────────────────────


def train(
    config: dict,
    dataset_type: str,
    exp_name: str,
    batch_size_override: int | None = None,
    epochs_override: int | None = None,
    lr_override: float | None = None,
) -> None:
    """Full training pipeline."""

    train_cfg = config["train"]
    hw_cfg = config["hardware"]
    exp_cfg = config["experiment"]
    data_cfg = config["data"]

    seed = exp_cfg["seed"]
    set_seed(seed)

    # Resolve paths
    processed_dir = Path(data_cfg["processed_dir"]) / dataset_type
    exp_dir = Path(exp_cfg["log_dir"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load species map for n_classes
    species_map_path = processed_dir / "species_map.json"
    if species_map_path.exists():
        with open(species_map_path) as f:
            species_map = json.load(f)
        n_classes = len(species_map)
        data_cfg["num_classes"] = n_classes
    else:
        n_classes = data_cfg["num_classes"]

    # Parameters (CLI overrides)
    batch_size = batch_size_override or train_cfg["batch_size"]
    epochs = epochs_override or train_cfg["epochs"]
    lr = lr_override or train_cfg["learning_rate"]
    use_amp = hw_cfg["mixed_precision"]

    # Device
    if hw_cfg["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Device: CUDA ({gpu_name}, {gpu_mem:.0f}GB)")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("Device: CPU (WARNING: training will be slow)")

    # ── DataLoaders ──
    print(f"\nLoading data from {processed_dir}...")
    loaders = get_dataloaders(
        processed_dir,
        batch_size=batch_size,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        n_points=data_cfg["num_points"],
    )
    class_weights = loaders["class_weights"].to(device)

    print(f"  Train: {len(loaders['train'].dataset)} samples, {len(loaders['train'])} batches")
    print(f"  Val:   {len(loaders['val'].dataset)} samples")
    print(f"  Test:  {len(loaders['test'].dataset)} samples")
    print(f"  Classes: {n_classes}, weights: {class_weights.cpu().tolist()}")

    # ── Model ──
    model = get_model(config).to(device)
    n_params = count_parameters(model)
    print(f"\nModel: PointMLP-Elite ({n_params:,} parameters)")

    # VRAM check with a dummy forward pass
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        try:
            dummy = torch.randn(batch_size, data_cfg["num_points"], 3, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(dummy)
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"VRAM estimate: ~{peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
            del dummy
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            if peak_mb > 5120:
                print(f"WARNING: peak > 5GB, consider reducing batch_size")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                new_bs = max(4, batch_size // 2)
                print(f"OOM with batch_size={batch_size}, reducing to {new_bs}")
                batch_size = new_bs
                loaders = get_dataloaders(
                    processed_dir,
                    batch_size=batch_size,
                    num_workers=train_cfg["num_workers"],
                    pin_memory=train_cfg["pin_memory"],
                    n_points=data_cfg["num_points"],
                )
                class_weights = loaders["class_weights"].to(device)
            else:
                raise

    # ── Optimizer, Scheduler, Loss ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=epochs,
    )
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    # Early stopping
    es_cfg = train_cfg["early_stopping"]
    early_stopping = EarlyStopping(
        patience=es_cfg["patience"],
        mode=es_cfg["mode"],
    ) if es_cfg["enabled"] else None

    # ── Logging setup ──
    log_csv_path = exp_dir / "training_log.csv"
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # Save config copy
    with open(exp_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # TensorBoard
    tb_writer = None
    if exp_cfg.get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = exp_dir / "tensorboard"
            tb_writer = SummaryWriter(str(tb_dir))
        except ImportError:
            print("TensorBoard not available, skipping.")

    # ── Print summary ──
    print(f"\n{'='*65}")
    print(f" PointMLP Tree Classifier")
    print(f"{'='*65}")
    print(f"  Experiment:     {exp_name}")
    print(f"  Dataset:        {dataset_type} ({processed_dir})")
    print(f"  Classes:        {n_classes}")
    print(f"  Batch size:     {batch_size} | Mixed precision: {'ON' if use_amp else 'OFF'}")
    print(f"  Epochs:         {epochs} | LR: {lr} | Weight decay: {train_cfg['weight_decay']}")
    print(f"  Early stopping: patience={es_cfg['patience']} on {es_cfg['metric']}")
    print(f"  Output:         {exp_dir}")
    print(f"{'='*65}\n")

    # ── Graceful Ctrl+C ──
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n  Ctrl+C detected, finishing current epoch and saving...")

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    # ── Training loop ──
    best_val_acc = 0.0
    best_epoch = 0

    try:
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            use_mixup = train_cfg.get("mixup", True)
            mixup_alpha = train_cfg.get("mixup_alpha", 0.4)
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, criterion,
                device, scaler,
                grad_clip=train_cfg["gradient_clip"],
                use_amp=use_amp,
                use_mixup=use_mixup,
                n_classes=n_classes,
                mixup_alpha=mixup_alpha,
            )

            # Validate
            val_metrics = validate(
                model, loaders["val"], criterion, device, use_amp=use_amp
            )

            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            # Log to CSV
            with open(log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_metrics['loss']:.4f}",
                    f"{train_metrics['accuracy']:.2f}",
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['accuracy']:.2f}",
                    f"{current_lr:.2e}",
                ])

            # Log to TensorBoard
            if tb_writer:
                tb_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                tb_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                tb_writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                tb_writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
                tb_writer.add_scalar("LR", current_lr, epoch)

            # Print epoch summary
            star = ""
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                star = " *"
                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": val_metrics["loss"],
                    "config": config,
                }, exp_dir / "best_model.pth")

            print(
                f"Epoch [{epoch:3d}/{epochs}] | "
                f"Train Loss: {train_metrics['loss']:.3f} Acc: {train_metrics['accuracy']:5.1f}% | "
                f"Val Loss: {val_metrics['loss']:.3f} Acc: {val_metrics['accuracy']:5.1f}% | "
                f"LR: {current_lr:.1e} | "
                f"{elapsed:.1f}s{star}"
            )

            if star:
                print(f"  >> New best model saved (val_acc: {best_val_acc:.1f}%)")

            # Save last model periodically
            if epoch % exp_cfg.get("save_every", 10) == 0 or epoch == epochs:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_metrics["accuracy"],
                    "config": config,
                }, exp_dir / "last_model.pth")

            # Early stopping
            if early_stopping is not None:
                es_value = (
                    val_metrics["accuracy"] if es_cfg["metric"] == "val_acc"
                    else val_metrics["loss"]
                )
                if early_stopping.step(es_value, epoch):
                    print(f"\nEarly stopping at epoch {epoch} (patience: {es_cfg['patience']})")
                    break

            if interrupted:
                print("\nTraining interrupted by user.")
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 4:
            torch.cuda.empty_cache()
            new_bs = max(4, batch_size // 2)
            print(f"\nOOM error! Reduce batch_size to {new_bs} and rerun:")
            print(f"  python src/train.py --dataset {dataset_type} --exp_name {exp_name} --batch_size {new_bs}")
            sys.exit(1)
        else:
            raise
    finally:
        # Always save last model on exit
        torch.save({
            "epoch": epoch if "epoch" in dir() else 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, exp_dir / "last_model.pth")

        if tb_writer:
            tb_writer.close()

        signal.signal(signal.SIGINT, original_handler)

    # ── Final summary ──
    print(f"\n{'='*65}")
    print(f" Training Complete")
    print(f"{'='*65}")
    print(f"  Best val accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")
    print(f"  Model saved:       {exp_dir / 'best_model.pth'}")
    print(f"  Training log:      {log_csv_path}")
    if tb_writer:
        print(f"  TensorBoard:       tensorboard --logdir {exp_dir / 'tensorboard'}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PointMLP tree classifier")
    parser.add_argument(
        "--dataset", type=str, choices=["real", "synthetic", "real_5cls", "real_3cls"], required=True,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--exp_name", type=str, required=True,
        help="Experiment name (e.g., model_A, model_B)",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(
        config=config,
        dataset_type=args.dataset,
        exp_name=args.exp_name,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
        lr_override=args.lr,
    )


if __name__ == "__main__":
    main()
