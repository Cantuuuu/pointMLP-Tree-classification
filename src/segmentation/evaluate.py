"""
Evaluate segmentation model and visualize results.
"""

import sys

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.segmentation.pointnet2_model import build_model
from src.segmentation.dataset import build_dataloaders
from src.segmentation.train import SegmentationLoss, compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Full evaluation with per-block predictions."""
    model.eval()
    all_preds = []
    all_targets = []

    for points, labels in loader:
        points = points.to(device)
        logits = model(points)  # (B, N, 2)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)     # (N_blocks, num_points)
    all_targets = np.concatenate(all_targets, axis=0)  # (N_blocks, num_points)

    return all_preds, all_targets


def compute_full_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute comprehensive metrics."""
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    metrics = {}
    metrics["accuracy"] = float((preds_flat == targets_flat).mean())

    for cls, name in [(0, "non_tree"), (1, "tree")]:
        tp = int(((preds_flat == cls) & (targets_flat == cls)).sum())
        fp = int(((preds_flat == cls) & (targets_flat != cls)).sum())
        fn = int(((preds_flat != cls) & (targets_flat == cls)).sum())
        tn = int(((preds_flat != cls) & (targets_flat != cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        metrics[f"{name}_tp"] = tp
        metrics[f"{name}_fp"] = fp
        metrics[f"{name}_fn"] = fn
        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1
        metrics[f"{name}_iou"] = iou

    metrics["mean_iou"] = (metrics["non_tree_iou"] + metrics["tree_iou"]) / 2

    # Confusion matrix
    metrics["confusion_matrix"] = {
        "predicted_non_tree_actual_non_tree": int(((preds_flat == 0) & (targets_flat == 0)).sum()),
        "predicted_tree_actual_non_tree": int(((preds_flat == 1) & (targets_flat == 0)).sum()),
        "predicted_non_tree_actual_tree": int(((preds_flat == 0) & (targets_flat == 1)).sum()),
        "predicted_tree_actual_tree": int(((preds_flat == 1) & (targets_flat == 1)).sum()),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (default: best_model.pth)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["hardware"]["device"]
                          if torch.cuda.is_available() else "cpu")

    log_dir = Path(cfg["experiment"]["log_dir"])
    checkpoint_path = args.checkpoint or str(log_dir / "best_model.pth")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Load data
    print("Loading data...")
    loaders = build_dataloaders(cfg)

    if args.split not in loaders:
        print(f"ERROR: {args.split} split not found!")
        return

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    preds, targets = evaluate_full(model, loaders[args.split], device)
    metrics = compute_full_metrics(preds, targets)

    # Print results
    print(f"\n{'='*50}")
    print(f"Results on {args.split} set ({len(preds)} blocks)")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU:         {metrics['mean_iou']:.4f}")
    print()
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}")
    print("-" * 52)
    for cls in ["non_tree", "tree"]:
        print(f"{cls:<12} {metrics[f'{cls}_precision']:>10.4f} "
              f"{metrics[f'{cls}_recall']:>10.4f} "
              f"{metrics[f'{cls}_f1']:>10.4f} "
              f"{metrics[f'{cls}_iou']:>10.4f}")
    print()
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  Pred\\True   Non-tree     Tree")
    print(f"  Non-tree    {cm['predicted_non_tree_actual_non_tree']:>8}  {cm['predicted_non_tree_actual_tree']:>8}")
    print(f"  Tree        {cm['predicted_tree_actual_non_tree']:>8}  {cm['predicted_tree_actual_tree']:>8}")

    # Save
    out_path = log_dir / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
