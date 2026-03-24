"""Evaluación de modelos entrenados con Test Time Augmentation (TTA).

Carga un checkpoint, evalúa en el test set con TTA, genera confusion matrix,
calcula métricas por clase (precision, recall, F1), y guarda resultados.

Uso:
    python src/evaluate.py --exp_name model_B_v3 --dataset real_5cls
    python src/evaluate.py --exp_name model_B_v6 --dataset real_5cls --tta 10
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from src.dataset import TreeDataset, augment_point_cloud_light
    from src.model import get_model, count_parameters
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.dataset import TreeDataset, augment_point_cloud_light
    from src.model import get_model, count_parameters


# ── TTA inference ────────────────────────────────────────────────────


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
    n_augmentations: int = 10,
    center: bool = True,
) -> np.ndarray:
    """Predict with Test Time Augmentation.

    For each sample, run inference on the original + N augmented versions,
    average the softmax probabilities, then take argmax.
    """
    model.eval()
    N = points.shape[0]
    all_preds = []

    for i in range(N):
        sample = points[i]  # (n_points, 3)

        if center:
            centroid = sample.mean(axis=0, keepdims=True)
            sample = sample - centroid

        # Original (no augmentation)
        versions = [sample.copy()]

        # Augmented versions
        for _ in range(n_augmentations):
            versions.append(augment_point_cloud_light(sample.copy()))

        # Stack and predict
        batch = np.stack(versions, axis=0)  # (1+n_aug, n_points, 3)
        batch_t = torch.from_numpy(batch).float().to(device)

        logits = model(batch_t)  # (1+n_aug, n_classes)
        probs = F.softmax(logits, dim=-1)

        # Average probabilities
        avg_probs = probs.mean(dim=0)  # (n_classes,)
        pred = avg_probs.argmax().item()
        all_preds.append(pred)

    return np.array(all_preds)


@torch.no_grad()
def predict_no_tta(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
    center: bool = True,
) -> np.ndarray:
    """Predict without TTA (single pass)."""
    model.eval()
    N = points.shape[0]

    pts = points.copy()
    if center:
        centroids = pts.mean(axis=1, keepdims=True)
        pts = pts - centroids

    all_preds = []
    bs = 32
    for start in range(0, N, bs):
        batch = pts[start:start+bs]
        batch_t = torch.from_numpy(batch).float().to(device)
        logits = model(batch_t)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)

    return np.concatenate(all_preds)


# ── Metrics ──────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """Compute accuracy, precision, recall, F1 per class and overall."""
    n_classes = len(class_names)
    results = {"overall": {}, "per_class": {}}

    # Overall accuracy
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    results["overall"]["accuracy"] = correct / total
    results["overall"]["correct"] = int(correct)
    results["overall"]["total"] = int(total)

    # Per-class metrics
    for i, name in enumerate(class_names):
        tp = ((y_pred == i) & (y_true == i)).sum()
        fp = ((y_pred == i) & (y_true != i)).sum()
        fn = ((y_pred != i) & (y_true == i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int((y_true == i).sum())

        results["per_class"][name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    # Macro averages
    precisions = [v["precision"] for v in results["per_class"].values()]
    recalls = [v["recall"] for v in results["per_class"].values()]
    f1s = [v["f1"] for v in results["per_class"].values()]
    results["overall"]["macro_precision"] = sum(precisions) / n_classes
    results["overall"]["macro_recall"] = sum(recalls) / n_classes
    results["overall"]["macro_f1"] = sum(f1s) / n_classes

    return results


def print_metrics(results: dict, class_names: list[str]) -> None:
    """Pretty-print evaluation metrics."""
    overall = results["overall"]
    per_class = results["per_class"]

    print(f"\n{'='*65}")
    print(f" Evaluation Results")
    print(f"{'='*65}")
    print(f"  Overall Accuracy: {overall['accuracy']*100:.1f}% ({overall['correct']}/{overall['total']})")
    print(f"  Macro F1:         {overall['macro_f1']*100:.1f}%")
    print(f"  Macro Precision:  {overall['macro_precision']*100:.1f}%")
    print(f"  Macro Recall:     {overall['macro_recall']*100:.1f}%")
    print()
    print(f"  {'Class':<10} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*38}")
    for name in class_names:
        m = per_class[name]
        print(f"  {name:<10} {m['precision']*100:5.1f}% {m['recall']*100:5.1f}% {m['f1']*100:5.1f}% {m['support']:>7}")
    print(f"{'='*65}\n")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    """Print text confusion matrix."""
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "  " + " " * 8 + "".join(f"{n:>7}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"  {name:<8}" + "".join(f"{cm[i][j]:>7}" for j in range(n_classes))
        print(row)
    print()


# ── Main ─────────────────────────────────────────────────────────────


def evaluate(
    config: dict,
    exp_name: str,
    dataset_type: str = "real_5cls",
    n_tta: int = 10,
    output_name: str | None = None,
    test_data: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Full evaluation pipeline.

    Args:
        output_name: Custom output filename (without .json). If None, uses
            'eval_results' for backwards compat. Useful for cross-evaluation
            (e.g., 'eval_synthetic', 'eval_real').
        test_data: Path to test directory containing points.npy and labels.npy.
            If None, uses processed_dir/dataset_type/test.
        output_dir: Directory to save results. If None, saves to exp_dir.
    """
    exp_dir = Path(config["experiment"]["log_dir"]) / exp_name
    checkpoint_path = exp_dir / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint (it contains its own config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)

    # Build model from checkpoint config
    model = get_model(ckpt_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val_acc: {ckpt.get('val_acc', '?'):.1f}%)")
    print(f"Model: {count_parameters(model):,} parameters")

    # Load test data
    if test_data:
        test_dir = Path(test_data)
        # Find species_map.json in parent directory
        species_map_path = test_dir.parent / "species_map.json"
    else:
        processed_dir = Path(config["data"]["processed_dir"]) / dataset_type
        test_dir = processed_dir / "test"
        species_map_path = processed_dir / "species_map.json"

    points = np.load(test_dir / "points.npy")
    labels = np.load(test_dir / "labels.npy")
    print(f"Test set: {len(labels)} samples from {test_dir}")

    # Load class names
    with open(species_map_path) as f:
        species_map = json.load(f)
    class_names = [k for k, v in sorted(species_map.items(), key=lambda x: x[1]["index"])]

    # Check centering flag from CLI config (not checkpoint config)
    use_centering = config.get("_centering", True)

    # Evaluate without TTA
    print(f"\n--- Without TTA (centering={use_centering}) ---")
    preds_no_tta = predict_no_tta(model, points, device, center=use_centering)
    results_no_tta = compute_metrics(labels, preds_no_tta, class_names)
    print_metrics(results_no_tta, class_names)
    print_confusion_matrix(labels, preds_no_tta, class_names)

    # Evaluate with TTA
    print(f"--- With TTA (x{n_tta} augmentations, centering={use_centering}) ---")
    preds_tta = predict_with_tta(model, points, device, n_augmentations=n_tta, center=use_centering)
    results_tta = compute_metrics(labels, preds_tta, class_names)
    print_metrics(results_tta, class_names)
    print_confusion_matrix(labels, preds_tta, class_names)

    # Build confusion matrices for downstream use
    n_classes = len(class_names)
    cm_no_tta = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds_no_tta):
        cm_no_tta[t][p] += 1
    cm_tta = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds_tta):
        cm_tta[t][p] += 1

    # Save results
    output = {
        "experiment": exp_name,
        "dataset": dataset_type,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "no_tta": results_no_tta,
        "tta": results_tta,
        "tta_augmentations": n_tta,
        "confusion_matrix_no_tta": cm_no_tta.tolist(),
        "confusion_matrix_tta": cm_tta.tolist(),
        "class_names": class_names,
    }

    # Determine output path
    if output_dir:
        save_dir = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{output_name}.json" if output_name else "eval_results.json"
        output_path = save_dir / fname
    else:
        fname = f"{output_name}.json" if output_name else "eval_results.json"
        output_path = exp_dir / fname

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model with TTA")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="real_5cls")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--tta", type=int, default=10, help="Number of TTA augmentations (0=disable)")
    parser.add_argument("--no-center", action="store_true", help="Disable centering (for models trained without it)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (without .json), e.g. 'eval_synthetic'")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test directory (overrides --dataset for data loading)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (overrides default exp_dir)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Pass centering flag via config
    if args.no_center:
        config["_centering"] = False

    evaluate(config, args.exp_name, args.dataset, n_tta=args.tta,
             output_name=args.output, test_data=args.test_data,
             output_dir=args.output_dir)


if __name__ == "__main__":
    main()
