"""
Run a single experiment on the MLBS crop and save results for comparison.

Usage:
    python run_experiment.py --name exp01_baseline
    python run_experiment.py --name exp02_smaller_window --clustering.local_max_window 3
    python run_experiment.py --name exp03_fine_chm --clustering.chm_resolution 0.25

All results land in results/experiments/<name>/
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml


LAZ_PATH = "C:/Users/cantu/Downloads/NeonTreeEvaluation/training/LiDAR/2018_MLBS_3_541000_4140000_image_crop.laz"
GT_XML   = "C:/Users/cantu/Downloads/NeonTreeEvaluation/annotations/annotations/2018_MLBS_3_541000_4140000_image_crop.xml"
SEG_CFG  = "configs/segmentation.yaml"
SCENE    = "2018_MLBS_3_541000_4140000_image_crop"


def count_gt_trees(xml_path: str) -> int:
    tree = ET.parse(xml_path)
    return sum(1 for _ in tree.findall(".//object"))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides: 'clustering.smooth_sigma=2.0'"""
    for override in overrides:
        key_path, _, value_str = override.partition("=")
        parts = key_path.strip().split(".")
        # Navigate to parent
        obj = cfg
        for part in parts[:-1]:
            obj = obj[part]
        # Try to keep numeric types
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
        obj[parts[-1]] = value
        print(f"  Override: {key_path} = {value}")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name (e.g. exp01_baseline)")
    parser.add_argument("--config", default=SEG_CFG)
    parser.add_argument("--seg-checkpoint", default=None)
    parser.add_argument("--notes", default="", help="Optional notes for this experiment")
    # Dot-notation overrides for any config key
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="Config overrides: clustering.smooth_sigma=2.0")
    args = parser.parse_args()

    out_dir = Path("results/experiments") / args.name
    if out_dir.exists():
        print(f"ERROR: Experiment '{args.name}' already exists at {out_dir}")
        print("  Use a different name or delete the folder first.")
        sys.exit(1)
    out_dir.mkdir(parents=True)

    # --- Load and patch config ---
    cfg = load_config(args.config)
    if args.overrides:
        print("Applying overrides:")
        cfg = apply_overrides(cfg, args.overrides)

    # Save patched config
    cfg_save = out_dir / "config.yaml"
    with open(cfg_save, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # --- Ground truth ---
    gt_count = count_gt_trees(GT_XML)
    print(f"\nGround truth trees: {gt_count}")

    # --- Load scene ---
    import laspy
    print(f"\nLoading {LAZ_PATH} ...")
    las = laspy.read(LAZ_PATH)
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    classification = np.array(las.classification, dtype=np.int32)
    print(f"  Points: {len(xyz):,}")

    # --- Segmentation ---
    from src.segmentation.pointnet2_model import build_model as build_seg_model
    from src.segmentation.pipeline import segment_scene, cluster_trees

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    seg_log_dir = Path(cfg["experiment"]["log_dir"])
    seg_ckpt = args.seg_checkpoint or str(seg_log_dir / "best_model.pth")

    print(f"\nSegmentation (model: {seg_ckpt}) ...")
    seg_model = build_seg_model(cfg).to(device)
    ckpt = torch.load(seg_ckpt, map_location=device, weights_only=False)
    seg_model.load_state_dict(ckpt["model_state_dict"])

    t0 = time.time()
    predictions = segment_scene(xyz, classification, seg_model, device, cfg)
    seg_time = time.time() - t0
    tree_mask = predictions == 1
    n_tree_pts = int(tree_mask.sum())
    print(f"  Tree points: {n_tree_pts:,} / {len(xyz):,}  ({n_tree_pts/len(xyz):.1%})")
    print(f"  Segmentation time: {seg_time:.1f}s")

    # --- Clustering ---
    print(f"\nClustering (method: {cfg['clustering']['method']}) ...")
    t0 = time.time()
    instance_labels = cluster_trees(xyz, tree_mask, cfg, classification)
    cluster_time = time.time() - t0

    n_detected = int((instance_labels >= 0).any())  # at least 1 instance
    unique_ids = set(instance_labels.tolist()) - {-1}
    n_detected = len(unique_ids)
    n_noise_pts = int((instance_labels == -1).sum())

    print(f"  Detected trees: {n_detected}")
    print(f"  Noise points: {n_noise_pts:,}")
    print(f"  Clustering time: {cluster_time:.1f}s")

    # --- Results ---
    results = {
        "experiment": args.name,
        "notes": args.notes,
        "scene": SCENE,
        "ground_truth_trees": gt_count,
        "detected_trees": n_detected,
        "difference": n_detected - gt_count,
        "detection_ratio": round(n_detected / gt_count, 4) if gt_count > 0 else None,
        "segmentation": {
            "total_points": int(len(xyz)),
            "tree_points": n_tree_pts,
            "tree_ratio": round(n_tree_pts / len(xyz), 4),
            "time_seconds": round(seg_time, 1),
            "checkpoint": seg_ckpt,
        },
        "clustering": {
            "noise_points": n_noise_pts,
            "time_seconds": round(cluster_time, 1),
            "params": cfg["clustering"],
        },
        "config_overrides": args.overrides,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*55}")
    print(f"  EXPERIMENT: {args.name}")
    print(f"{'='*55}")
    print(f"  Ground truth trees : {gt_count}")
    print(f"  Detected trees     : {n_detected}")
    print(f"  Difference         : {n_detected - gt_count:+d}  "
          f"({'over' if n_detected > gt_count else 'under'}-detected)")
    print(f"  Detection ratio    : {n_detected/gt_count:.3f}x")
    print(f"{'='*55}")
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    main()
