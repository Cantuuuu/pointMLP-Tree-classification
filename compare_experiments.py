"""
Compare all saved experiments in results/experiments/.

Usage:
    python compare_experiments.py
    python compare_experiments.py --sort detected_trees
    python compare_experiments.py --sort difference
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sort", default="experiment",
                        choices=["experiment", "detected_trees", "difference", "detection_ratio"],
                        help="Sort column")
    args = parser.parse_args()

    exp_dir = Path("results/experiments")
    if not exp_dir.exists():
        print("No experiments found (results/experiments/ does not exist).")
        return

    experiments = []
    for results_file in sorted(exp_dir.glob("*/results.json")):
        with open(results_file) as f:
            experiments.append(json.load(f))

    if not experiments:
        print("No experiment results found.")
        return

    # Sort
    experiments.sort(key=lambda x: x.get(args.sort, x["experiment"]))

    gt = experiments[0]["ground_truth_trees"]

    # Header
    print(f"\n{'='*90}")
    print(f"  EXPERIMENT COMPARISON  |  Ground truth: {gt} trees (MLBS crop)")
    print(f"{'='*90}")
    header = (f"{'Experiment':<30} {'Detected':>9} {'Diff':>7} {'Ratio':>7} "
              f"{'TreePts':>9} {'Seg(s)':>7} {'Clust(s)':>9}")
    print(header)
    print("-" * 90)

    for exp in experiments:
        detected = exp["detected_trees"]
        diff = detected - gt
        ratio = exp.get("detection_ratio", 0)
        tree_pts = exp["segmentation"]["tree_points"]
        seg_t = exp["segmentation"]["time_seconds"]
        clust_t = exp["clustering"]["time_seconds"]
        sign = "+" if diff >= 0 else ""

        print(f"  {exp['experiment']:<28} {detected:>9} {sign}{diff:>6} {ratio:>7.3f} "
              f"{tree_pts:>9,} {seg_t:>7.1f} {clust_t:>9.1f}")

        if exp.get("notes"):
            print(f"    > {exp['notes']}")
        if exp.get("config_overrides"):
            print(f"    > overrides: {', '.join(exp['config_overrides'])}")

    print(f"{'='*90}")
    print(f"\nClustering params by experiment:")
    for exp in experiments:
        params = exp["clustering"]["params"]
        key_params = (
            f"chm_res={params.get('chm_resolution')}  "
            f"sigma={params.get('smooth_sigma')}  "
            f"min_h={params.get('min_tree_height')}  "
            f"min_px={params.get('min_crown_pixels')}  "
            f"window={params.get('local_max_window')}"
        )
        print(f"  {exp['experiment']:<30} {key_params}")


if __name__ == "__main__":
    main()
