"""
Watershed Parameter Grid Search — CLI

Examples:
    # Quick exploration (27 combos, first 5 tiles)
    python watershed_search.py --mode quick --n-tiles 5

    # Full search (300+ combos, all tiles)
    python watershed_search.py --mode full

    # Custom parameter grid
    python watershed_search.py --mode custom \\
        --chm-res 0.3 0.5 \\
        --sigma 1.0 1.5 2.0 \\
        --window 5 7 \\
        --min-height 3.0 \\
        --min-pixels 4 8

    # Specify LiDAR directory explicitly
    python watershed_search.py --mode quick --laz-dir path/to/laz/files
"""

import argparse
import sys
import io
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str = "configs/segmentation.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_laz_files(laz_dir: Path, limit: int = None) -> list:
    files = sorted(laz_dir.glob("*.laz"))
    if limit:
        files = files[:limit]
    return files


def main():
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Watershed parameter grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["quick", "full", "custom"], default="quick",
                        help="quick=27 combos, full=300+ combos, custom=specify params")
    parser.add_argument("--n-tiles", type=int, default=None,
                        help="Max number of LiDAR tiles to use (default: all)")
    parser.add_argument("--laz-dir", type=str, default=None,
                        help="Directory with .laz files (default: from config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/watershed_grid_search/<timestamp>)")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml")

    # Custom grid parameters
    parser.add_argument("--chm-res", nargs="+", type=float, default=[0.3, 0.5, 0.75])
    parser.add_argument("--sigma", nargs="+", type=float, default=[0.5, 1.5, 2.5])
    parser.add_argument("--window", nargs="+", type=int, default=[3, 5, 9])
    parser.add_argument("--min-height", nargs="+", type=float, default=[3.0])
    parser.add_argument("--min-pixels", nargs="+", type=int, default=[4])

    args = parser.parse_args()

    # --- Imports ---
    from src.segmentation.watershed_grid_search import (
        PARAM_GRIDS, run_grid_search, summarize_results,
        save_csv, generate_html_report,
    )

    # --- Config ---
    cfg = load_config(args.config)
    neon_dir = Path(cfg["data"]["neon_dir"])

    # --- LiDAR files ---
    if args.laz_dir:
        laz_dir = Path(args.laz_dir)
    else:
        laz_dir = neon_dir / "evaluation" / "evaluation" / "LiDAR"

    if not laz_dir.exists():
        print(f"ERROR: LiDAR directory not found: {laz_dir}")
        print("Use --laz-dir to specify the directory with .laz files.")
        sys.exit(1)

    laz_files = find_laz_files(laz_dir, limit=args.n_tiles)
    if not laz_files:
        print(f"ERROR: No .laz files found in {laz_dir}")
        sys.exit(1)

    print(f"Found {len(laz_files)} .laz files in {laz_dir}")
    for f in laz_files:
        print(f"  {f.name}")

    # --- Parameter grid ---
    if args.mode == "custom":
        param_grid = {
            "chm_resolution":   args.chm_res,
            "smooth_sigma":     args.sigma,
            "local_max_window": args.window,
            "min_tree_height":  args.min_height,
            "min_crown_pixels": args.min_pixels,
        }
    else:
        param_grid = PARAM_GRIDS[args.mode]

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    total = len(laz_files) * n_combos
    print(f"\nParameter grid ({args.mode}): {n_combos} combinations")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    print(f"Total runs: {total}  (tiles × combos = {len(laz_files)} × {n_combos})")

    # --- Output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/watershed_grid_search") / f"{args.mode}_{timestamp}"

    # --- Run ---
    all_results = run_grid_search(
        laz_files=laz_files,
        param_grid=param_grid,
        output_dir=output_dir,
        verbose=True,
    )

    if not all_results:
        print("No results generated.")
        sys.exit(1)

    # --- Summarize & save ---
    summaries = summarize_results(all_results)

    save_csv(all_results, output_dir / "all_results.csv")
    save_csv(summaries,   output_dir / "summary.csv")

    print(f"\nSaved CSVs:")
    print(f"  {output_dir / 'all_results.csv'}  (raw, one row per tile×params)")
    print(f"  {output_dir / 'summary.csv'}       (aggregated over tiles, ranked)")

    # --- Report ---
    print("\nGenerating HTML report...")
    generate_html_report(all_results, summaries, param_grid, output_dir)

    # --- Print top 5 ---
    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*60}")
    headers = ["chm_resolution", "smooth_sigma", "local_max_window",
               "min_tree_height", "mean_quality_score",
               "mean_density_ha", "mean_median_crown_m2", "mean_noise_frac"]
    row_fmt = "{:<6} {:<6} {:<7} {:<7} {:<8} {:<10} {:<13} {:<10}"
    print(row_fmt.format(*[h.replace("mean_","") for h in headers]))
    print("-" * 70)
    for s in summaries[:5]:
        print(row_fmt.format(
            s.get("chm_resolution", ""),
            s.get("smooth_sigma", ""),
            s.get("local_max_window", ""),
            s.get("min_tree_height", ""),
            f"{s.get('mean_quality_score', 0):.3f}",
            f"{s.get('mean_density_ha', 0):.0f}",
            f"{s.get('mean_median_crown_m2', 0):.1f}",
            f"{s.get('mean_noise_frac', 0):.3f}",
        ))

    best = summaries[0]
    print(f"\nBest config for your clustering: section in segmentation.yaml:")
    print(f"  clustering:")
    print(f"    method: watershed")
    print(f"    chm_resolution:   {best.get('chm_resolution')}")
    print(f"    smooth_sigma:     {best.get('smooth_sigma')}")
    print(f"    local_max_window: {best.get('local_max_window')}")
    print(f"    min_tree_height:  {best.get('min_tree_height', 3.0)}")
    print(f"    min_crown_pixels: {best.get('min_crown_pixels', 4)}")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
