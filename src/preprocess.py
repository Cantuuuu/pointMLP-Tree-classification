"""Pipeline de preprocesamiento: .laz/.las -> .npy

Soporta dos formatos de datos reales:
  - IDTReeS: tiles .las + shapefiles ITC + CSVs (Field/)
  - Simple: un .laz/.las por arbol + labels.csv

Soporta dos formatos de datos sinteticos:
  - Individual: un .laz por arbol con etiqueta en classification/label/species_id
  - Unico: un solo .laz grande con tree_id en user_data y especie en classification

Uso:
    python src/preprocess.py --dataset real
    python src/preprocess.py --dataset synthetic
    python src/preprocess.py --dataset real --n_points 1024 --min_samples 20
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── UTF-8 output on Windows ──────────────────────────────────────────

if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

ASPRS_GROUND = 2  # ASPRS classification code for ground points
TREE_ID_COLUMNS = ["indvdID", "tree_id", "individualID", "treeID", "filename", "id"]
SPECIES_COLUMNS = ["taxonID", "species", "species_id", "label", "class", "genus"]
LAZ_LABEL_FIELDS = ["species_id", "classification", "label"]

# Quality filters
MIN_POINTS_AFTER_FILTER = 50
MIN_BBOX_SIZE = 1.0       # meters
MIN_HEIGHT_RANGE = 1.0    # meters


# ── Config ────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data discovery ────────────────────────────────────────────────────


def discover_data_structure(raw_dir: Path) -> dict[str, Any]:
    """Analyze the data structure in raw_dir and return a description dict.

    Returns dict with:
        format: 'idtrees' | 'individual_laz' | 'single_laz'
        las_files: list of Paths to .las/.laz files
        csv_files: list of Paths to CSV files
        shp_files: list of Paths to shapefiles
    """
    las_files: list[Path] = []
    for ext in ("*.las", "*.laz", "*.LAS", "*.LAZ"):
        las_files.extend(raw_dir.rglob(ext))
    las_files = sorted(las_files)

    csv_files = sorted(raw_dir.rglob("*.csv"))
    shp_files = sorted(raw_dir.rglob("*.shp"))

    # Detect IDTReeS: has LAS/ + ITC/ + Field/ sub-directories
    has_las_dir = (raw_dir / "LAS").is_dir()
    has_itc_dir = (raw_dir / "ITC").is_dir()
    has_field_dir = (raw_dir / "Field").is_dir()

    if has_las_dir and has_itc_dir and has_field_dir:
        fmt = "idtrees"
    elif len(las_files) == 1:
        fmt = "single_laz"
    else:
        fmt = "individual_laz"

    info = {
        "format": fmt,
        "las_files": las_files,
        "csv_files": csv_files,
        "shp_files": shp_files,
    }
    log.info("Detected format: %s (%d .las files, %d CSVs, %d shapefiles)",
             fmt, len(las_files), len(csv_files), len(shp_files))
    return info


# ── Species mapping ───────────────────────────────────────────────────


def load_species_mapping(
    csv_dir: Path,
    min_samples: int = 20,
    group_by_genus: bool = False,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Load IDTReeS CSVs and build species mappings.

    Returns:
        id_to_species: {indvdID: taxonID}
        species_to_index: {taxonID: int} (filtered by min_samples)
        taxon_to_name: {taxonID: scientific_name}
    """
    # Find train_data.csv
    train_csv = None
    for csv_path in sorted(csv_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        cols_lower = {c.lower(): c for c in df.columns}
        if "indvdid" in cols_lower and "taxonid" in cols_lower:
            train_csv = csv_path
            break

    if train_csv is None:
        raise FileNotFoundError(
            f"No CSV with indvdID + taxonID found in {csv_dir}"
        )

    log.info("Using label CSV: %s", train_csv)
    df = pd.read_csv(train_csv, low_memory=False)

    # Build id -> species map
    id_col = [c for c in df.columns if c.lower() == "indvdid"][0]
    sp_col = [c for c in df.columns if c.lower() == "taxonid"][0]
    df[sp_col] = df[sp_col].astype(str).str.strip()
    df[id_col] = df[id_col].astype(str).str.strip()

    if group_by_genus:
        # Map to genus level (first 2-3 uppercase chars before digits)
        df["_genus"] = df[sp_col].str.extract(r"^([A-Z]+)", expand=False)
        df.loc[df["_genus"].isna(), "_genus"] = df.loc[df["_genus"].isna(), sp_col]
        sp_col_use = "_genus"
    else:
        sp_col_use = sp_col

    id_to_species: dict[str, str] = dict(zip(df[id_col], df[sp_col_use]))

    # Count samples per species and filter
    species_counts = pd.Series(list(id_to_species.values())).value_counts()
    viable = species_counts[species_counts >= min_samples].index.tolist()
    species_to_index = {sp: i for i, sp in enumerate(sorted(viable))}

    log.info("Species with >= %d samples: %d / %d total",
             min_samples, len(viable), len(species_counts))
    for sp in sorted(viable):
        log.info("  %s: %d samples", sp, species_counts[sp])

    # Load scientific names
    taxon_to_name: dict[str, str] = {}
    for csv_path in sorted(csv_dir.rglob("*.csv")):
        try:
            df_names = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        cols_lower = {c.lower(): c for c in df_names.columns}
        if "taxonid" in cols_lower and "scientificname" in cols_lower:
            tid = cols_lower["taxonid"]
            sname = cols_lower["scientificname"]
            for _, row in df_names.iterrows():
                taxon_to_name[str(row[tid]).strip()] = str(row[sname]).strip()
            break

    return id_to_species, species_to_index, taxon_to_name


# ── Point cloud operations ────────────────────────────────────────────


def extract_tree_from_tile(
    las_data: laspy.LasData,
    polygon,  # shapely geometry
    remove_ground: bool = True,
) -> np.ndarray:
    """Clip a point cloud from a tile using a polygon boundary.

    Returns (N, 3) array of XYZ points inside the polygon.
    """
    from shapely.geometry import Point

    x = np.array(las_data.x)
    y = np.array(las_data.y)
    z = np.array(las_data.z)

    # Fast bounding-box pre-filter
    minx, miny, maxx, maxy = polygon.bounds
    bbox_mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)

    if not bbox_mask.any():
        return np.empty((0, 3), dtype=np.float64)

    # Fine filter: point-in-polygon for points inside bbox
    x_sub = x[bbox_mask]
    y_sub = y[bbox_mask]
    z_sub = z[bbox_mask]

    # Use prepared geometry for speed
    from shapely.prepared import prep
    prepared = prep(polygon)

    in_poly = np.array([
        prepared.contains(Point(xi, yi))
        for xi, yi in zip(x_sub, y_sub)
    ])

    if not in_poly.any():
        return np.empty((0, 3), dtype=np.float64)

    x_tree = x_sub[in_poly]
    y_tree = y_sub[in_poly]
    z_tree = z_sub[in_poly]

    # Remove ground points if classification available, but keep all if
    # removal would leave too few points
    if remove_ground and hasattr(las_data, "classification"):
        cls_all = np.array(las_data.classification)
        cls_sub = cls_all[bbox_mask][in_poly]
        non_ground = cls_sub != ASPRS_GROUND
        if non_ground.sum() >= MIN_POINTS_AFTER_FILTER:
            x_tree = x_tree[non_ground]
            y_tree = y_tree[non_ground]
            z_tree = z_tree[non_ground]

    return np.column_stack((x_tree, y_tree, z_tree))


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud: center XY at origin, Z relative to min, scale to unit sphere."""
    if len(points) == 0:
        return points

    pts = points.copy().astype(np.float64)

    # Center XY at (0, 0)
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()

    # Z relative to minimum (ground level = 0)
    pts[:, 2] -= pts[:, 2].min()

    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts /= max_dist

    return pts.astype(np.float32)


def farthest_point_sampling(points: np.ndarray, n_points: int) -> np.ndarray:
    """Farthest Point Sampling in pure numpy.

    Iteratively selects the point that is farthest from already selected points.
    Returns (n_points, 3) array.
    """
    n = len(points)
    if n <= n_points:
        return points

    selected = np.zeros(n_points, dtype=np.int64)
    distances = np.full(n, np.inf)

    # Start with a random point
    selected[0] = np.random.randint(0, n)

    for i in range(1, n_points):
        last = selected[i - 1]
        dist_to_last = np.sum((points - points[last]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected[i] = np.argmax(distances)

    return points[selected]


def sample_points(
    points: np.ndarray,
    n_points: int = 1024,
    method: str = "fps",
) -> np.ndarray:
    """Sample exactly n_points from the cloud.

    If N > n_points: use FPS or random subsampling.
    If N < n_points: repeat points with small jitter.
    If N == n_points: return as-is.
    """
    n = len(points)

    if n == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    if n == n_points:
        return points.astype(np.float32)

    if n > n_points:
        if method == "fps":
            # For very large clouds, pre-subsample randomly for efficiency
            if n > 4096:
                idx = np.random.choice(n, 4096, replace=False)
                points = points[idx]
            return farthest_point_sampling(points, n_points).astype(np.float32)
        else:
            idx = np.random.choice(n, n_points, replace=False)
            return points[idx].astype(np.float32)

    # n < n_points: repeat with jitter
    repeats = n_points // n
    remainder = n_points % n
    parts = [points] * repeats
    if remainder > 0:
        idx = np.random.choice(n, remainder, replace=False)
        parts.append(points[idx])
    result = np.concatenate(parts, axis=0)
    # Add small jitter to duplicated points to avoid exact overlaps
    jitter = np.random.normal(0, 1e-4, result.shape).astype(result.dtype)
    # Only jitter the duplicated portion
    jitter[:n] = 0
    result = result + jitter
    return result.astype(np.float32)


# ── Quality checks ────────────────────────────────────────────────────


def quality_check(points: np.ndarray) -> tuple[bool, str]:
    """Check if a tree point cloud passes quality filters.

    Returns (passed, reason).
    """
    if len(points) < MIN_POINTS_AFTER_FILTER:
        return False, "too_few_points"

    if np.isnan(points).any() or np.isinf(points).any():
        return False, "nan_or_inf"

    xrange = points[:, 0].max() - points[:, 0].min()
    yrange = points[:, 1].max() - points[:, 1].min()
    zrange = points[:, 2].max() - points[:, 2].min()

    if xrange < MIN_BBOX_SIZE and yrange < MIN_BBOX_SIZE:
        return False, "bbox_too_small"

    if zrange < MIN_HEIGHT_RANGE:
        return False, "too_short"

    return True, "ok"


# ── IDTReeS processing ───────────────────────────────────────────────


def _build_tile_map(raw_dir: Path) -> dict[str, Path]:
    """Build a mapping from tile stem (e.g. 'OSBS_1') to .las file path."""
    las_dir = raw_dir / "LAS"
    tile_map: dict[str, Path] = {}
    for ext in ("*.las", "*.laz"):
        for f in las_dir.rglob(ext):
            tile_map[f.stem] = f
    return tile_map


def _build_tree_tile_mapping(raw_dir: Path) -> dict[str, str]:
    """Build mapping from indvdID -> tile stem using itc_rsFile.csv."""
    mapping: dict[str, str] = {}
    for csv_path in sorted(raw_dir.rglob("itc_rsFile.csv")):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            indv_id = str(row["indvdID"]).strip()
            rs_file = str(row["rsFile"]).strip()
            # rsFile is like 'MLBS_1.tif', convert to tile stem
            tile_stem = Path(rs_file).stem
            mapping[indv_id] = tile_stem
        break
    return mapping


def process_idtrees_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: dict,
    min_samples: int = 20,
    n_points: int = 1024,
    group_by_genus: bool = False,
) -> dict[str, Any]:
    """Full pipeline for IDTReeS data."""
    log.info("Processing IDTReeS dataset from %s", raw_dir)

    # 1. Load species mapping
    id_to_species, species_to_index, taxon_to_name = load_species_mapping(
        raw_dir, min_samples=min_samples, group_by_genus=group_by_genus
    )

    # 2. Load ITC shapefiles
    all_itc = []
    for shp_path in sorted((raw_dir / "ITC").rglob("*.shp")):
        gdf = gpd.read_file(shp_path)
        all_itc.append(gdf)
    itc = pd.concat(all_itc, ignore_index=True)
    log.info("Loaded %d ITC polygons", len(itc))

    # Detect indvdID column in shapefile
    itc_id_col = None
    cols_lower = {c.lower(): c for c in itc.columns}
    for candidate in TREE_ID_COLUMNS:
        if candidate.lower() in cols_lower:
            itc_id_col = cols_lower[candidate.lower()]
            break
    if itc_id_col is None:
        raise ValueError("No indvdID column found in ITC shapefiles")

    itc[itc_id_col] = itc[itc_id_col].astype(str).str.strip()

    # 3. Build tile mapping
    tile_map = _build_tile_map(raw_dir)
    tree_tile_mapping = _build_tree_tile_mapping(raw_dir)
    log.info("Tile map: %d tiles, tree->tile mapping: %d entries",
             len(tile_map), len(tree_tile_mapping))

    # 4. Group trees by tile for efficient processing
    # (read each tile only once)
    tile_to_trees: dict[str, list[tuple[str, Any]]] = {}  # tile_stem -> [(indvdID, geometry)]

    for _, row in itc.iterrows():
        indv_id = row[itc_id_col]

        # Skip if no species label or species not viable
        if indv_id not in id_to_species:
            continue
        species = id_to_species[indv_id]
        if species not in species_to_index:
            continue

        # Find which tile this tree belongs to
        tile_stem = tree_tile_mapping.get(indv_id)
        if tile_stem is None or tile_stem not in tile_map:
            continue

        if tile_stem not in tile_to_trees:
            tile_to_trees[tile_stem] = []
        tile_to_trees[tile_stem].append((indv_id, row.geometry, species))

    total_trees = sum(len(v) for v in tile_to_trees.values())
    log.info("Trees to extract: %d across %d tiles", total_trees, len(tile_to_trees))

    # 5. Extract trees from tiles
    trees: list[np.ndarray] = []
    labels: list[int] = []
    tree_ids: list[str] = []
    discard_reasons: dict[str, int] = {}

    for tile_stem in tqdm(sorted(tile_to_trees.keys()), desc="Processing tiles"):
        tile_path = tile_map[tile_stem]
        try:
            las_data = laspy.read(str(tile_path))
        except Exception as e:
            log.warning("Failed to read tile %s: %s", tile_path.name, e)
            discard_reasons["unreadable_tile"] = discard_reasons.get("unreadable_tile", 0) + len(tile_to_trees[tile_stem])
            continue

        for indv_id, geometry, species in tile_to_trees[tile_stem]:
            # Extract tree points
            points = extract_tree_from_tile(las_data, geometry, remove_ground=True)

            # Quality check (before normalization, on raw coordinates)
            passed, reason = quality_check(points)
            if not passed:
                discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
                continue

            # Normalize
            points = normalize_point_cloud(points)

            # Sample to fixed size
            points = sample_points(points, n_points=n_points, method="fps")

            trees.append(points)
            labels.append(species_to_index[species])
            tree_ids.append(indv_id)

    log.info("Extracted %d trees, discarded %d", len(trees), sum(discard_reasons.values()))
    for reason, count in sorted(discard_reasons.items(), key=lambda x: -x[1]):
        log.info("  Discarded: %s = %d", reason, count)

    if len(trees) == 0:
        log.error("No trees extracted! Check data paths and quality filters.")
        sys.exit(1)

    # 6. Save
    split_ratios = (
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
    )
    stats = save_processed_data(
        output_dir=output_dir,
        trees=trees,
        labels=labels,
        tree_ids=tree_ids,
        species_to_index=species_to_index,
        taxon_to_name=taxon_to_name,
        split_ratios=split_ratios,
        seed=config["experiment"]["seed"],
    )
    stats["total_found"] = total_trees
    stats["discarded"] = sum(discard_reasons.values())
    stats["discard_reasons"] = discard_reasons
    return stats


# ── Simple individual .laz processing ─────────────────────────────────


def process_individual_laz(
    raw_dir: Path,
    output_dir: Path,
    config: dict,
    min_samples: int = 20,
    n_points: int = 1024,
) -> dict[str, Any]:
    """Process dataset where each .laz/.las is one tree + labels.csv."""
    log.info("Processing individual .laz files from %s", raw_dir)

    # Find label CSV
    id_to_species, species_to_index, taxon_to_name = load_species_mapping(
        raw_dir, min_samples=min_samples
    )

    las_files = []
    for ext in ("*.las", "*.laz"):
        las_files.extend(raw_dir.rglob(ext))
    las_files = sorted(las_files)

    trees: list[np.ndarray] = []
    labels: list[int] = []
    tree_ids: list[str] = []
    discard_reasons: dict[str, int] = {}

    for las_path in tqdm(las_files, desc="Processing trees"):
        stem = las_path.stem
        if stem not in id_to_species:
            continue
        species = id_to_species[stem]
        if species not in species_to_index:
            continue

        try:
            las = laspy.read(str(las_path))
            points = np.column_stack((las.x, las.y, las.z))
        except Exception as e:
            log.warning("Failed to read %s: %s", las_path.name, e)
            discard_reasons["unreadable"] = discard_reasons.get("unreadable", 0) + 1
            continue

        passed, reason = quality_check(points)
        if not passed:
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
            continue

        points = normalize_point_cloud(points)
        points = sample_points(points, n_points=n_points, method="fps")

        trees.append(points)
        labels.append(species_to_index[species])
        tree_ids.append(stem)

    if len(trees) == 0:
        log.error("No trees processed!")
        sys.exit(1)

    split_ratios = (
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
    )
    return save_processed_data(
        output_dir=output_dir,
        trees=trees,
        labels=labels,
        tree_ids=tree_ids,
        species_to_index=species_to_index,
        taxon_to_name=taxon_to_name,
        split_ratios=split_ratios,
        seed=config["experiment"]["seed"],
    )


# ── Synthetic data processing ─────────────────────────────────────────


def _find_csv_label_column(
    directory: Path,
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """Find a CSV with tree_id and species columns."""
    for csv_path in sorted(directory.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        tree_col = species_col = None
        cols_lower = {c.lower(): c for c in df.columns}
        for candidate in TREE_ID_COLUMNS:
            if candidate.lower() in cols_lower:
                tree_col = cols_lower[candidate.lower()]
                break
        for candidate in SPECIES_COLUMNS:
            if candidate.lower() in cols_lower:
                species_col = cols_lower[candidate.lower()]
                break
        if tree_col and species_col:
            return df, tree_col, species_col
    return None, None, None


def process_synthetic_dataset(
    raw_dir: Path,
    output_dir: Path,
    config: dict,
    min_samples: int = 20,
    n_points: int = 1024,
) -> dict[str, Any]:
    """Process synthetic dataset. Auto-detects format."""
    log.info("Processing synthetic dataset from %s", raw_dir)

    las_files = []
    for ext in ("*.las", "*.laz"):
        las_files.extend(raw_dir.rglob(ext))
    las_files = sorted(las_files)

    if not las_files:
        log.error("No .las/.laz files found in %s", raw_dir)
        sys.exit(1)

    # Load species_map.json if it exists (maps numeric codes to class names)
    species_map_file = raw_dir / "species_map.json"
    ext_species_map: Optional[dict[str, str]] = None
    if species_map_file.exists():
        with open(species_map_file, "r") as f:
            ext_species_map = json.load(f)
        log.info("Loaded species_map.json with %d entries", len(ext_species_map))

    # Build forced class indices from config class_names if available
    # This ensures synthetic data uses the SAME indices as the real dataset
    forced_species_to_index: Optional[dict[str, int]] = None
    class_names = config["data"].get("class_names")
    if class_names:
        forced_species_to_index = {name: i for i, name in enumerate(class_names)}
        log.info("Forcing class indices from config: %s", forced_species_to_index)

    # Try CSV labels first
    df_labels, tree_col, species_col = _find_csv_label_column(raw_dir)

    if df_labels is not None:
        log.info("Using CSV labels: %s -> %s", tree_col, species_col)
        return _process_synthetic_with_csv(
            raw_dir, output_dir, config, df_labels, tree_col, species_col,
            las_files, min_samples, n_points,
        )

    # Single large file with embedded labels
    if len(las_files) == 1:
        log.info("Single .laz detected, reading embedded labels")
        return _process_synthetic_single_laz(
            las_files[0], output_dir, config, ext_species_map,
            min_samples, n_points, forced_species_to_index,
        )

    # Multiple files with embedded labels
    log.info("Multiple .laz files, reading embedded labels per file")
    return _process_synthetic_embedded(
        las_files, output_dir, config, ext_species_map,
        min_samples, n_points,
    )


def _process_synthetic_with_csv(
    raw_dir: Path,
    output_dir: Path,
    config: dict,
    df_labels: pd.DataFrame,
    tree_col: str,
    species_col: str,
    las_files: list[Path],
    min_samples: int,
    n_points: int,
) -> dict[str, Any]:
    """Process synthetic data using a CSV for labels."""
    df_labels[tree_col] = df_labels[tree_col].astype(str).str.strip()
    df_labels[species_col] = df_labels[species_col].astype(str).str.strip()

    id_to_species = dict(zip(df_labels[tree_col], df_labels[species_col]))
    species_counts = df_labels[species_col].value_counts()
    viable = species_counts[species_counts >= min_samples].index.tolist()
    species_to_index = {sp: i for i, sp in enumerate(sorted(viable))}

    trees: list[np.ndarray] = []
    labels: list[int] = []
    tree_ids: list[str] = []
    discard_reasons: dict[str, int] = {}

    for las_path in tqdm(las_files, desc="Processing synthetic trees"):
        stem = las_path.stem
        if stem not in id_to_species:
            continue
        species = id_to_species[stem]
        if species not in species_to_index:
            continue

        try:
            las = laspy.read(str(las_path))
            points = np.column_stack((las.x, las.y, las.z))
        except Exception as e:
            log.warning("Failed to read %s: %s", las_path.name, e)
            discard_reasons["unreadable"] = discard_reasons.get("unreadable", 0) + 1
            continue

        passed, reason = quality_check(points)
        if not passed:
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
            continue

        points = normalize_point_cloud(points)
        points = sample_points(points, n_points=n_points, method="fps")
        trees.append(points)
        labels.append(species_to_index[species])
        tree_ids.append(stem)

    if len(trees) == 0:
        log.error("No synthetic trees processed!")
        sys.exit(1)

    split_ratios = (
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
    )
    return save_processed_data(
        output_dir, trees, labels, tree_ids, species_to_index, {},
        split_ratios, config["experiment"]["seed"],
    )


def _detect_simulator_format(las: laspy.LasData) -> bool:
    """Check if this .laz uses the simulator convention:
    point_source_id = tree_id, user_data = species_id.

    Heuristic: user_data has very few unique values (species codes)
    while point_source_id has many (one per tree).
    """
    if not hasattr(las, "user_data") or not hasattr(las, "point_source_id"):
        return False
    ud_unique = len(np.unique(np.array(las.user_data)))
    ps_unique = len(np.unique(np.array(las.point_source_id)))
    return ud_unique <= 10 and ps_unique > ud_unique


def _process_synthetic_single_laz(
    las_path: Path,
    output_dir: Path,
    config: dict,
    ext_species_map: Optional[dict[str, str]],
    min_samples: int,
    n_points: int,
    forced_species_to_index: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Process a single large .laz with embedded tree and species info.

    Supports two conventions:
      - Simulator: point_source_id = tree_id, user_data = species_id
      - Legacy: user_data = tree_id, classification = species_id
    """
    las = laspy.read(str(las_path))
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)

    simulator_fmt = _detect_simulator_format(las)

    if simulator_fmt:
        log.info("Detected simulator format (point_source_id=tree, user_data=species)")
        tree_id_vals = np.array(las.point_source_id)
        species_vals = np.array(las.user_data)
    else:
        log.info("Using legacy format (user_data=tree, classification=species)")
        # Find label field
        label_field = None
        for field in LAZ_LABEL_FIELDS:
            if hasattr(las, field):
                label_field = field
                break
            for dim in las.point_format.extra_dimensions:
                if dim.name.lower() == field.lower():
                    label_field = dim.name
                    break
            if label_field:
                break

        if label_field is None:
            log.error("No label field found in %s", las_path)
            sys.exit(1)

        species_vals = np.array(getattr(las, label_field))

        # Try to find tree_id field
        tree_id_vals = None
        for field in ["user_data", "tree_id", "point_source_id"]:
            if hasattr(las, field):
                tree_id_vals = np.array(getattr(las, field))
                break
            for dim in las.point_format.extra_dimensions:
                if dim.name.lower() == field.lower():
                    tree_id_vals = np.array(las[dim.name])
                    break
            if tree_id_vals is not None:
                break

        if tree_id_vals is None:
            log.error("No tree_id field found in %s", las_path)
            sys.exit(1)

    unique_trees = np.unique(tree_id_vals)
    log.info("Found %d unique tree IDs in single .laz", len(unique_trees))

    # Gather per-tree data
    all_labels: list[str] = []
    all_points: list[np.ndarray] = []
    all_ids: list[str] = []

    for tid in tqdm(unique_trees, desc="Segmenting trees"):
        mask = tree_id_vals == tid
        pts = np.column_stack((x[mask], y[mask], z[mask]))
        sp_vals = species_vals[mask]
        # Dominant species
        uv, uc = np.unique(sp_vals, return_counts=True)
        sp = str(uv[np.argmax(uc)])

        # Skip ground (species_id=0 or tree_id=0)
        if sp == "0" or str(tid) == "0":
            continue

        if ext_species_map and sp in ext_species_map:
            sp = ext_species_map[sp]

        all_points.append(pts)
        all_labels.append(sp)
        all_ids.append(str(tid))

    # Use forced class indices if provided, otherwise auto-detect
    if forced_species_to_index is not None:
        species_to_index = forced_species_to_index
        log.info("Using forced species_to_index: %s", species_to_index)
    else:
        sp_counts = pd.Series(all_labels).value_counts()
        viable = sp_counts[sp_counts >= min_samples].index.tolist()
        species_to_index = {sp: i for i, sp in enumerate(sorted(viable))}

    trees: list[np.ndarray] = []
    labels: list[int] = []
    tree_ids: list[str] = []
    discard_reasons: dict[str, int] = {}

    for pts, sp, tid in zip(all_points, all_labels, all_ids):
        if sp not in species_to_index:
            discard_reasons["rare_species"] = discard_reasons.get("rare_species", 0) + 1
            continue
        passed, reason = quality_check(pts)
        if not passed:
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
            continue
        pts = normalize_point_cloud(pts)
        pts = sample_points(pts, n_points=n_points, method="fps")
        trees.append(pts)
        labels.append(species_to_index[sp])
        tree_ids.append(tid)

    log.info("Processed %d trees, discarded %d", len(trees), sum(discard_reasons.values()))
    for reason, count in sorted(discard_reasons.items(), key=lambda x: -x[1]):
        log.info("  Discarded: %s = %d", reason, count)

    if len(trees) == 0:
        log.error("No synthetic trees processed!")
        sys.exit(1)

    split_ratios = (
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
    )
    return save_processed_data(
        output_dir, trees, labels, tree_ids, species_to_index, {},
        split_ratios, config["experiment"]["seed"],
    )


def _process_synthetic_embedded(
    las_files: list[Path],
    output_dir: Path,
    config: dict,
    ext_species_map: Optional[dict[str, str]],
    min_samples: int,
    n_points: int,
) -> dict[str, Any]:
    """Process multiple .laz files where each has an embedded label field."""
    # First pass: read all labels to find viable species
    file_labels: list[tuple[Path, str]] = []
    for las_path in tqdm(las_files, desc="Reading labels"):
        try:
            las = laspy.read(str(las_path))
        except Exception:
            continue
        label = None
        for field in LAZ_LABEL_FIELDS:
            vals = None
            if hasattr(las, field):
                vals = np.array(getattr(las, field))
            else:
                for dim in las.point_format.extra_dims:
                    if dim.name.lower() == field.lower():
                        vals = np.array(las[dim.name])
                        break
            if vals is not None:
                uv, uc = np.unique(vals, return_counts=True)
                label = str(uv[np.argmax(uc)])
                break
        if label is None:
            continue
        if ext_species_map and label in ext_species_map:
            label = ext_species_map[label]
        file_labels.append((las_path, label))

    sp_counts = pd.Series([l for _, l in file_labels]).value_counts()
    viable = sp_counts[sp_counts >= min_samples].index.tolist()
    species_to_index = {sp: i for i, sp in enumerate(sorted(viable))}

    # Second pass: process viable trees
    trees: list[np.ndarray] = []
    labels: list[int] = []
    tree_ids: list[str] = []
    discard_reasons: dict[str, int] = {}

    for las_path, species in tqdm(file_labels, desc="Processing trees"):
        if species not in species_to_index:
            continue
        try:
            las = laspy.read(str(las_path))
            points = np.column_stack((las.x, las.y, las.z))
        except Exception:
            discard_reasons["unreadable"] = discard_reasons.get("unreadable", 0) + 1
            continue

        passed, reason = quality_check(points)
        if not passed:
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
            continue

        points = normalize_point_cloud(points)
        points = sample_points(points, n_points=n_points, method="fps")
        trees.append(points)
        labels.append(species_to_index[species])
        tree_ids.append(las_path.stem)

    if len(trees) == 0:
        log.error("No synthetic trees processed!")
        sys.exit(1)

    split_ratios = (
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
    )
    return save_processed_data(
        output_dir, trees, labels, tree_ids, species_to_index, {},
        split_ratios, config["experiment"]["seed"],
    )


# ── Save processed data ──────────────────────────────────────────────


def save_processed_data(
    output_dir: Path,
    trees: list[np.ndarray],
    labels: list[int],
    tree_ids: list[str],
    species_to_index: dict[str, int],
    taxon_to_name: dict[str, str],
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, Any]:
    """Save processed data with stratified train/val/test split."""
    all_points = np.stack(trees)     # (N, n_points, 3)
    all_labels = np.array(labels)    # (N,)
    all_ids = np.array(tree_ids)     # (N,)

    train_ratio, val_ratio, test_ratio = split_ratios

    # Stratified split: first split off test, then split remainder into train/val
    test_size = test_ratio
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)

    idx_trainval, idx_test = train_test_split(
        np.arange(len(all_labels)),
        test_size=test_size,
        random_state=seed,
        stratify=all_labels,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=all_labels[idx_trainval],
    )

    splits = {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
    }

    # Save each split
    for split_name, idx in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "points.npy", all_points[idx])
        np.save(split_dir / "labels.npy", all_labels[idx])
        np.save(split_dir / "tree_ids.npy", all_ids[idx])

    # Save species map
    index_to_species = {v: k for k, v in species_to_index.items()}
    species_info = {}
    for taxon, idx in sorted(species_to_index.items(), key=lambda x: x[1]):
        count = int((all_labels == idx).sum())
        species_info[taxon] = {
            "index": idx,
            "count": count,
            "scientific_name": taxon_to_name.get(taxon, ""),
        }
    with open(output_dir / "species_map.json", "w") as f:
        json.dump(species_info, f, indent=2)

    # Save stats
    stats: dict[str, Any] = {
        "total_processed": len(all_labels),
        "n_points": int(all_points.shape[1]),
        "n_classes": len(species_to_index),
        "splits": {},
    }
    for split_name, idx in splits.items():
        split_labels = all_labels[idx]
        class_dist = {}
        for taxon, cls_idx in species_to_index.items():
            class_dist[taxon] = int((split_labels == cls_idx).sum())
        stats["splits"][split_name] = {
            "count": len(idx),
            "class_distribution": class_dist,
        }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    _print_summary(all_points, all_labels, splits, species_to_index,
                   taxon_to_name, output_dir)

    return stats


def _print_summary(
    all_points: np.ndarray,
    all_labels: np.ndarray,
    splits: dict[str, np.ndarray],
    species_to_index: dict[str, int],
    taxon_to_name: dict[str, str],
    output_dir: Path,
) -> None:
    """Print a nice summary of the processed data."""
    n_total = len(all_labels)
    n_points = all_points.shape[1]

    print(f"\n{'='*60}")
    print(f" Preprocesamiento completado")
    print(f"{'='*60}")
    print(f"  Arboles procesados: {n_total}")
    print(f"  Puntos por arbol:   {n_points}")
    print(f"  Clases:             {len(species_to_index)}")

    # Species distribution with bar chart
    print(f"\n  Distribucion por especie:")
    max_count = max((all_labels == idx).sum() for idx in species_to_index.values())
    bar_width = 30

    for taxon, idx in sorted(species_to_index.items(),
                              key=lambda x: -(all_labels == x[1]).sum()):
        count = int((all_labels == idx).sum())
        bar_len = int(count / max_count * bar_width)
        bar = "#" * bar_len
        name = taxon_to_name.get(taxon, "")
        name_str = f" ({name})" if name else ""
        print(f"    {taxon:8s}: {count:4d} {bar}{name_str}")

    # Split info
    print(f"\n  Split:")
    for split_name, idx in splits.items():
        pct = len(idx) / n_total * 100
        print(f"    {split_name:5s}: {len(idx):4d} arboles ({pct:.1f}%)")

    # File info
    print(f"\n  Archivos guardados en: {output_dir}/")
    for split_name, idx in splits.items():
        shape_p = f"({len(idx)}, {n_points}, 3)"
        shape_l = f"({len(idx)},)"
        print(f"    {split_name}/points.npy {shape_p:>20s}   "
              f"{split_name}/labels.npy {shape_l:>10s}")
    print(f"    species_map.json")
    print(f"    stats.json")
    print()


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesar nubes de puntos .laz/.las -> .npy"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["real", "synthetic"],
        required=True,
        help="Que dataset preprocesar",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=None,
        help="Puntos por muestra (override config)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=20,
        help="Minimo de muestras por especie",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help="Proporcion de test (override config)",
    )
    parser.add_argument(
        "--group_by_genus",
        action="store_true",
        help="Agrupar especies a nivel de genero",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobreescribir datos procesados existentes",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    n_points = args.n_points or config["data"]["num_points"]
    raw_dir = Path(config["data"]["raw_dir"]) / args.dataset
    output_dir = Path(config["data"]["processed_dir"]) / args.dataset

    if args.test_ratio is not None:
        config["data"]["test_split"] = args.test_ratio
        remaining = 1.0 - args.test_ratio
        config["data"]["train_split"] = remaining * 0.85
        config["data"]["val_split"] = remaining * 0.15

    # Check if output already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            response = input(
                f"Output directory {output_dir} already has data. "
                f"Overwrite? [y/N]: "
            )
            if response.lower() != "y":
                log.info("Aborted.")
                return
        log.info("Overwriting existing data in %s", output_dir)

    if not raw_dir.exists():
        log.error("Raw data directory does not exist: %s", raw_dir)
        sys.exit(1)

    np.random.seed(config["experiment"]["seed"])

    if args.dataset == "real":
        info = discover_data_structure(raw_dir)
        if info["format"] == "idtrees":
            process_idtrees_dataset(
                raw_dir, output_dir, config,
                min_samples=args.min_samples,
                n_points=n_points,
                group_by_genus=args.group_by_genus,
            )
        else:
            process_individual_laz(
                raw_dir, output_dir, config,
                min_samples=args.min_samples,
                n_points=n_points,
            )
    else:
        process_synthetic_dataset(
            raw_dir, output_dir, config,
            min_samples=args.min_samples,
            n_points=n_points,
        )


if __name__ == "__main__":
    main()
