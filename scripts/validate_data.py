"""Validación de datos crudos antes del preprocesamiento.

Soporta dos formatos de datos reales:
  - IDTReeS nativo: tiles .las + shapefiles ITC + CSVs en Field/
  - Formato simple: un .laz por árbol + labels.csv

Uso:
    python scripts/validate_data.py --dataset real
    python scripts/validate_data.py --dataset synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import io
import os

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import yaml

# Force UTF-8 output on Windows
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Configuración de checks ──────────────────────────────────────────

MIN_CLASSES = 3
MIN_SAMPLES_PER_CLASS = 20
MIN_POINTS_PER_CLOUD = 10

# Nombres de columna aceptados para tree_id y especie en CSVs
# Ordered by priority — more specific names first, generic "id" last
TREE_ID_COLUMNS = ["indvdID", "tree_id", "individualID", "treeID", "filename", "id"]
SPECIES_COLUMNS = ["taxonID", "species", "species_id", "label", "class", "genus", "scientificName"]

# Campos dentro del .laz que pueden contener la etiqueta de especie
LAZ_LABEL_FIELDS = ["species_id", "classification", "label"]


# ── Utilidades ────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_laz_files(directory: Path) -> list[Path]:
    """Busca archivos .laz y .las recursivamente."""
    files = []
    for ext in ("*.laz", "*.las", "*.LAZ", "*.LAS"):
        files.extend(directory.rglob(ext))
    return sorted(files)


def find_csv_files(directory: Path) -> list[Path]:
    """Busca archivos CSV recursivamente."""
    return sorted(directory.rglob("*.csv"))


def find_shapefiles(directory: Path) -> list[Path]:
    """Busca archivos .shp recursivamente."""
    return sorted(directory.rglob("*.shp"))


def find_label_csv(
    directory: Path,
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """Busca un CSV con columnas de tree_id y especie.

    Returns:
        (dataframe, tree_id_col, species_col) o (None, None, None) si no se encuentra.
    """
    csv_files = find_csv_files(directory)
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue

        tree_col = None
        species_col = None

        for col in df.columns:
            col_lower = col.strip().lower()
            if tree_col is None:
                for candidate in TREE_ID_COLUMNS:
                    if col_lower == candidate.lower():
                        tree_col = col
                        break
            if species_col is None:
                for candidate in SPECIES_COLUMNS:
                    if col_lower == candidate.lower():
                        species_col = col
                        break

        if tree_col is not None and species_col is not None:
            return df, tree_col, species_col

    return None, None, None


def read_laz_xyz(file_path: Path) -> Optional[np.ndarray]:
    """Lee un .laz/.las y retorna coordenadas XYZ como (N, 3) array, o None si falla."""
    try:
        las = laspy.read(str(file_path))
        xyz = np.column_stack((las.x, las.y, las.z))
        return xyz
    except Exception:
        return None


def get_laz_label_field(file_path: Path) -> Optional[tuple[str, np.ndarray]]:
    """Intenta leer un campo de etiqueta de especie del .laz."""
    try:
        las = laspy.read(str(file_path))
        for field_name in LAZ_LABEL_FIELDS:
            if hasattr(las, field_name):
                values = np.array(getattr(las, field_name))
                return field_name, values
            for dim in las.point_format.extra_dims:
                if dim.name.lower() == field_name.lower():
                    return dim.name, np.array(las[dim.name])
        return None
    except Exception:
        return None


def load_species_map(directory: Path) -> Optional[dict[str, str]]:
    """Carga species_map.json si existe."""
    map_path = directory / "species_map.json"
    if map_path.exists():
        with open(map_path, "r") as f:
            return json.load(f)
    return None


def detect_idtrees_structure(data_dir: Path) -> bool:
    """Detecta si el directorio tiene estructura IDTReeS (LAS/ + ITC/ + Field/)."""
    has_las_dir = (data_dir / "LAS").is_dir()
    has_itc_dir = (data_dir / "ITC").is_dir()
    has_field_dir = (data_dir / "Field").is_dir()
    return has_las_dir and has_itc_dir and has_field_dir


# ── Clase de validación ──────────────────────────────────────────────


class ValidationResult:
    def __init__(self) -> None:
        self.checks: list[tuple[bool, str]] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        self.checks.append((True, message))

    def fail(self, message: str) -> None:
        self.checks.append((False, message))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks)

    def print_report(self) -> None:
        for ok, msg in self.checks:
            symbol = "\u2713" if ok else "\u2717"
            print(f"  {symbol} {msg}")
        for w in self.warnings:
            print(f"  ! {w}")


# ── Validación IDTReeS ───────────────────────────────────────────────


def validate_real_idtrees(data_dir: Path) -> ValidationResult:
    """Valida datos reales en formato IDTReeS: LAS/ (tiles) + ITC/ (shapefiles) + Field/ (CSVs)."""
    result = ValidationResult()
    print(f"\n{'='*60}")
    print(f" Validando datos REALES (formato IDTReeS): {data_dir}")
    print(f"{'='*60}\n")

    result.ok("Estructura IDTReeS detectada (LAS/ + ITC/ + Field/)")

    # ── 1. Tiles .las ──
    las_dir = data_dir / "LAS"
    laz_files = find_laz_files(las_dir)
    if laz_files:
        result.ok(f"Encontrados {len(laz_files)} tiles .las en LAS/")
    else:
        result.fail("No se encontraron archivos .las en LAS/")
        result.print_report()
        return result

    # ── 2. Shapefiles ITC ──
    itc_dir = data_dir / "ITC"
    shapefiles = find_shapefiles(itc_dir)
    if shapefiles:
        result.ok(f"Encontrados {len(shapefiles)} shapefiles en ITC/")
    else:
        result.fail("No se encontraron shapefiles (.shp) en ITC/")
        result.print_report()
        return result

    # Leer todos los shapefiles y contar polígonos de copa
    all_itc = []
    itc_id_col = None
    for shp_path in shapefiles:
        try:
            gdf = gpd.read_file(shp_path)
            all_itc.append(gdf)
            # Detectar columna de indvdID (priorizar por orden de TREE_ID_COLUMNS)
            if itc_id_col is None:
                gdf_cols_lower = {c.lower(): c for c in gdf.columns}
                for candidate in TREE_ID_COLUMNS:
                    if candidate.lower() in gdf_cols_lower:
                        itc_id_col = gdf_cols_lower[candidate.lower()]
                        break
        except Exception as e:
            result.fail(f"Error leyendo shapefile {shp_path.name}: {e}")

    if all_itc:
        itc_combined = pd.concat(all_itc, ignore_index=True)
        total_crowns = len(itc_combined)
        result.ok(f"{total_crowns} polígonos de copa (árboles individuales) en shapefiles")
    else:
        result.fail("No se pudieron leer los shapefiles")
        result.print_report()
        return result

    if itc_id_col:
        result.ok(f"Columna de ID en shapefiles: '{itc_id_col}'")
    else:
        result.warn(
            "No se encontró columna de indvdID en shapefiles. "
            "Se usará el índice como identificador."
        )

    # ── 3. CSV de etiquetas (train_data.csv) ──
    field_dir = data_dir / "Field"
    df_labels, tree_col, species_col = find_label_csv(field_dir)
    if df_labels is not None:
        result.ok(
            f"CSV de etiquetas encontrado — columnas: '{tree_col}' (id), '{species_col}' (especie)"
        )
    else:
        result.fail(
            "No se encontró CSV con columnas de tree_id y especie en Field/. "
            f"Se esperaba train_data.csv con columnas como {TREE_ID_COLUMNS} y {SPECIES_COLUMNS}"
        )
        result.print_report()
        return result

    # ── 4. CSV de mapeo tree → tile (itc_rsFile.csv) ──
    mapping_csv = field_dir / "itc_rsFile.csv"
    if mapping_csv.exists():
        df_mapping = pd.read_csv(mapping_csv)
        result.ok(
            f"Mapeo tree→tile encontrado (itc_rsFile.csv): {len(df_mapping)} entradas"
        )
    else:
        result.warn(
            "No se encontró itc_rsFile.csv — el preprocesamiento usará "
            "intersección espacial para asignar árboles a tiles."
        )
        df_mapping = None

    # Limpiar etiquetas
    df_labels[species_col] = df_labels[species_col].astype(str).str.strip()
    df_labels = df_labels.dropna(subset=[species_col])
    df_labels = df_labels[df_labels[species_col] != ""]
    df_labels = df_labels[df_labels[species_col].str.lower() != "nan"]

    # ── 5. Cruzar: qué árboles del CSV tienen polígono ITC ──
    if itc_id_col and tree_col:
        itc_ids = set(itc_combined[itc_id_col].astype(str).str.strip())
        label_ids = set(df_labels[tree_col].astype(str).str.strip())
        matched = itc_ids & label_ids
        only_itc = itc_ids - label_ids
        only_labels = label_ids - itc_ids
        result.ok(
            f"{len(matched)} árboles con polígono ITC Y etiqueta de especie (usables)"
        )
        if only_itc:
            result.warn(f"{len(only_itc)} árboles en ITC sin etiqueta en CSV (se ignorarán)")
        if only_labels:
            result.warn(f"{len(only_labels)} árboles en CSV sin polígono ITC (se ignorarán)")

        # Filtrar solo los matcheados para el análisis de clases
        df_usable = df_labels[df_labels[tree_col].astype(str).str.strip().isin(matched)]
    else:
        df_usable = df_labels

    # ── 6. Distribución de especies (solo árboles usables) ──
    species_counts = df_usable[species_col].value_counts()
    num_classes = len(species_counts)
    if num_classes >= MIN_CLASSES:
        result.ok(f"{num_classes} especies en árboles usables (min: {MIN_CLASSES})")
    else:
        result.fail(f"Solo {num_classes} especies en árboles usables (min: {MIN_CLASSES})")

    # Clases con suficientes muestras
    viable_classes = species_counts[species_counts >= MIN_SAMPLES_PER_CLASS]
    small_classes = species_counts[species_counts < MIN_SAMPLES_PER_CLASS]
    if len(viable_classes) >= MIN_CLASSES:
        result.ok(
            f"{len(viable_classes)} especies con >= {MIN_SAMPLES_PER_CLASS} muestras"
        )
    else:
        result.fail(
            f"Solo {len(viable_classes)} especies con >= {MIN_SAMPLES_PER_CLASS} muestras "
            f"(min: {MIN_CLASSES})"
        )

    # ── 7. Tiles legibles con laspy ──
    sample_tiles = laz_files[:10] if len(laz_files) > 10 else laz_files
    readable = 0
    unreadable: list[str] = []
    point_counts: list[int] = []
    xyz_valid = 0
    xyz_invalid: list[str] = []

    for laz_path in sample_tiles:
        xyz = read_laz_xyz(laz_path)
        if xyz is not None:
            readable += 1
            point_counts.append(len(xyz))
            has_nan = np.isnan(xyz).any()
            all_zero = np.allclose(xyz, 0)
            too_few = len(xyz) < MIN_POINTS_PER_CLOUD
            if not has_nan and not all_zero and not too_few:
                xyz_valid += 1
            else:
                reasons = []
                if has_nan:
                    reasons.append("NaN")
                if all_zero:
                    reasons.append("todo ceros")
                if too_few:
                    reasons.append(f"solo {len(xyz)} puntos")
                xyz_invalid.append(f"{laz_path.name} ({', '.join(reasons)})")
        else:
            unreadable.append(laz_path.name)

    tested = len(sample_tiles)
    if readable == tested:
        result.ok(f"Todos los {tested} tiles probados son legibles con laspy")
    else:
        result.fail(
            f"{len(unreadable)}/{tested} tiles no se pudieron leer: "
            f"{unreadable[:5]}{'...' if len(unreadable) > 5 else ''}"
        )

    if xyz_valid == readable:
        result.ok(f"Todos los {readable} tiles tienen XYZ válido")
    else:
        result.fail(
            f"{len(xyz_invalid)} tiles con XYZ inválido: "
            f"{xyz_invalid[:5]}{'...' if len(xyz_invalid) > 5 else ''}"
        )

    # ── Reporte ──
    print("\n  ── Distribución de clases (árboles usables) ──")
    for species, count in species_counts.items():
        viable = ">=" if count >= MIN_SAMPLES_PER_CLASS else "< "
        print(f"    {species:12s}: {count:4d} muestras  ({viable}{MIN_SAMPLES_PER_CLASS})")

    print(f"\n  ── Resumen ──")
    print(f"    Total tiles .las:       {len(laz_files)}")
    print(f"    Total polígonos ITC:    {total_crowns}")
    print(f"    Total en CSV:           {len(df_labels)}")
    print(f"    Árboles usables:        {len(df_usable)}")
    print(f"    Especies viables:       {len(viable_classes)} ({', '.join(viable_classes.index.tolist())})")
    print(f"    Árboles en clases viables: {viable_classes.sum()}")

    if point_counts:
        print(f"\n  ── Puntos por tile (muestra de {tested}) ──")
        print(f"    Min: {min(point_counts):,}")
        print(f"    Max: {max(point_counts):,}")
        print(f"    Media: {int(np.mean(point_counts)):,}")

    print()
    result.print_report()
    return result


# ── Validación datos reales formato simple ────────────────────────────


def validate_real_simple(data_dir: Path) -> ValidationResult:
    """Valida datos reales en formato simple: un .laz por árbol + labels.csv."""
    result = ValidationResult()
    print(f"\n{'='*60}")
    print(f" Validando datos REALES (formato simple): {data_dir}")
    print(f"{'='*60}\n")

    # 1. Existen archivos .laz/.las
    laz_files = find_laz_files(data_dir)
    if laz_files:
        result.ok(f"Encontrados {len(laz_files)} archivos .laz/.las")
    else:
        result.fail("No se encontraron archivos .laz/.las")
        result.print_report()
        return result

    # 2. Existe CSV con mapeo tree_id → especie
    df_labels, tree_col, species_col = find_label_csv(data_dir)
    if df_labels is not None:
        result.ok(
            f"CSV de etiquetas encontrado — columnas: '{tree_col}' (id), '{species_col}' (especie)"
        )
    else:
        result.fail(
            "No se encontró CSV con columnas de tree_id y especie. "
            f"Columnas de id aceptadas: {TREE_ID_COLUMNS}. "
            f"Columnas de especie aceptadas: {SPECIES_COLUMNS}"
        )
        result.print_report()
        return result

    # Limpiar etiquetas
    df_labels[species_col] = df_labels[species_col].astype(str).str.strip()
    df_labels = df_labels.dropna(subset=[species_col])
    df_labels = df_labels[df_labels[species_col] != ""]
    df_labels = df_labels[df_labels[species_col].str.lower() != "nan"]

    # 3. Clases
    species_counts = df_labels[species_col].value_counts()
    num_classes = len(species_counts)
    if num_classes >= MIN_CLASSES:
        result.ok(f"{num_classes} especies encontradas (min: {MIN_CLASSES})")
    else:
        result.fail(f"Solo {num_classes} especies encontradas (min: {MIN_CLASSES})")

    # 4. Muestras por clase
    small_classes = species_counts[species_counts < MIN_SAMPLES_PER_CLASS]
    if len(small_classes) == 0:
        result.ok(f"Todas las clases tienen >= {MIN_SAMPLES_PER_CLASS} muestras")
    else:
        result.fail(
            f"{len(small_classes)} clases con < {MIN_SAMPLES_PER_CLASS} muestras: "
            f"{dict(small_classes)}"
        )

    # 5. Archivos legibles
    laz_name_map = {f.stem: f for f in laz_files}
    sample_files = laz_files[:50] if len(laz_files) > 50 else laz_files
    readable = 0
    unreadable: list[str] = []
    point_counts: list[int] = []
    xyz_valid = 0
    xyz_invalid: list[str] = []

    for laz_path in sample_files:
        xyz = read_laz_xyz(laz_path)
        if xyz is not None:
            readable += 1
            point_counts.append(len(xyz))
            has_nan = np.isnan(xyz).any()
            all_zero = np.allclose(xyz, 0)
            too_few = len(xyz) < MIN_POINTS_PER_CLOUD
            if not has_nan and not all_zero and not too_few:
                xyz_valid += 1
            else:
                reasons = []
                if has_nan:
                    reasons.append("NaN")
                if all_zero:
                    reasons.append("todo ceros")
                if too_few:
                    reasons.append(f"solo {len(xyz)} puntos")
                xyz_invalid.append(f"{laz_path.name} ({', '.join(reasons)})")
        else:
            unreadable.append(laz_path.name)

    tested = len(sample_files)
    if readable == tested:
        result.ok(f"Todos los {tested} archivos probados son legibles con laspy")
    else:
        result.fail(
            f"{len(unreadable)}/{tested} archivos no se pudieron leer: "
            f"{unreadable[:5]}{'...' if len(unreadable) > 5 else ''}"
        )

    if xyz_valid == readable:
        result.ok(f"Todos los {readable} archivos tienen XYZ válido")
    else:
        result.fail(
            f"{len(xyz_invalid)} archivos con XYZ inválido: "
            f"{xyz_invalid[:5]}{'...' if len(xyz_invalid) > 5 else ''}"
        )

    # Matchear .laz con CSV
    tree_ids_csv = set(df_labels[tree_col].astype(str).str.strip())
    laz_stems = set(laz_name_map.keys())
    matched = tree_ids_csv & laz_stems
    if matched:
        result.ok(f"{len(matched)} archivos .laz coinciden con entradas del CSV")
    else:
        result.warn(
            "Ningún nombre de archivo .laz coincide con el tree_id del CSV. "
            f"Verifica que los stems coincidan con la columna '{tree_col}'."
        )

    # Reporte
    print("\n  ── Distribución de clases ──")
    for species, count in species_counts.items():
        print(f"    {species}: {count} muestras")

    if point_counts:
        print(f"\n  ── Puntos por nube (muestra de {tested}) ──")
        print(f"    Min: {min(point_counts):,}")
        print(f"    Max: {max(point_counts):,}")
        print(f"    Media: {int(np.mean(point_counts)):,}")
        print(f"    Mediana: {int(np.median(point_counts)):,}")

    print(f"\n  Total archivos .laz: {len(laz_files)}")
    print(f"  Total entradas en CSV: {len(df_labels)}")
    print()

    result.print_report()
    return result


# ── Validación de datos sintéticos ───────────────────────────────────


def validate_synthetic(data_dir: Path) -> ValidationResult:
    result = ValidationResult()
    print(f"\n{'='*60}")
    print(f" Validando datos SINTETICOS: {data_dir}")
    print(f"{'='*60}\n")

    # 1. Existen archivos .laz/.las
    laz_files = find_laz_files(data_dir)
    if laz_files:
        result.ok(f"Encontrados {len(laz_files)} archivos .laz/.las")
    else:
        result.fail("No se encontraron archivos .laz/.las")
        result.print_report()
        return result

    # 2. Buscar etiquetas: primero CSV externo, luego campos embebidos
    df_labels, tree_col, species_col = find_label_csv(data_dir)
    species_map = load_species_map(data_dir)

    species_counts = pd.Series(dtype=int)

    if df_labels is not None:
        result.ok(
            f"CSV de etiquetas encontrado — columnas: '{tree_col}' (id), '{species_col}' (especie)"
        )
        df_labels[species_col] = df_labels[species_col].astype(str).str.strip()
        df_labels = df_labels.dropna(subset=[species_col])
        df_labels = df_labels[df_labels[species_col] != ""]
        df_labels = df_labels[df_labels[species_col].str.lower() != "nan"]
        species_counts = df_labels[species_col].value_counts()
    else:
        # Intentar leer etiquetas de los archivos .laz
        result.warn("No se encontró CSV de etiquetas — buscando campos en los .laz...")
        sample_file = laz_files[0]
        label_info = get_laz_label_field(sample_file)
        if label_info is not None:
            field_name, _ = label_info
            result.ok(f"Campo de etiqueta encontrado en .laz: '{field_name}'")

            all_labels: list[str] = []
            scan_files = laz_files[:200] if len(laz_files) > 200 else laz_files
            for f in scan_files:
                info = get_laz_label_field(f)
                if info is not None:
                    _, vals = info
                    unique_vals = np.unique(vals)
                    if len(unique_vals) == 1:
                        lbl = str(unique_vals[0])
                    else:
                        values, counts = np.unique(vals, return_counts=True)
                        lbl = str(values[np.argmax(counts)])
                    if species_map and lbl in species_map:
                        lbl = species_map[lbl]
                    all_labels.append(lbl)

            species_counts = pd.Series(all_labels).value_counts()
        else:
            result.fail(
                "No se encontró CSV de etiquetas ni campo de especie en los .laz. "
                "Incluye un labels.csv o asegúrate de que los .laz tengan un campo "
                f"de etiqueta ({LAZ_LABEL_FIELDS})."
            )
            result.print_report()
            return result

    if species_map:
        result.ok(f"species_map.json encontrado con {len(species_map)} mapeos")

    # 3. Clases
    num_classes = len(species_counts)
    if num_classes >= MIN_CLASSES:
        result.ok(f"{num_classes} especies encontradas (min: {MIN_CLASSES})")
    else:
        result.fail(f"Solo {num_classes} especies encontradas (min: {MIN_CLASSES})")

    # 4. Muestras por clase
    small_classes = species_counts[species_counts < MIN_SAMPLES_PER_CLASS]
    if len(small_classes) == 0:
        result.ok(f"Todas las clases tienen >= {MIN_SAMPLES_PER_CLASS} muestras")
    else:
        result.fail(
            f"{len(small_classes)} clases con < {MIN_SAMPLES_PER_CLASS} muestras: "
            f"{dict(small_classes)}"
        )

    # 5. Legibilidad y validez XYZ
    sample_files = laz_files[:50] if len(laz_files) > 50 else laz_files
    readable = 0
    unreadable: list[str] = []
    point_counts: list[int] = []
    xyz_valid = 0
    xyz_invalid: list[str] = []

    for laz_path in sample_files:
        xyz = read_laz_xyz(laz_path)
        if xyz is not None:
            readable += 1
            point_counts.append(len(xyz))
            has_nan = np.isnan(xyz).any()
            all_zero = np.allclose(xyz, 0)
            too_few = len(xyz) < MIN_POINTS_PER_CLOUD
            if not has_nan and not all_zero and not too_few:
                xyz_valid += 1
            else:
                reasons = []
                if has_nan:
                    reasons.append("NaN")
                if all_zero:
                    reasons.append("todo ceros")
                if too_few:
                    reasons.append(f"solo {len(xyz)} puntos")
                xyz_invalid.append(f"{laz_path.name} ({', '.join(reasons)})")
        else:
            unreadable.append(laz_path.name)

    tested = len(sample_files)
    if readable == tested:
        result.ok(f"Todos los {tested} archivos probados son legibles con laspy")
    else:
        result.fail(
            f"{len(unreadable)}/{tested} archivos no se pudieron leer: "
            f"{unreadable[:5]}{'...' if len(unreadable) > 5 else ''}"
        )

    if xyz_valid == readable:
        result.ok(f"Todos los {readable} archivos tienen XYZ válido")
    else:
        result.fail(
            f"{len(xyz_invalid)} archivos con XYZ inválido: "
            f"{xyz_invalid[:5]}{'...' if len(xyz_invalid) > 5 else ''}"
        )

    # Reporte
    print("\n  ── Distribución de clases ──")
    for species, count in species_counts.items():
        print(f"    {species}: {count} muestras")

    if point_counts:
        print(f"\n  ── Puntos por nube (muestra de {tested}) ──")
        print(f"    Min: {min(point_counts):,}")
        print(f"    Max: {max(point_counts):,}")
        print(f"    Media: {int(np.mean(point_counts)):,}")
        print(f"    Mediana: {int(np.median(point_counts)):,}")

    print(f"\n  Total archivos .laz: {len(laz_files)}")
    print()

    result.print_report()
    return result


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validar datos crudos antes del preprocesamiento"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["real", "synthetic"],
        required=True,
        help="Qué dataset validar",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Ruta al archivo de configuración",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["data"]["raw_dir"]) / args.dataset

    if not raw_dir.exists():
        print(f"\n\u2717 El directorio {raw_dir} no existe.")
        print(f"  Crea el directorio y coloca los archivos antes de validar.")
        sys.exit(1)

    if args.dataset == "real":
        # Auto-detectar formato
        if detect_idtrees_structure(raw_dir):
            result = validate_real_idtrees(raw_dir)
        else:
            result = validate_real_simple(raw_dir)
    else:
        result = validate_synthetic(raw_dir)

    # Veredicto final
    print()
    if result.passed:
        print("=" * 60)
        print(" DATOS LISTOS PARA PREPROCESAMIENTO")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print(" DATOS INCOMPLETOS -- revisar los errores arriba")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
