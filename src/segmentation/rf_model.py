"""
Random Forest semántico para segmentación LiDAR punto a punto.

Características derivadas de XYZ normalizado (mismo input que PointNet++):

  f0  z              — altura normalizada dentro del bloque [-1, 1]
  f1  x              — posición X normalizada [-1, 1]
  f2  y              — posición Y normalizada [-1, 1]
  f3  z_rank_pct     — rango percentil de z en el bloque [0, 1]
  f4  z_above_mean   — z − media(z del bloque)
  f5  z_norm_block   — z normalizada al rango del bloque [0, 1]
  f6  dist_center    — distancia 2D al centro del bloque
  f7  local_cnt_r10  — densidad local radio 0.10 (unid. norm.)
  f8  local_mean_r10 — z media de vecinos en radio 0.10
  f9  local_max_r10  — z máxima de vecinos en radio 0.10
  f10 local_std_r10  — desviación típica z en radio 0.10
  f11 local_cnt_r20  — densidad local radio 0.20
  f12 local_mean_r20 — z media de vecinos en radio 0.20
  f13 local_max_r20  — z máxima de vecinos en radio 0.20
  f14 local_std_r20  — desviación típica z en radio 0.20

Principio de equidad: todas las características se derivan exclusivamente de las
tres coordenadas XYZ normalizadas — la misma representación de entrada que
recibe PointNet++. No se usa clasificación ASPRS ni ninguna otra etiqueta externa.

Modelo: sklearn RandomForestClassifier con class_weight='balanced' para manejar
el desbalance de clases (puntos árbol ≈ 15-20 % del total).
"""

import numpy as np
import joblib
from pathlib import Path


FEATURE_NAMES = [
    "z", "x", "y",
    "z_rank_pct", "z_above_mean", "z_norm_block", "dist_center",
    "local_cnt_k16", "local_mean_k16", "local_max_k16", "local_std_k16",
    "local_cnt_k32", "local_mean_k32", "local_max_k32", "local_std_k32",
]
N_FEATURES = len(FEATURE_NAMES)  # 15


def extract_features(points: np.ndarray, k_near: int = 32) -> np.ndarray:
    """Extrae 15 características por punto de un bloque XYZ normalizado.

    Usa kNN fijo (k=32 vecinos) para estadísticas locales, que es O(N log N)
    y completamente vectorizable — mucho más rápido que radius search variable.
    Dos escalas: k//2 vecinos (radio corto) y k vecinos (radio largo).

    Args:
        points: (N, 3) float32 — coordenadas normalizadas, aprox. en [-1, 1]
        k_near: número de vecinos para estadísticas locales

    Returns:
        features: (N, 15) float32
    """
    from scipy.spatial import cKDTree

    N = len(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # ---------- Estadísticas globales del bloque -------------------------
    z_mean  = float(z.mean())
    z_min   = float(z.min())
    z_max   = float(z.max())
    z_range = z_max - z_min

    # f0-f2: coordenadas brutas
    f_z = z.astype(np.float32)
    f_x = x.astype(np.float32)
    f_y = y.astype(np.float32)

    # f3: rango percentil de z (posición robusta en la distribución vertical)
    f_z_rank = np.argsort(np.argsort(z)).astype(np.float32) / max(N - 1, 1)

    # f4: desviación respecto a la media del bloque
    f_z_above_mean = (z - z_mean).astype(np.float32)

    # f5: z normalizada al rango del bloque [0, 1]
    if z_range > 1e-4:
        f_z_norm_block = ((z - z_min) / z_range).astype(np.float32)
    else:
        f_z_norm_block = np.zeros(N, dtype=np.float32)

    # f6: distancia 2D al centro del bloque (centrado en origen por normalización)
    f_dist_center = np.sqrt(x**2 + y**2).astype(np.float32)

    # ---------- Estadísticas locales por vecindad (kNN vectorizado) ------
    # query retorna (N, k) de distancias e índices — completamente vectorizable
    k = min(k_near, N - 1)
    kdt = cKDTree(points[:, :2])
    dists, nbr_idx = kdt.query(points[:, :2], k=k + 1)  # +1 porque incluye el propio punto
    # Excluir el punto propio (índice 0, distancia 0)
    dists   = dists[:, 1:]    # (N, k)
    nbr_idx = nbr_idx[:, 1:]  # (N, k)

    z_nbrs = z[nbr_idx]  # (N, k) — z de todos los vecinos, vectorizado

    # Escala corta: primeros k//2 vecinos (vecindad pequeña)
    k_short = max(k // 2, 1)
    z_short = z_nbrs[:, :k_short]
    local_feats_short = [
        (k_short / 100.0 * np.ones(N, dtype=np.float32)),   # cnt (constante para kNN)
        z_short.mean(axis=1).astype(np.float32),              # mean_z
        z_short.max(axis=1).astype(np.float32),               # max_z
        z_short.std(axis=1).astype(np.float32),               # std_z
    ]

    # Escala larga: todos los k vecinos (vecindad grande)
    local_feats_long = [
        (k / 100.0 * np.ones(N, dtype=np.float32)),           # cnt
        z_nbrs.mean(axis=1).astype(np.float32),                # mean_z
        z_nbrs.max(axis=1).astype(np.float32),                 # max_z
        z_nbrs.std(axis=1).astype(np.float32),                 # std_z
    ]

    features = np.column_stack([
        f_z, f_x, f_y,
        f_z_rank, f_z_above_mean, f_z_norm_block, f_dist_center,
        *local_feats_short,
        *local_feats_long,
    ]).astype(np.float32)

    return features  # (N, 15)


def extract_features_batch(blocks_points: np.ndarray,
                           max_blocks: int = None,
                           verbose: bool = True) -> tuple:
    """Extrae features de múltiples bloques.

    Args:
        blocks_points: (B, N, 3) — bloques normalizados
        max_blocks: si no None, limita a ese número de bloques
        verbose: muestra progreso

    Returns:
        X: (B*N, 15) features
        indices: lista de (start, end) para reconstruir por bloque
    """
    B = len(blocks_points)
    if max_blocks and B > max_blocks:
        rng = np.random.default_rng(42)
        sel = rng.choice(B, max_blocks, replace=False)
        blocks_points = blocks_points[sel]
        B = max_blocks
        if verbose:
            print(f"  Submuestreando a {B} bloques para extracción de features")

    all_feats = []
    indices   = []
    cursor    = 0

    for i, pts in enumerate(blocks_points):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Features: bloque {i+1}/{B}", flush=True)
        feats = extract_features(pts)
        all_feats.append(feats)
        indices.append((cursor, cursor + len(feats)))
        cursor += len(feats)

    X = np.vstack(all_feats)
    return X, indices


# ---------------------------------------------------------------------------
# Modelo RF
# ---------------------------------------------------------------------------

class RFSegmentationModel:
    """
    Random Forest para segmentación semántica punto a punto en LiDAR.

    Parámetros elegidos para equidad en comparación con PointNet++:
    - n_estimators=200: suficiente para convergencia estable
    - max_depth=20: profundidad que permite capturar patrones no lineales
    - min_samples_leaf=5: regularización para evitar overfitting
    - class_weight='balanced': compensa el desbalance árbol/no-árbol (≈1:5)
    - n_jobs=-1: paralelismo completo (sin restricción de VRAM como PointNet++)
    """

    DEFAULT_PARAMS = dict(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        params = {**self.DEFAULT_PARAMS, **kwargs}
        self.clf = RandomForestClassifier(**params)
        self.feature_names = FEATURE_NAMES
        self.n_features = N_FEATURES
        self.train_stats: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFSegmentationModel":
        """Entrena el RF. X: (N, 15), y: (N,) binario."""
        import time
        t0 = time.time()
        self.clf.fit(X, y)
        self.train_stats["fit_time_s"] = round(time.time() - t0, 2)
        self.train_stats["n_train_points"] = len(y)
        self.train_stats["tree_ratio"] = float(y.mean())
        if hasattr(self.clf, "oob_score_"):
            self.train_stats["oob_score"] = round(float(self.clf.oob_score_), 4)
        return self

    def predict_proba_tree(self, X: np.ndarray) -> np.ndarray:
        """Probabilidad de clase árbol (columna 1). Retorna (N,) float32."""
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción binaria. Retorna (N,) int."""
        return self.clf.predict(X)

    def feature_importances(self) -> dict:
        imp = self.clf.feature_importances_
        return {k: round(float(v), 5) for k, v in zip(self.feature_names, imp)}

    def model_size_mb(self) -> float:
        """Tamaño aproximado del modelo serializado en MB."""
        import pickle
        return len(pickle.dumps(self.clf)) / 1e6

    def save(self, path: str):
        joblib.dump(self, path, compress=3)
        print(f"  RF guardado en {path} ({self.model_size_mb():.1f} MB)", flush=True)

    @staticmethod
    def load(path: str) -> "RFSegmentationModel":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Inferencia sobre tile completo (equivalente a segment_scene de pipeline.py)
# ---------------------------------------------------------------------------

def segment_scene_rf(xyz: np.ndarray, classification: np.ndarray,
                     model: "RFSegmentationModel", cfg: dict) -> np.ndarray:
    """
    Segmentación RF de un tile LiDAR completo mediante ventanas deslizantes.

    Replica exactamente el protocolo de segment_scene (PointNet++) en pipeline.py:
    - Misma ventana (block_size) y stride
    - Misma normalización de coordenadas por bloque
    - Mismo voto por punto (acumulación de probabilidades + umbral 0.5)
    - Mismo fallback ASPRS para puntos no cubiertos

    Returns:
        predictions: (N,) int32 — 0=no-árbol, 1=árbol
    """
    import time

    block_size     = cfg["data"]["block_size"]
    stride         = cfg["data"]["block_stride"]
    normalize_h    = cfg["data"]["normalize_height"]
    tree_classes   = cfg["data"]["tree_classes"]
    ignore_classes = cfg["data"]["ignore_classes"]

    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    N = len(xyz)

    vote_tree  = np.zeros(N, dtype=np.float64)
    vote_count = np.zeros(N, dtype=np.int32)

    ignore_mask = np.zeros(N, dtype=bool)
    for cls in ignore_classes:
        ignore_mask |= (classification == cls)

    x_starts = np.arange(x_min, x_max - block_size / 2, stride)
    y_starts = np.arange(y_min, y_max - block_size / 2, stride)

    t_feat = 0.0
    t_infer = 0.0
    n_blocks = 0

    for x0 in x_starts:
        for y0 in y_starts:
            mask = (
                (x >= x0) & (x < x0 + block_size) &
                (y >= y0) & (y < y0 + block_size) &
                (~ignore_mask)
            )
            indices = np.where(mask)[0]
            n_pts = len(indices)
            if n_pts < 64:
                continue

            block_xyz = xyz[indices].copy().astype(np.float64)

            # Normalización de altura (idéntica a pipeline.py)
            if normalize_h:
                ground_mask = classification[indices] == 2
                if ground_mask.sum() > 0:
                    ground_z = np.median(block_xyz[ground_mask, 2])
                else:
                    ground_z = np.percentile(block_xyz[:, 2], 5)
                block_xyz[:, 2] -= ground_z

            # Normalización XY al centro del bloque
            block_xyz[:, 0] -= (x0 + block_size / 2)
            block_xyz[:, 1] -= (y0 + block_size / 2)
            block_xyz[:, 0] /= (block_size / 2)
            block_xyz[:, 1] /= (block_size / 2)

            # Normalización Z al rango del bloque
            z_range = block_xyz[:, 2].max() - block_xyz[:, 2].min()
            if z_range > 0:
                block_xyz[:, 2] = (
                    (block_xyz[:, 2] - block_xyz[:, 2].min()) / z_range * 2 - 1
                )
            else:
                block_xyz[:, 2] = 0.0

            # Extracción de features y predicción RF
            t0 = time.perf_counter()
            feats = extract_features(block_xyz.astype(np.float32))
            t_feat += time.perf_counter() - t0

            t0 = time.perf_counter()
            probs = model.predict_proba_tree(feats)  # (n_pts,)
            t_infer += time.perf_counter() - t0

            vote_tree[indices]  += probs.astype(np.float64)
            vote_count[indices] += 1
            n_blocks += 1

    # Umbral y fallback
    valid = vote_count > 0
    predictions = np.zeros(N, dtype=np.int32)
    predictions[valid] = (vote_tree[valid] / vote_count[valid] > 0.5).astype(np.int32)

    never_seen = ~valid & ~ignore_mask
    for tc in tree_classes:
        predictions[never_seen & (classification == tc)] = 1

    timing = {
        "n_blocks": n_blocks,
        "t_features_s": round(t_feat, 3),
        "t_inference_s": round(t_infer, 3),
    }

    return predictions, timing
