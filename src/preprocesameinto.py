"""Preprocesamiento morfológico y clustering DBSCAN en dos etapas (Pregunta 1).

Implementa la limpieza estructural de ``Dataset.csv``, análisis exploratorio y
de colinealidad, búsqueda y sensibilidad de hiperparámetros para un esquema
DBSCAN en dos etapas:
  1. Etapa 1 (RobustScaler 16D): separa ruido global y preserva la subpoblación
     de granos gigantes como clase propia.
  2. Etapa 2 (StandardScaler + PCA(2) sobre el grupo principal): identifica los
     estratos morfológicos internos y el ruido de frontera/periferia.

Nota: el nombre del archivo respeta exactamente la ortografía solicitada en
``2o parcial.pdf`` (``preprocesameinto.py``).
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

RANDOM_STATE: int = 42
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

FEATURE_COLUMNS: list[str] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

SIZE_LOG_COLUMNS: list[str] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
]

CLASS_DESCRIPTIONS: dict[int, str] = {
    0: "Clase 0: Pequenos-medianos compactos",
    1: "Clase 1: Medianos-grandes",
    2: "Clase 2: Alargados / excentricos",
    3: "Clase 3: Gigantes (Area alta)",
}


def load_and_clean_dataset(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carga ``Dataset.csv``, elimina columnas vacías y duplicados exactos.

    Args:
        csv_path: Ruta al archivo CSV original.

    Returns:
        Tupla ``(df_limpio, resumen_auditoria)`` con las 16 columnas reales
        deduplicadas y sin valores nulos ni infinitos.
    """
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw_shape = df_raw.shape

    all_nan_cols = [col for col in df_raw.columns if df_raw[col].isna().all()]
    df = df_raw.drop(columns=all_nan_cols).copy()

    if list(df.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"Columnas inesperadas tras limpiar vacías: {list(df.columns)}"
        )

    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    if df.isna().any().any():
        raise AssertionError("Se encontraron valores NaN tras la limpieza.")
    if np.isinf(df.to_numpy(dtype=float)).any():
        raise AssertionError("Se encontraron valores infinitos tras la limpieza.")

    invariants: dict[str, float | int] = {
        "area_gt_convex_count": int((df["Area"] > df["ConvexArea"]).sum()),
        "max_diff_solidity": float(
            np.max(np.abs(df["Solidity"] - df["Area"] / df["ConvexArea"]))
        ),
        "max_diff_aspect_ratio": float(
            np.max(
                np.abs(
                    df["AspectRation"] - df["MajorAxisLength"] / df["MinorAxisLength"]
                )
            )
        ),
        "max_diff_equiv_diameter": float(
            np.max(np.abs(df["EquivDiameter"] - np.sqrt(4.0 * df["Area"] / np.pi)))
        ),
        "max_diff_roundness": float(
            np.max(
                np.abs(
                    df["roundness"]
                    - (4.0 * np.pi * df["Area"]) / (df["Perimeter"] ** 2)
                )
            )
        ),
        "max_diff_compactness": float(
            np.max(
                np.abs(df["Compactness"] - df["EquivDiameter"] / df["MajorAxisLength"])
            )
        ),
        "max_diff_sf3": float(
            np.max(np.abs(df["ShapeFactor3"] - df["Compactness"] ** 2))
        ),
    }

    summary: dict[str, Any] = {
        "raw_rows": int(raw_shape[0]),
        "raw_cols": int(raw_shape[1]),
        "dropped_all_nan_cols": all_nan_cols,
        "exact_duplicates_removed": n_duplicates,
        "clean_rows": int(df.shape[0]),
        "clean_cols": int(df.shape[1]),
        "invariants": invariants,
    }
    return df, summary


def compute_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas, asimetría y conteo de outliers 16D."""
    records: list[dict[str, Any]] = []
    for col in FEATURE_COLUMNS:
        s = df[col]
        q1 = float(s.quantile(0.25))
        med = float(s.quantile(0.50))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        mean_v = float(s.mean())
        std_v = float(s.std(ddof=1))
        iqr_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        z3_out = int((np.abs(s - mean_v) > 3.0 * std_v).sum())
        records.append(
            {
                "feature": col,
                "mean": mean_v,
                "std": std_v,
                "min": float(s.min()),
                "q1": q1,
                "median": med,
                "q3": q3,
                "max": float(s.max()),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurt()),
                "iqr_outliers": iqr_out,
                "z3_outliers": z3_out,
            }
        )
    return pd.DataFrame(records)


def compute_k_distance_curve(
    X: np.ndarray, k: int
) -> tuple[np.ndarray, float, float]:
    """Calcula la curva k-distance ordenada y su codo geométrico por cuerda.

    Args:
        X: Matriz de características escalada.
        k: Número de vecinos (corresponde a ``min_samples``).

    Returns:
        Tupla ``(distancias_ordenadas, eps_codo, percentil_codo)``.
    """
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    dists, _ = nn.kneighbors(X)
    kdist = np.sort(dists[:, -1])

    x_norm = np.linspace(0.0, 1.0, len(kdist))
    denom = float(kdist[-1] - kdist[0])
    y_norm = (kdist - kdist[0]) / denom if denom > 0 else np.zeros_like(kdist)
    knee_idx = int(np.argmax(x_norm - y_norm))
    knee_val = float(kdist[knee_idx])
    knee_pct = float(100.0 * knee_idx / max(len(kdist) - 1, 1))
    return kdist, knee_val, knee_pct


def _safe_ari_against_reference(ref_labels: np.ndarray, cand_labels: np.ndarray) -> float:
    """Calcula ARI entre puntos no ruido comunes; retorna 0.0 si ``k < 2``."""
    n_clusters_cand = int(np.sum(np.unique(cand_labels) != -1))
    if n_clusters_cand < 2:
        return 0.0
    common = (ref_labels != -1) & (cand_labels != -1)
    if int(common.sum()) == 0:
        return 0.0
    return float(adjusted_rand_score(ref_labels[common], cand_labels[common]))


def fit_two_stage_dbscan(
    df: pd.DataFrame,
    eps1: float = 1.50,
    ms1: int = 20,
    eps2: float = 0.38,
    ms2: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Ejecuta el clustering DBSCAN en dos etapas y asigna clases ``0..k-1``.

    Etapa 1: ``RobustScaler`` sobre 16 variables para detectar ruido global y
    aislar la subpoblación cohesiva de granos gigantes.
    Etapa 2: ``StandardScaler + PCA(2)`` ajustado únicamente sobre el grupo
    principal de la Etapa 1 para descubrir los estratos internos.

    Args:
        df: DataFrame limpio de 16 columnas morfológicas.
        eps1: Radio epsilon para Etapa 1.
        ms1: ``min_samples`` para Etapa 1.
        eps2: Radio epsilon para Etapa 2.
        ms2: ``min_samples`` para Etapa 2.
        random_state: Semilla reproducible para PCA.

    Returns:
        Diccionario con etiquetas combinadas, orígenes de ruido, objetos
        ajustados y proyecciones para diagnósticos y visualización.
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    # Etapa 1: RobustScaler en 16D
    scaler1 = RobustScaler()
    X_rob = scaler1.fit_transform(X)
    lbl1 = DBSCAN(eps=eps1, min_samples=ms1, n_jobs=-1).fit_predict(X_rob)

    u1, c1 = np.unique(lbl1[lbl1 != -1], return_counts=True)
    if len(u1) < 2:
        raise ValueError(
            f"Etapa 1 produjo menos de 2 clusters válidos (k={len(u1)}) "
            f"con eps1={eps1}, ms1={ms1}."
        )
    if len(u1) != 2 and np.isclose(eps1, 1.50) and ms1 == 20:
        raise ValueError(
            f"Etapa 1 debía producir exactamente 2 clusters con eps1=1.50, ms1=20; obtuvo k={len(u1)}."
        )

    main_id = int(u1[np.argmax(c1)])
    other_ids = [int(uid) for uid in u1 if int(uid) != main_id]
    giant_id = max(
        other_ids, key=lambda uid: float(np.median(df.loc[lbl1 == uid, "Area"]))
    )

    main_mask = lbl1 == main_id
    giant_mask = lbl1 == giant_id
    noise1_mask = lbl1 == -1

    # Etapa 2: StandardScaler + PCA(2) ajustado exclusivamente sobre el grupo principal
    X_main = X[main_mask]
    scaler2 = StandardScaler()
    X_main_std = scaler2.fit_transform(X_main)
    pca2 = PCA(n_components=2, random_state=random_state)
    X_main_pca2 = pca2.fit_transform(X_main_std)

    lbl2_raw = DBSCAN(eps=eps2, min_samples=ms2, n_jobs=-1).fit_predict(X_main_pca2)
    u2, c2 = np.unique(lbl2_raw[lbl2_raw != -1], return_counts=True)
    if len(u2) < 2:
        raise ValueError(
            f"Etapa 2 produjo menos de 2 clusters válidos (k={len(u2)}) "
            f"con eps2={eps2}, ms2={ms2}."
        )

    order2 = u2[np.argsort(-c2)]
    remap2 = {int(old_lbl): idx for idx, old_lbl in enumerate(order2)}
    lbl2_mapped = np.array(
        [remap2[int(v)] if v != -1 else -1 for v in lbl2_raw], dtype=np.int64
    )

    giant_class_id = len(order2)
    labels_combined = np.full(len(df), -1, dtype=np.int64)
    labels_combined[main_mask] = lbl2_mapped
    labels_combined[giant_mask] = giant_class_id

    noise_origin = np.full(len(df), "", dtype=object)
    noise_origin[noise1_mask] = "etapa1_global"
    main_indices = np.flatnonzero(main_mask)
    noise2_local_mask = lbl2_mapped == -1
    noise_origin[main_indices[noise2_local_mask]] = "etapa2_interno"

    # Representación global estandarizada únicamente para evaluación métrica 16D y figura global
    scaler_global_std = StandardScaler()
    X_std_full = scaler_global_std.fit_transform(X)
    pca_global = PCA(n_components=2, random_state=random_state)
    X_pca2_global = pca_global.fit_transform(X_std_full)

    return {
        "X": X,
        "X_rob": X_rob,
        "X_std_full": X_std_full,
        "X_pca2_global": X_pca2_global,
        "pca_global_var": pca_global.explained_variance_ratio_,
        "X_main_std": X_main_std,
        "X_main_pca2": X_main_pca2,
        "pca2_var": pca2.explained_variance_ratio_,
        "pca2_components": pca2.components_,
        "scaler1": scaler1,
        "scaler2": scaler2,
        "pca2": pca2,
        "lbl1": lbl1,
        "lbl2": lbl2_mapped,
        "labels_combined": labels_combined,
        "noise_origin": noise_origin,
        "main_mask": main_mask,
        "giant_mask": giant_mask,
        "noise1_mask": noise1_mask,
        "noise2_mask": labels_combined == -1,
        "giant_class_id": giant_class_id,
        "k2": len(order2),
    }


def run_stage1_sensitivity_grid(
    X_rob: np.ndarray, ref_lbl1: np.ndarray
) -> pd.DataFrame:
    """Evalúa malla ``eps × min_samples`` para la Etapa 1 (RobustScaler 16D)."""
    ms_grid = [15, 20, 25, 32, 40]
    eps_grid = [1.20, 1.35, 1.50, 1.65, 1.80]
    rows: list[dict[str, Any]] = []
    n = len(X_rob)

    for ms in ms_grid:
        for eps in eps_grid:
            lbl = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(X_rob)
            u, c = np.unique(lbl, return_counts=True)
            n_noise = int(c[u == -1][0]) if -1 in u else 0
            k_cl = int(np.sum(u != -1))
            mask = lbl != -1

            if k_cl >= 2:
                sil_clean = float(
                    silhouette_score(
                        X_rob[mask],
                        lbl[mask],
                        sample_size=min(4000, int(mask.sum())),
                        random_state=RANDOM_STATE,
                    )
                )
                sil_all = float(
                    silhouette_score(
                        X_rob,
                        lbl,
                        sample_size=min(4000, n),
                        random_state=RANDOM_STATE,
                    )
                )
                ch = float(calinski_harabasz_score(X_rob[mask], lbl[mask]))
                db = float(davies_bouldin_score(X_rob[mask], lbl[mask]))
            else:
                sil_clean = np.nan
                sil_all = np.nan
                ch = np.nan
                db = np.nan

            ari = _safe_ari_against_reference(ref_lbl1, lbl)
            rows.append(
                {
                    "min_samples": ms,
                    "eps": eps,
                    "k": k_cl,
                    "n_noise": n_noise,
                    "pct_noise": 100.0 * n_noise / n,
                    "silhouette_excl_noise": sil_clean,
                    "silhouette_incl_noise": sil_all,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "ari_vs_ref": ari,
                }
            )
    return pd.DataFrame(rows)


def run_stage2_sensitivity_grid(
    X_main_pca2: np.ndarray, X_main_std: np.ndarray, ref_lbl2: np.ndarray
) -> pd.DataFrame:
    """Evalúa malla ``eps × min_samples`` para la Etapa 2 (PCA(2) del grupo principal)."""
    ms_grid = [30, 40, 50, 60, 70]
    eps_grid = [0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44]
    rows: list[dict[str, Any]] = []
    n_main = len(X_main_pca2)

    for ms in ms_grid:
        for eps in eps_grid:
            lbl = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(X_main_pca2)
            u, c = np.unique(lbl, return_counts=True)
            n_noise = int(c[u == -1][0]) if -1 in u else 0
            k_cl = int(np.sum(u != -1))
            mask = lbl != -1

            if k_cl >= 2:
                sil_p2_clean = float(
                    silhouette_score(
                        X_main_pca2[mask],
                        lbl[mask],
                        sample_size=min(4000, int(mask.sum())),
                        random_state=RANDOM_STATE,
                    )
                )
                sil_p2_all = float(
                    silhouette_score(
                        X_main_pca2,
                        lbl,
                        sample_size=min(4000, n_main),
                        random_state=RANDOM_STATE,
                    )
                )
                sil_16_clean = float(
                    silhouette_score(
                        X_main_std[mask],
                        lbl[mask],
                        sample_size=min(4000, int(mask.sum())),
                        random_state=RANDOM_STATE,
                    )
                )
                ch = float(calinski_harabasz_score(X_main_pca2[mask], lbl[mask]))
                db = float(davies_bouldin_score(X_main_pca2[mask], lbl[mask]))
            else:
                sil_p2_clean = np.nan
                sil_p2_all = np.nan
                sil_16_clean = np.nan
                ch = np.nan
                db = np.nan

            ari = _safe_ari_against_reference(ref_lbl2, lbl)
            rows.append(
                {
                    "min_samples": ms,
                    "eps": eps,
                    "k": k_cl,
                    "n_noise": n_noise,
                    "pct_noise_main": 100.0 * n_noise / n_main,
                    "silhouette_pca2_excl_noise": sil_p2_clean,
                    "silhouette_pca2_incl_noise": sil_p2_all,
                    "silhouette_16d_excl_noise": sil_16_clean,
                    "calinski_harabasz": ch,
                    "davies_bouldin": db,
                    "ari_vs_ref": ari,
                }
            )
    return pd.DataFrame(rows)


def run_log_skewness_comparison(
    df_main: pd.DataFrame, ref_lbl2: np.ndarray, ms2: int = 50, eps2: float = 0.38
) -> pd.DataFrame:
    """Compara la Etapa 2 sin logaritmo frente a transformación log en tamaño."""
    df_log = df_main.copy()
    for col in SIZE_LOG_COLUMNS:
        df_log[col] = np.log(df_log[col].to_numpy(dtype=float))

    X_log_pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        StandardScaler().fit_transform(df_log[FEATURE_COLUMNS].to_numpy(dtype=float))
    )
    X_nolog_pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        StandardScaler().fit_transform(df_main[FEATURE_COLUMNS].to_numpy(dtype=float))
    )

    rows: list[dict[str, Any]] = []
    configs: list[tuple[str, np.ndarray, int, float]] = [
        ("Sin log (referencia)", X_nolog_pca2, ms2, eps2)
    ] + [
        ("Con log(tamano)", X_log_pca2, ms2, eps_val)
        for eps_val in [0.28, 0.30, 0.32, 0.34, 0.36, 0.38]
    ]

    for label_cfg, X_space, ms_v, eps_v in configs:
        lbl = DBSCAN(eps=eps_v, min_samples=ms_v, n_jobs=-1).fit_predict(X_space)
        u, c = np.unique(lbl, return_counts=True)
        n_noise = int(c[u == -1][0]) if -1 in u else 0
        k_cl = int(np.sum(u != -1))
        mask = lbl != -1
        sil = (
            float(silhouette_score(X_space[mask], lbl[mask]))
            if k_cl >= 2
            else np.nan
        )
        ari = _safe_ari_against_reference(ref_lbl2, lbl)
        rows.append(
            {
                "tratamiento": label_cfg,
                "min_samples": ms_v,
                "eps": eps_v,
                "k": k_cl,
                "n_noise": n_noise,
                "pct_noise_main": 100.0 * n_noise / len(X_space),
                "silhouette_pca2_excl_noise": sil,
                "ari_vs_sin_log": ari,
            }
        )
    return pd.DataFrame(rows)


def run_subsampling_stability(
    X_rob: np.ndarray,
    lbl1_ref: np.ndarray,
    X_main_pca2: np.ndarray,
    eps1: float,
    ms1: int,
    eps2: float,
    ms2: int,
    n_runs: int = 20,
) -> pd.DataFrame:
    """Evalúa estabilidad de clusters bajo submuestreos aleatorios al 80 % y 90 %."""
    stage2_candidates: list[tuple[int, float, str]] = [
        (40, 0.36, "Etapa 2 (ms=40, eps=0.36)"),
        (50, 0.36, "Etapa 2 (ms=50, eps=0.36)"),
        (ms2, eps2, f"Etapa 2 ELEGIDA (ms={ms2}, eps={eps2:.2f})"),
        (50, 0.40, "Etapa 2 (ms=50, eps=0.40)"),
        (60, 0.40, "Etapa 2 (ms=60, eps=0.40)"),
    ]
    # Evitar duplicado exacto si el usuario elige una de las alternativas vía CLI
    seen_s2: set[tuple[int, float]] = set()
    unique_s2: list[tuple[int, float, str]] = []
    for ms_c, eps_c, desc in stage2_candidates:
        key = (ms_c, round(eps_c, 4))
        if key not in seen_s2:
            seen_s2.add(key)
            unique_s2.append((ms_c, eps_c, desc))

    specs: list[tuple[str, np.ndarray, np.ndarray, int, float, int]] = [
        (
            f"Etapa 1 ELEGIDA (ms={ms1}, eps={eps1:.2f})",
            X_rob,
            lbl1_ref,
            ms1,
            eps1,
            2,
        )
    ]
    for ms_c, eps_c, desc in unique_s2:
        ref_c = DBSCAN(eps=eps_c, min_samples=ms_c, n_jobs=-1).fit_predict(X_main_pca2)
        specs.append((desc, X_main_pca2, ref_c, ms_c, eps_c, 3))

    rows: list[dict[str, Any]] = []
    for desc, X_space, ref_lbl, ms_v, eps_v, k_target in specs:
        n_total = len(X_space)
        for frac in [0.90, 0.80]:
            rng = np.random.default_rng(RANDOM_STATE)
            ks: list[int] = []
            aris: list[float] = []
            sub_size = int(frac * n_total)
            for _ in range(n_runs):
                idx = rng.choice(n_total, size=sub_size, replace=False)
                lbl_sub = DBSCAN(eps=eps_v, min_samples=ms_v, n_jobs=-1).fit_predict(
                    X_space[idx]
                )
                k_sub = int(np.sum(np.unique(lbl_sub) != -1))
                ks.append(k_sub)
                aris.append(_safe_ari_against_reference(ref_lbl[idx], lbl_sub))
            u_k, c_k = np.unique(ks, return_counts=True)
            k_dist_str = "; ".join(f"k={int(k)}:{int(c)}" for k, c in zip(u_k, c_k))
            rows.append(
                {
                    "configuracion": desc,
                    "min_samples": ms_v,
                    "eps": eps_v,
                    "fraccion_submuestra": frac,
                    "n_corridas": n_runs,
                    "k_objetivo": k_target,
                    "corridas_k_objetivo": int(sum(1 for k in ks if k == k_target)),
                    "distribucion_k": k_dist_str,
                    "ari_medio": float(np.mean(aris)),
                    "ari_std": float(np.std(aris, ddof=1)),
                    "ari_min": float(np.min(aris)),
                    "ari_max": float(np.max(aris)),
                }
            )
    return pd.DataFrame(rows)


def build_noise_breakdown_table(
    df: pd.DataFrame, fit_res: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Desglosa el contenido físico y geométrico del ruido en Etapa 1 y Etapa 2."""
    n_total = len(df)
    subcat = np.full(n_total, "", dtype=object)

    # 1. Ruido Etapa 1 (mutuamente excluyente por prioridad diagnóstica)
    n1_idx = np.flatnonzero(fit_res["noise1_mask"])
    for idx in n1_idx:
        row = df.iloc[idx]
        if row["Area"] > 120_000.0:
            subcat[idx] = "Etapa 1: Gigantes perifericos (Area > 120k)"
        elif row["Solidity"] < 0.975:
            subcat[idx] = "Etapa 1: Baja solidez (< 0.975)"
        elif row["ShapeFactor4"] < 0.985 or row["roundness"] < 0.75:
            subcat[idx] = "Etapa 1: Deformidad borde/redondez"
        else:
            subcat[idx] = "Etapa 1: Otros atipicos globales 16D"

    # 2. Ruido Etapa 2 (geométricamente clasificado en PCA(2) del grupo principal)
    X_p2 = fit_res["X_main_pca2"]
    lbl2 = fit_res["lbl2"]
    main_idx = np.flatnonzero(fit_res["main_mask"])
    u2 = sorted([int(u) for u in np.unique(lbl2) if u != -1])
    centroids = np.array([X_p2[lbl2 == u].mean(axis=0) for u in u2])
    radii_p95 = [
        float(np.quantile(np.linalg.norm(X_p2[lbl2 == u] - centroids[i], axis=1), 0.95))
        for i, u in enumerate(u2)
    ]

    noise2_local = np.flatnonzero(lbl2 == -1)
    pts2 = X_p2[noise2_local]
    dists = np.linalg.norm(pts2[:, None, :] - centroids[None, :, :], axis=2)
    dsorted = np.sort(dists, axis=1)
    nearest_cl = np.argmin(dists, axis=1)

    for j, local_i in enumerate(noise2_local):
        global_i = int(main_idx[local_i])
        is_bridge = (dsorted[j, 0] / dsorted[j, 1]) > 0.65
        is_outer = dists[j, nearest_cl[j]] > radii_p95[int(nearest_cl[j])]
        if is_bridge and not is_outer:
            subcat[global_i] = "Etapa 2: Puente/frontera interior entre clusters"
        elif is_bridge and is_outer:
            subcat[global_i] = "Etapa 2: Puente/frontera exterior entre clusters"
        elif (not is_bridge) and is_outer:
            subcat[global_i] = "Etapa 2: Periferia externa (>p95 radio)"
        else:
            subcat[global_i] = "Etapa 2: Baja densidad local interior"

    order_cats = [
        "Etapa 1: Gigantes perifericos (Area > 120k)",
        "Etapa 1: Baja solidez (< 0.975)",
        "Etapa 1: Deformidad borde/redondez",
        "Etapa 1: Otros atipicos globales 16D",
        "Etapa 2: Puente/frontera interior entre clusters",
        "Etapa 2: Puente/frontera exterior entre clusters",
        "Etapa 2: Periferia externa (>p95 radio)",
        "Etapa 2: Baja densidad local interior",
    ]
    n_noise1_total = int(fit_res["noise1_mask"].sum())
    n_noise2_total = int(len(noise2_local))

    rows: list[dict[str, Any]] = []
    for cat in order_cats:
        mask = subcat == cat
        cnt = int(mask.sum())
        etapa = "Etapa 1" if cat.startswith("Etapa 1") else "Etapa 2"
        denom_etapa = n_noise1_total if etapa == "Etapa 1" else n_noise2_total
        sub_df = df.loc[mask]
        rows.append(
            {
                "etapa": etapa,
                "subcategoria_ruido": cat,
                "n_instancias": cnt,
                "pct_ruido_etapa": 100.0 * cnt / max(denom_etapa, 1),
                "pct_dataset_total": 100.0 * cnt / n_total,
                "area_mediana": float(sub_df["Area"].median()) if cnt > 0 else np.nan,
                "aspect_mediana": (
                    float(sub_df["AspectRation"].median()) if cnt > 0 else np.nan
                ),
                "compactness_mediana": (
                    float(sub_df["Compactness"].median()) if cnt > 0 else np.nan
                ),
                "solidity_mediana": (
                    float(sub_df["Solidity"].median()) if cnt > 0 else np.nan
                ),
                "n_ind_area_gt_120k": int((sub_df["Area"] > 120_000.0).sum()),
                "n_ind_solidity_lt_0975": int((sub_df["Solidity"] < 0.975).sum()),
                "n_ind_sf4_lt_0985": int((sub_df["ShapeFactor4"] < 0.985).sum()),
            }
        )
    return pd.DataFrame(rows), subcat


def build_cluster_profiles(df: pd.DataFrame, fit_res: dict[str, Any]) -> pd.DataFrame:
    """Genera perfil morfológico de los 4 clusters finales y los 2 grupos de ruido."""
    labels = fit_res["labels_combined"]
    origin = fit_res["noise_origin"]
    n_total = len(df)
    n_clean = int((labels != -1).sum())

    groups: list[tuple[str, np.ndarray]] = [
        (CLASS_DESCRIPTIONS.get(c, f"Clase {c}"), labels == c)
        for c in sorted(int(u) for u in np.unique(labels) if u != -1)
    ] + [
        ("Ruido Etapa 1 (Global 16D)", origin == "etapa1_global"),
        ("Ruido Etapa 2 (Frontera/Periferia)", origin == "etapa2_interno"),
    ]

    rows: list[dict[str, Any]] = []
    for name, mask in groups:
        sub = df.loc[mask]
        cnt = int(mask.sum())
        is_noise = name.startswith("Ruido")
        rows.append(
            {
                "grupo": name,
                "n": cnt,
                "pct_total": 100.0 * cnt / n_total,
                "pct_limpio": (100.0 * cnt / n_clean) if not is_noise else np.nan,
                "Area_min": float(sub["Area"].min()),
                "Area_median": float(sub["Area"].median()),
                "Area_max": float(sub["Area"].max()),
                "Perimeter_median": float(sub["Perimeter"].median()),
                "MajorAxis_median": float(sub["MajorAxisLength"].median()),
                "MinorAxis_median": float(sub["MinorAxisLength"].median()),
                "AspectRation_median": float(sub["AspectRation"].median()),
                "Compactness_median": float(sub["Compactness"].median()),
                "Solidity_median": float(sub["Solidity"].median()),
                "roundness_median": float(sub["roundness"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_multi_space_metrics(fit_res: dict[str, Any]) -> pd.DataFrame:
    """Calcula métricas exactas (sin submuestreo) en espacio clustering y en 16D."""
    X_rob = fit_res["X_rob"]
    X_std = fit_res["X_std_full"]
    lbl1 = fit_res["lbl1"]
    X_p2 = fit_res["X_main_pca2"]
    lbl2 = fit_res["lbl2"]
    comb = fit_res["labels_combined"]

    eval_specs: list[tuple[str, np.ndarray, np.ndarray, bool]] = [
        ("Etapa 1 (RobustScaler 16D, k=2)", X_rob, lbl1, False),
        ("Etapa 1 (RobustScaler 16D, k=2)", X_rob, lbl1, True),
        ("Etapa 2 (StdScaler+PCA2 Grupo Principal, k=3)", X_p2, lbl2, False),
        ("Etapa 2 (StdScaler+PCA2 Grupo Principal, k=3)", X_p2, lbl2, True),
        ("Particion Final 4 Clases en 16D StandardScaler", X_std, comb, False),
        ("Particion Final 4 Clases en 16D StandardScaler", X_std, comb, True),
        ("Particion Final 4 Clases en 16D RobustScaler", X_rob, comb, False),
        ("Particion Final 4 Clases en 16D RobustScaler", X_rob, comb, True),
    ]

    rows: list[dict[str, Any]] = []
    for desc, X_sp, lbl, include_noise in eval_specs:
        if include_noise:
            X_eval, lbl_eval = X_sp, lbl
            regimen = "Incluyendo ruido (-1)"
        else:
            m = lbl != -1
            X_eval, lbl_eval = X_sp[m], lbl[m]
            regimen = "Excluyendo ruido (-1)"

        rows.append(
            {
                "espacio_evaluacion": desc,
                "regimen": regimen,
                "n_puntos": int(len(lbl_eval)),
                "n_etiquetas": int(len(np.unique(lbl_eval))),
                "silhouette": float(silhouette_score(X_eval, lbl_eval)),
                "calinski_harabasz": float(calinski_harabasz_score(X_eval, lbl_eval)),
                "davies_bouldin": float(davies_bouldin_score(X_eval, lbl_eval)),
            }
        )
    return pd.DataFrame(rows)


def generate_eda_figures(df: pd.DataFrame, fig_dir: Path) -> None:
    """Genera histogramas de las 16 variables y mapa de calor de colinealidad."""
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Distribuciones 4x4
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for ax, col in zip(axes.flat, FEATURE_COLUMNS):
        vals = df[col].to_numpy(dtype=float)
        sns.histplot(vals, bins=40, kde=True, ax=ax, color="#2563eb", alpha=0.65)
        sk = float(df[col].skew())
        ax.set_title(f"{col}\n(skew={sk:+.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "Distribuciones Morfológicas tras Limpieza (N = 13,543 registros únicos)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(fig_dir / "eda_distribuciones.png", dpi=300)
    plt.close(fig)

    # 2. Correlación de Pearson triangular inferior
    corr = df[FEATURE_COLUMNS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.4,
        annot_kws={"size": 7.5},
        cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson (r)"},
        ax=ax,
    )
    ax.set_title(
        "Matriz de Correlación Morfológica (27 pares con |r| >= 0.85)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_correlacion.png", dpi=300)
    plt.close(fig)


def generate_k_distance_figure(
    fit_res: dict[str, Any], eps1: float, ms1: int, eps2: float, ms2: int, fig_dir: Path
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Genera curvas k-distance de Etapa 1 y Etapa 2 con anotación honesta del codo."""
    kd1, knee1_val, knee1_pct = compute_k_distance_curve(fit_res["X_rob"], k=ms1)
    kd2, knee2_val, knee2_pct = compute_k_distance_curve(fit_res["X_main_pca2"], k=ms2)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # Panel izquierdo: Etapa 1
    ax = axes[0]
    x1 = np.arange(len(kd1))
    ax.plot(x1, kd1, color="#1d4ed8", linewidth=1.8, label=f"k-distance ordenada (k={ms1})")
    ax.axhline(knee1_val, color="#dc2626", linestyle="--", linewidth=1.3,
               label=f"Codo k-dist = {knee1_val:.3f} (p{knee1_pct:.1f}%)")
    ax.axhline(eps1, color="#059669", linestyle="-.", linewidth=1.5,
               label=f"eps1 elegido = {eps1:.2f} (región del codo)")
    ax.set_ylim(0, min(6.0, float(np.quantile(kd1, 0.998)) * 1.15))
    ax.set_title("Etapa 1: RobustScaler 16D — Ruido Global y Gigantes", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Instancias ordenadas por distancia al k-ésimo vecino")
    ax.set_ylabel(f"Distancia Euclidiana 16D al vecino k={ms1}")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.35)

    # Panel derecho: Etapa 2
    ax = axes[1]
    x2 = np.arange(len(kd2))
    ax.plot(x2, kd2, color="#1d4ed8", linewidth=1.8, label=f"k-distance ordenada (k={ms2})")
    ax.axhline(knee2_val, color="#dc2626", linestyle="--", linewidth=1.3,
               label=f"Codo k-dist = {knee2_val:.3f} (p{knee2_pct:.1f}% -> colapsa a k=1)")
    ax.axhline(eps2, color="#059669", linestyle="-.", linewidth=1.6,
               label=f"eps2 elegido = {eps2:.2f} (meseta estable k=3, NO codo)")
    ax.axhspan(0.30, 0.42, color="#10b981", alpha=0.14, label="Meseta estable k=3 (ms=50)")
    ax.set_ylim(0, min(1.5, float(np.quantile(kd2, 0.995)) * 1.15))
    ax.set_title("Etapa 2: StdScaler + PCA(2) Grupo Principal", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Instancias del grupo principal ordenadas")
    ax.set_ylabel(f"Distancia Euclidiana PCA(2) al vecino k={ms2}")
    ax.legend(loc="upper left", fontsize=8.8, frameon=True)
    ax.grid(True, alpha=0.35)

    fig.suptitle(
        "Análisis de Curvas k-NN Distance para Selección de Parámetros DBSCAN",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "dbscan_k_distance_etapas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return (knee1_val, knee1_pct), (knee2_val, knee2_pct)


def _plot_sensitivity_heatmaps(
    df_grid: pd.DataFrame,
    noise_col: str,
    sil_col: str,
    ref_ms: int,
    ref_eps: float,
    title: str,
    out_path: Path,
    figsize: tuple[float, float],
) -> None:
    """Dibuja malla 2x2 de sensibilidad (k, % ruido, silueta, ARI) con celda destacada."""
    metrics = [
        ("k", "Número de Clusters Válidos (k)", "YlGnBu", ".0f", None, None),
        (noise_col, "% de Ruido", "OrRd", ".1f", None, None),
        (sil_col, "Índice de Silueta (sin ruido)", "viridis", ".3f", 0.25, 0.70),
        ("ari_vs_ref", "ARI vs. Configuración Elegida", "Blues", ".2f", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ms_vals = sorted(df_grid["min_samples"].unique())
    eps_vals = sorted(df_grid["eps"].unique())
    ref_row_idx = ms_vals.index(ref_ms) if ref_ms in ms_vals else None
    ref_col_idx = (
        int(np.argmin(np.abs(np.array(eps_vals) - ref_eps)))
        if len(eps_vals) > 0
        else None
    )

    for ax, (col, subtitle, cmap, fmt, vmin, vmax) in zip(axes.flat, metrics):
        piv = df_grid.pivot(index="min_samples", columns="eps", values=col)
        piv.columns = [f"{c:.2f}" for c in piv.columns]
        mask_na = piv.isna()
        sns.heatmap(
            piv,
            mask=mask_na,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.6,
            linecolor="white",
            annot_kws={"size": 8.5},
            ax=ax,
        )
        # Marcar celdas NaN en gris claro como N/A
        for r_i in range(piv.shape[0]):
            for c_i in range(piv.shape[1]):
                if mask_na.iloc[r_i, c_i]:
                    ax.add_patch(
                        patches.Rectangle((c_i, r_i), 1, 1, fill=True, color="#e5e7eb", lw=0)
                    )
                    ax.text(
                        c_i + 0.5,
                        r_i + 0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#4b5563",
                    )

        if ref_row_idx is not None and ref_col_idx is not None:
            ax.add_patch(
                patches.Rectangle(
                    (ref_col_idx, ref_row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="#dc2626",
                    linewidth=2.6,
                )
            )
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("eps")
        ax.set_ylabel("min_samples")

    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_clusters_2d_figure(
    fit_res: dict[str, Any], subcat: np.ndarray, fig_dir: Path
) -> None:
    """Genera proyección PCA(2) global (4 clases + ruido) y de Etapa 2 (con frontera)."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))

    palette_cl = {0: "#2563eb", 1: "#16a34a", 2: "#f59e0b", 3: "#9333ea"}
    short_names = {
        0: "Clase 0: Peq.-med. compactos",
        1: "Clase 1: Medianos-grandes",
        2: "Clase 2: Alargados/excéntricos",
        3: "Clase 3: Gigantes (Area alta)",
    }

    # Panel izquierdo: espacio PCA(2) global (13,543)
    ax = axes[0]
    Xp_glob = fit_res["X_pca2_global"]
    comb = fit_res["labels_combined"]
    origin = fit_res["noise_origin"]
    var_g = fit_res["pca_global_var"]

    m_n2 = origin == "etapa2_interno"
    m_n1 = origin == "etapa1_global"
    ax.scatter(
        Xp_glob[m_n2, 0],
        Xp_glob[m_n2, 1],
        s=9,
        c="#cbd5e1",
        alpha=0.55,
        label=f"Ruido Etapa 2 (frontera/perif., n={m_n2.sum()})",
        rasterized=True,
    )
    for c_id in sorted(palette_cl.keys()):
        mc = comb == c_id
        ax.scatter(
            Xp_glob[mc, 0],
            Xp_glob[mc, 1],
            s=10,
            c=palette_cl[c_id],
            alpha=0.65,
            label=f"{short_names[c_id]} (n={mc.sum()})",
            rasterized=True,
        )
    ax.scatter(
        Xp_glob[m_n1, 0],
        Xp_glob[m_n1, 1],
        s=22,
        c="#dc2626",
        marker="x",
        linewidths=1.0,
        alpha=0.85,
        label=f"Ruido Global Etapa 1 (n={m_n1.sum()})",
    )
    ax.set_title(
        f"Vista Global 4 Clases + Ruido (PCA global: {100*var_g.sum():.1f}% var.)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(f"PC1 global ({100*var_g[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 global ({100*var_g[1]:.1f}% var.)")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.4, frameon=True)
    ax.grid(True, alpha=0.3)

    # Panel derecho: espacio PCA(2) de Etapa 2 (12,836 del grupo principal)
    ax = axes[1]
    Xp_m = fit_res["X_main_pca2"]
    lbl2 = fit_res["lbl2"]
    var_2 = fit_res["pca2_var"]
    subcat_main = subcat[fit_res["main_mask"]]

    m_bridge = (lbl2 == -1) & np.isin(
        subcat_main,
        [
            "Etapa 2: Puente/frontera interior entre clusters",
            "Etapa 2: Puente/frontera exterior entre clusters",
        ],
    )
    m_periph = (lbl2 == -1) & (~m_bridge)

    ax.scatter(
        Xp_m[m_periph, 0],
        Xp_m[m_periph, 1],
        s=10,
        c="#94a3b8",
        alpha=0.55,
        label=f"Ruido Etapa 2: Periferia/disperso (n={m_periph.sum()})",
        rasterized=True,
    )
    ax.scatter(
        Xp_m[m_bridge, 0],
        Xp_m[m_bridge, 1],
        s=14,
        c="#ef4444",
        marker="^",
        alpha=0.75,
        label=f"Ruido Etapa 2: Puente/frontera (n={m_bridge.sum()})",
        rasterized=True,
    )
    for c_id in [0, 1, 2]:
        mc = lbl2 == c_id
        ax.scatter(
            Xp_m[mc, 0],
            Xp_m[mc, 1],
            s=10,
            c=palette_cl[c_id],
            alpha=0.65,
            label=f"{short_names[c_id]} (n={mc.sum()})",
            rasterized=True,
        )
    ax.set_title(
        f"Etapa 2: Grupo Principal n=12,836 (PCA2: {100*var_2.sum():.1f}% var.)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(f"PC1 Grupo Principal ({100*var_2[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 Grupo Principal ({100*var_2[1]:.1f}% var.)")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.4, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Partición DBSCAN en Dos Etapas: Preservación de Gigantes y Ruido de Frontera",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "dbscan_clusters_2etapas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_preprocessing(
    input_csv: Path,
    output_csv: Path,
    noise_csv: Path,
    eps1: float = 1.50,
    ms1: int = 20,
    eps2: float = 0.38,
    ms2: int = 50,
    models_dir: Path | None = None,
    fig_dir: Path = PROJECT_ROOT / "results" / "figuras",
    table_dir: Path = PROJECT_ROOT / "results" / "tablas",
    skip_diagnostics: bool = False,
) -> pd.DataFrame:
    """Ejecuta el pipeline completo de preprocesamiento y clustering DBSCAN."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    noise_csv.parent.mkdir(parents=True, exist_ok=True)
    models_out = models_dir if models_dir is not None else output_csv.parent
    models_out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df_clean, audit = load_and_clean_dataset(input_csv)
    fit_res = fit_two_stage_dbscan(
        df_clean, eps1=eps1, ms1=ms1, eps2=eps2, ms2=ms2, random_state=RANDOM_STATE
    )

    # Desglose de ruido (siempre calculado para etiquetar ruido_descartado.csv)
    noise_table, subcat = build_noise_breakdown_table(df_clean, fit_res)

    labels = fit_res["labels_combined"]
    valid_mask = labels != -1

    df_clusters = df_clean.loc[valid_mask].copy().reset_index(drop=True)
    df_clusters["cluster"] = labels[valid_mask].astype(np.int64)

    df_noise = df_clean.loc[~valid_mask].copy().reset_index(drop=True)
    df_noise["origen_ruido"] = fit_res["noise_origin"][~valid_mask]
    df_noise["subcategoria_ruido"] = subcat[~valid_mask]

    df_clusters.to_csv(output_csv, index=False)
    df_noise.to_csv(noise_csv, index=False)

    # Serialización exclusiva de transformadores del clustering (jamás usados en MLP)
    joblib.dump(fit_res["scaler1"], models_out / "scaler_etapa1_dbscan.joblib")
    joblib.dump(fit_res["scaler2"], models_out / "scaler_etapa2_dbscan.joblib")
    joblib.dump(fit_res["pca2"], models_out / "pca_etapa2_dbscan.joblib")

    print("=" * 72)
    print("FASE 1 — PREPROCESAMIENTO Y DBSCAN EN DOS ETAPAS COMPLETADO")
    print("=" * 72)
    print(
        f"Lectura cruda: {audit['raw_rows']} x {audit['raw_cols']} | "
        f"Columnas vacías eliminadas: {audit['dropped_all_nan_cols']}"
    )
    print(
        f"Duplicados exactos eliminados: {audit['exact_duplicates_removed']} | "
        f"Instancias únicas limpias: {audit['clean_rows']}"
    )
    print(
        f"Etapa 1 (RobustScaler 16D, eps1={eps1}, ms1={ms1}) -> "
        f"Principal={int(fit_res['main_mask'].sum())}, "
        f"Gigantes={int(fit_res['giant_mask'].sum())}, "
        f"Ruido1={int(fit_res['noise1_mask'].sum())}"
    )
    print(
        f"Etapa 2 (StdScaler+PCA2 [{100*fit_res['pca2_var'].sum():.2f}% var.], "
        f"eps2={eps2}, ms2={ms2}) -> k2={fit_res['k2']}, "
        f"Ruido2={int((fit_res['lbl2'] == -1).sum())}"
    )
    counts = df_clusters["cluster"].value_counts().sort_index().to_dict()
    print(
        f"Dataset limpio final: {len(df_clusters)} instancias en {len(counts)} clases "
        f"{counts} | Ruido total descartado: {len(df_noise)}"
    )

    if not skip_diagnostics:
        eda_df = compute_eda_summary(df_clean)
        eda_df.to_csv(table_dir / "eda_estadisticas_16vars.csv", index=False)

        grid1 = run_stage1_sensitivity_grid(fit_res["X_rob"], fit_res["lbl1"])
        grid1.to_csv(table_dir / "dbscan_sensibilidad_etapa1.csv", index=False)

        grid2 = run_stage2_sensitivity_grid(
            fit_res["X_main_pca2"], fit_res["X_main_std"], fit_res["lbl2"]
        )
        grid2.to_csv(table_dir / "dbscan_sensibilidad_etapa2.csv", index=False)

        log_comp = run_log_skewness_comparison(
            df_clean.loc[fit_res["main_mask"]], fit_res["lbl2"], ms2=ms2, eps2=eps2
        )
        log_comp.to_csv(table_dir / "dbscan_comparacion_asimetria_log.csv", index=False)

        stab_df = run_subsampling_stability(
            fit_res["X_rob"],
            fit_res["lbl1"],
            fit_res["X_main_pca2"],
            eps1=eps1,
            ms1=ms1,
            eps2=eps2,
            ms2=ms2,
            n_runs=20,
        )
        stab_df.to_csv(table_dir / "dbscan_estabilidad_submuestras.csv", index=False)

        noise_table.to_csv(table_dir / "desglose_ruido_dbscan.csv", index=False)

        profiles = build_cluster_profiles(df_clean, fit_res)
        profiles.to_csv(table_dir / "perfil_clusters_4clases.csv", index=False)

        metrics_df = build_multi_space_metrics(fit_res)
        metrics_df.to_csv(table_dir / "dbscan_metricas_espacios.csv", index=False)

        generate_eda_figures(df_clean, fig_dir)
        knee1, knee2 = generate_k_distance_figure(
            fit_res, eps1=eps1, ms1=ms1, eps2=eps2, ms2=ms2, fig_dir=fig_dir
        )
        _plot_sensitivity_heatmaps(
            grid1,
            noise_col="pct_noise",
            sil_col="silhouette_excl_noise",
            ref_ms=ms1,
            ref_eps=eps1,
            title="Etapa 1 (RobustScaler 16D): Mapa de Sensibilidad eps x min_samples",
            out_path=fig_dir / "dbscan_heatmap_etapa1.png",
            figsize=(12.8, 9.2),
        )
        _plot_sensitivity_heatmaps(
            grid2,
            noise_col="pct_noise_main",
            sil_col="silhouette_pca2_excl_noise",
            ref_ms=ms2,
            ref_eps=eps2,
            title="Etapa 2 (StdScaler+PCA2 Grupo Principal): Mapa de Sensibilidad eps x min_samples",
            out_path=fig_dir / "dbscan_heatmap_etapa2.png",
            figsize=(15.5, 10.0),
        )
        generate_clusters_2d_figure(fit_res, subcat=subcat, fig_dir=fig_dir)

        print(
            f"Codos k-distance: Etapa 1 (k={ms1}) = {knee1[0]:.4f} (p{knee1[1]:.2f}%) | "
            f"Etapa 2 (k={ms2}) = {knee2[0]:.4f} (p{knee2[1]:.2f}%, colapsa a k=1; "
            f"eps2={eps2} elegido en meseta estable)"
        )
        print(f"Tablas exportadas en: {table_dir}/ | Figuras exportadas en: {fig_dir}/")

    return df_clusters


def parse_args() -> argparse.Namespace:
    """Configura argumentos de línea de comandos para preprocesamiento DBSCAN."""
    parser = argparse.ArgumentParser(
        description="Preprocesamiento y clustering DBSCAN en dos etapas."
    )
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "Dataset.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "dataset_limpio_clusters.csv",
    )
    parser.add_argument(
        "--noise-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ruido_descartado.csv",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=PROJECT_ROOT / "models"
    )
    parser.add_argument("--eps1", type=float, default=1.50)
    parser.add_argument("--ms1", type=int, default=20)
    parser.add_argument("--eps2", type=float, default=0.38)
    parser.add_argument("--ms2", type=int, default=50)
    parser.add_argument(
        "--fig-dir", type=Path, default=PROJECT_ROOT / "results" / "figuras"
    )
    parser.add_argument(
        "--table-dir", type=Path, default=PROJECT_ROOT / "results" / "tablas"
    )
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    run_preprocessing(
        input_csv=args.input,
        output_csv=args.output,
        noise_csv=args.noise_output,
        eps1=args.eps1,
        ms1=args.ms1,
        eps2=args.eps2,
        ms2=args.ms2,
        models_dir=args.models_dir,
        fig_dir=args.fig_dir,
        table_dir=args.table_dir,
        skip_diagnostics=args.skip_diagnostics,
    )


if __name__ == "__main__":
    main()
