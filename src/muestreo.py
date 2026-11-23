"""Muestreo estratificado, validación de representatividad y ablación de balanceo (Pregunta 2).

Realiza:
  1. Carga de ``dataset_limpio_clusters.csv`` o invocación directa de
     ``preprocesameinto.py`` sobre ``Dataset.csv``.
  2. Partición estratificada disjunta 70 % train / 15 % validation / 15 % test.
  3. Pruebas estadísticas de representatividad (Kolmogorov-Smirnov por variable
     y Chi-cuadrada de homogeneidad de clases).
  4. Validación cruzada estratificada (StratifiedKFold k=5 principal y contraste k=10).
  5. Generación de variantes de balanceo (duplicación simple y SMOTE) EXCLUSIVAMENTE
     sobre ``train.csv`` para ablación controlada en ``validation.csv`` (nunca en CV).
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import preprocesameinto as prep

RANDOM_STATE: int = 42
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS: list[str] = prep.FEATURE_COLUMNS


def load_clustered_sample(
    input_path: Path,
    clean_csv: Path = PROJECT_ROOT / "data" / "processed" / "dataset_limpio_clusters.csv",
    noise_csv: Path = PROJECT_ROOT / "data" / "processed" / "ruido_descartado.csv",
    models_dir: Path = PROJECT_ROOT / "models",
    fig_dir: Path = PROJECT_ROOT / "results" / "figuras",
    table_dir: Path = PROJECT_ROOT / "results" / "tablas",
) -> pd.DataFrame:
    """Carga el dataset etiquetado o lo genera desde la muestra original cruda."""
    df_probe = pd.read_csv(input_path, encoding="utf-8-sig", nrows=5)
    if "cluster" in df_probe.columns:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    else:
        df = prep.run_preprocessing(
            input_csv=input_path,
            output_csv=clean_csv,
            noise_csv=noise_csv,
            eps1=1.50,
            ms1=20,
            eps2=0.38,
            ms2=50,
            models_dir=models_dir,
            fig_dir=fig_dir,
            table_dir=table_dir,
            skip_diagnostics=True,
        )

    expected_cols = FEATURE_COLUMNS + ["cluster"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Columnas inesperadas en muestra etiquetada: {list(df.columns)}")
    df["cluster"] = df["cluster"].astype(np.int64)
    if df.isna().any().any():
        raise AssertionError("Existen valores NaN en la muestra etiquetada.")
    return df


def stratified_split_70_15_15(
    df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide de forma estratificada y disjunta en 70 % train, 15 % val, 15 % test."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["cluster"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["cluster"],
        random_state=random_state,
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)

    for sub in (train_df, val_df, test_df):
        assert not sub.isna().any().any()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_representativeness_tests(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa representatividad estadística mediante pruebas KS y Chi-cuadrada."""
    ks_rows: list[dict[str, Any]] = []
    for col in FEATURE_COLUMNS:
        x_tr = train_df[col].to_numpy(dtype=float)
        x_va = val_df[col].to_numpy(dtype=float)
        x_te = test_df[col].to_numpy(dtype=float)

        st_tv, p_tv = ks_2samp(x_tr, x_va, alternative="two-sided")
        st_tt, p_tt = ks_2samp(x_tr, x_te, alternative="two-sided")
        st_vt, p_vt = ks_2samp(x_va, x_te, alternative="two-sided")
        ks_rows.append(
            {
                "feature": col,
                "ks_stat_tr_val": float(st_tv),
                "p_val_tr_val": float(p_tv),
                "ks_stat_tr_te": float(st_tt),
                "p_val_tr_te": float(p_tt),
                "ks_stat_val_te": float(st_vt),
                "p_val_val_te": float(p_vt),
                "todas_p_gt_005": bool(min(p_tv, p_tt, p_vt) > 0.05),
            }
        )
    ks_df = pd.DataFrame(ks_rows)

    classes = sorted(int(c) for c in train_df["cluster"].unique())
    contingency = np.array(
        [
            [int((split_df["cluster"] == c).sum()) for c in classes]
            for split_df in (train_df, val_df, test_df)
        ]
    )
    chi2_stat, p_val, dof, _ = chi2_contingency(contingency)
    chi2_df = pd.DataFrame(
        [
            {
                "particion": name,
                **{f"clase_{c}": int(contingency[i, j]) for j, c in enumerate(classes)},
                "total": int(contingency[i].sum()),
                "chi2_stat": float(chi2_stat),
                "dof": int(dof),
                "p_value": float(p_val),
            }
            for i, name in enumerate(["train", "validation", "test"])
        ]
    )
    return ks_df, chi2_df


def verify_stratified_kfold(train_df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Verifica proporciones por partición y folds de StratifiedKFold (k=5 y k=10)."""
    y = train_df["cluster"].to_numpy(dtype=np.int64)
    classes = sorted(int(c) for c in np.unique(y))
    rows: list[dict[str, Any]] = []

    for k_splits in (5, 10):
        skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=random_state)
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_df[FEATURE_COLUMNS], y), start=1):
            y_va = y[va_idx]
            y_tr = y[tr_idx]
            row: dict[str, Any] = {
                "esquema_cv": f"StratifiedKFold(k={k_splits})",
                "k_splits": k_splits,
                "fold": fold_idx,
                "n_train_fold": int(len(tr_idx)),
                "n_val_fold": int(len(va_idx)),
            }
            for c in classes:
                cnt_va = int((y_va == c).sum())
                cnt_tr = int((y_tr == c).sum())
                row[f"val_n_c{c}"] = cnt_va
                row[f"val_pct_c{c}"] = 100.0 * cnt_va / len(y_va)
                row[f"train_n_c{c}"] = cnt_tr
            rows.append(row)
    return pd.DataFrame(rows)


def random_oversample_minorities(
    train_df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Duplicación simple aleatoria con reemplazo de las clases minoritarias en train."""
    rng = np.random.default_rng(random_state)
    counts = train_df["cluster"].value_counts()
    max_count = int(counts.max())

    parts: list[pd.DataFrame] = []
    for cls_id, cls_count in counts.sort_index().items():
        sub = train_df.loc[train_df["cluster"] == cls_id].reset_index(drop=True)
        parts.append(sub)
        n_needed = max_count - int(cls_count)
        if n_needed > 0:
            dup_idx = rng.integers(0, len(sub), size=n_needed)
            parts.append(sub.iloc[dup_idx].reset_index(drop=True))

    balanced = pd.concat(parts, ignore_index=True)
    perm = rng.permutation(len(balanced))
    return balanced.iloc[perm].reset_index(drop=True)


def smote_oversample(
    train_df: pd.DataFrame, k_neighbors: int = 5, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """SMOTE nativo en NumPy + NearestNeighbors sobre ``train`` exclusivamente.

    Busca los k vecinos en espacio estandarizado sobre ``train`` para evitar
    distorsión por magnitud de ``Area``, e interpola en la escala original.
    """
    rng = np.random.default_rng(random_state)
    X = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train_df["cluster"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())

    synth_X_list: list[np.ndarray] = [X]
    synth_y_list: list[np.ndarray] = [y]

    for cls_id, cls_count in zip(classes, counts):
        n_needed = max_count - int(cls_count)
        if n_needed <= 0:
            continue
        cls_mask = y == cls_id
        X_cls_orig = X[cls_mask]
        X_cls_scaled = X_scaled[cls_mask]

        k_eff = min(k_neighbors, len(X_cls_orig) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1).fit(X_cls_scaled)
        _, indices = nn.kneighbors(X_cls_scaled)
        neighbor_pool = indices[:, 1:]  # excluye el punto mismo

        base_choices = rng.integers(0, len(X_cls_orig), size=n_needed)
        nn_choices = rng.integers(0, k_eff, size=n_needed)
        alphas = rng.uniform(0.0, 1.0, size=(n_needed, 1))

        base_pts = X_cls_orig[base_choices]
        nn_pts = X_cls_orig[neighbor_pool[base_choices, nn_choices]]
        synth_pts = base_pts + alphas * (nn_pts - base_pts)

        synth_X_list.append(synth_pts)
        synth_y_list.append(np.full(n_needed, cls_id, dtype=np.int64))

    X_all = np.vstack(synth_X_list)
    y_all = np.concatenate(synth_y_list)
    perm = rng.permutation(len(y_all))

    out_df = pd.DataFrame(X_all[perm], columns=FEATURE_COLUMNS)
    out_df["cluster"] = y_all[perm].astype(np.int64)
    return out_df


def build_split_summary_table(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dup_df: pd.DataFrame,
    smote_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye tabla resumen de conteos y proporciones en splits y ablación."""
    datasets: list[tuple[str, pd.DataFrame]] = [
        ("Limpio Total (DBSCAN 2 etapas)", clean_df),
        ("Train (70% - sin balanceo)", train_df),
        ("Validation (15% - intacto)", val_df),
        ("Test (15% - intacto)", test_df),
        ("Train Duplicacion Simple (solo ablacion)", dup_df),
        ("Train SMOTE k=5 (solo ablacion)", smote_df),
    ]
    classes = sorted(int(c) for c in clean_df["cluster"].unique())
    rows: list[dict[str, Any]] = []
    for name, d in datasets:
        n = len(d)
        row: dict[str, Any] = {"subconjunto": name, "n_total": int(n)}
        for c in classes:
            cnt = int((d["cluster"] == c).sum())
            row[f"n_c{c}"] = cnt
            row[f"pct_c{c}"] = 100.0 * cnt / n
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sampling_balance(
    summary_df: pd.DataFrame, folds_df: pd.DataFrame, out_path: Path
) -> None:
    """Genera gráfica comparativa de proporciones estratificadas y brazos de balanceo."""
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6))

    colors = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea"]
    class_labels = [
        "Clase 0 (Peq.-med.)",
        "Clase 1 (Med.-grandes)",
        "Clase 2 (Alargados)",
        "Clase 3 (Gigantes)",
    ]

    # Panel izquierdo: proporciones (%) en Limpio, Train, Val, Test y media Folds k=5
    ax = axes[0]
    k5 = folds_df[folds_df["k_splits"] == 5]
    fold_pcts = [float(k5[f"val_pct_c{c}"].mean()) for c in range(4)]

    split_rows = summary_df.iloc[:4]
    names_left = [
        f"Limpio\n({int(split_rows.iloc[0]['n_total']):,})",
        f"Train 70%\n({int(split_rows.iloc[1]['n_total']):,})",
        f"Val 15%\n({int(split_rows.iloc[2]['n_total']):,})",
        f"Test 15%\n({int(split_rows.iloc[3]['n_total']):,})",
        "CV k=5\n(media fold)",
    ]
    pct_matrix = np.vstack(
        [
            split_rows[[f"pct_c{c}" for c in range(4)]].to_numpy(dtype=float),
            np.array([fold_pcts]),
        ]
    )

    x = np.arange(len(names_left))
    bottom = np.zeros(len(names_left))
    for c in range(4):
        vals = pct_matrix[:, c]
        bars = ax.bar(x, vals, bottom=bottom, width=0.58, color=colors[c], label=class_labels[c])
        for i, v in enumerate(vals):
            if v >= 3.5:
                ax.text(
                    x[i],
                    bottom[i] + v / 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names_left, fontsize=9)
    ax.set_ylabel("Proporción de clases (%)")
    ax.set_ylim(0, 121)
    ax.set_title("Preservación de Estratificación (70/15/15 y CV k=5)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper center", fontsize=8, ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.35)

    # Panel derecho: conteos absolutos en los 3 brazos de ablación sobre Train
    ax = axes[1]
    arm_rows = summary_df.iloc[[1, 4, 5]]
    arm_names = [
        f"Sin balanceo\n(n={int(arm_rows.iloc[0]['n_total']):,})",
        f"Duplicación simple\n(n={int(arm_rows.iloc[1]['n_total']):,})",
        f"SMOTE (k=5)\n(n={int(arm_rows.iloc[2]['n_total']):,})",
    ]
    x_a = np.arange(len(arm_names))
    width = 0.18
    for c in range(4):
        vals = arm_rows[f"n_c{c}"].to_numpy(dtype=float)
        offset = (c - 1.5) * width
        bars = ax.bar(x_a + offset, vals, width=width, color=colors[c], label=class_labels[c])
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + 110,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                rotation=90 if h > 4000 else 0,
            )

    ax.set_xticks(x_a)
    ax.set_xticklabels(arm_names, fontsize=9.5)
    ax.set_ylabel("Número de instancias en entrenamiento")
    ax.set_ylim(0, float(arm_rows["n_c0"].max()) * 1.38)
    ax.set_title("Ablación de Balanceo Exclusiva sobre Train (Sin Fuga)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper center", fontsize=8, ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle(
        "Muestreo Estratificado 70/15/15 y Estrategia de Balanceo Controlada",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sampling(
    input_path: Path = PROJECT_ROOT / "data" / "raw" / "Dataset.csv",
    train_out: Path = PROJECT_ROOT / "data" / "processed" / "train.csv",
    val_out: Path = PROJECT_ROOT / "data" / "processed" / "validation.csv",
    test_out: Path = PROJECT_ROOT / "data" / "processed" / "test.csv",
    dup_out: Path = PROJECT_ROOT / "data" / "processed" / "train_dup.csv",
    smote_out: Path = PROJECT_ROOT / "data" / "processed" / "train_smote.csv",
    models_dir: Path = PROJECT_ROOT / "models",
    fig_dir: Path = PROJECT_ROOT / "results" / "figuras",
    table_dir: Path = PROJECT_ROOT / "results" / "tablas",
) -> None:
    """Ejecuta el flujo completo de muestreo estratificado y pruebas estadísticas."""
    for p in (train_out.parent, val_out.parent, test_out.parent, dup_out.parent, smote_out.parent):
        p.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    clean_df = load_clustered_sample(
        input_path,
        clean_csv=train_out.parent / "dataset_limpio_clusters.csv",
        noise_csv=train_out.parent / "ruido_descartado.csv",
        models_dir=models_dir,
        fig_dir=fig_dir,
        table_dir=table_dir,
    )
    train_df, val_df, test_df = stratified_split_70_15_15(clean_df, random_state=RANDOM_STATE)

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    ks_df, chi2_df = compute_representativeness_tests(train_df, val_df, test_df)
    ks_df.to_csv(table_dir / "pruebas_representatividad_ks.csv", index=False)
    chi2_df.to_csv(table_dir / "prueba_chi2_clases.csv", index=False)

    folds_df = verify_stratified_kfold(train_df, random_state=RANDOM_STATE)
    folds_df.to_csv(table_dir / "muestreo_estratificado_folds.csv", index=False)

    dup_df = random_oversample_minorities(train_df, random_state=RANDOM_STATE)
    smote_df = smote_oversample(train_df, k_neighbors=5, random_state=RANDOM_STATE)

    dup_df.to_csv(dup_out, index=False)
    smote_df.to_csv(smote_out, index=False)

    n_c0_tr = int((train_df["cluster"] == 0).sum())
    assert len(dup_df) == 4 * n_c0_tr and len(smote_df) == 4 * n_c0_tr

    summary_df = build_split_summary_table(
        clean_df, train_df, val_df, test_df, dup_df, smote_df
    )
    summary_df.to_csv(table_dir / "resumen_muestreo_y_balanceo.csv", index=False)
    plot_sampling_balance(summary_df, folds_df, fig_dir / "muestreo_balance_clases.png")

    print("=" * 72)
    print("FASE 2 — MUESTREO ESTRATIFICADO Y ABLACIÓN DE BALANCEO COMPLETADO")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(
        f"\nPrueba Chi2 de homogeneidad de clases (Train/Val/Test): "
        f"chi2={chi2_df['chi2_stat'].iloc[0]:.4f}, dof={chi2_df['dof'].iloc[0]}, "
        f"p={chi2_df['p_value'].iloc[0]:.4f}"
    )
    min_ks_p = float(ks_df[["p_val_tr_val", "p_val_tr_te", "p_val_val_te"]].min().min())
    print(
        f"Pruebas KS (16 variables x 3 pares = 48 pruebas): "
        f"p-valor mínimo = {min_ks_p:.4f} | Todas > 0.05: {bool(min_ks_p > 0.05)}"
    )
    print("\nProporciones exactas por fold en StratifiedKFold(k=5):")
    k5 = folds_df[folds_df["k_splits"] == 5][
        ["fold", "n_train_fold", "n_val_fold", "val_n_c0", "val_n_c1", "val_n_c2", "val_n_c3",
         "val_pct_c0", "val_pct_c1", "val_pct_c2", "val_pct_c3"]
    ]
    print(k5.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Configura argumentos de línea de comandos para ``muestreo.py``."""
    parser = argparse.ArgumentParser(description="Muestreo estratificado 70/15/15 y CV.")
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "Dataset.csv"
    )
    parser.add_argument(
        "--train-out", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train.csv"
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "validation.csv",
    )
    parser.add_argument(
        "--test-out", type=Path, default=PROJECT_ROOT / "data" / "processed" / "test.csv"
    )
    parser.add_argument(
        "--dup-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train_dup.csv",
    )
    parser.add_argument(
        "--smote-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "train_smote.csv",
    )
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--fig-dir", type=Path, default=PROJECT_ROOT / "results" / "figuras"
    )
    parser.add_argument(
        "--table-dir", type=Path, default=PROJECT_ROOT / "results" / "tablas"
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    run_sampling(
        input_path=args.input,
        train_out=args.train_out,
        val_out=args.val_out,
        test_out=args.test_out,
        dup_out=args.dup_out,
        smote_out=args.smote_out,
        models_dir=args.models_dir,
        fig_dir=args.fig_dir,
        table_dir=args.table_dir,
    )


if __name__ == "__main__":
    main()
