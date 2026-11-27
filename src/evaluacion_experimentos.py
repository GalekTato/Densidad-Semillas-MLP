"""Evaluación experimental, control de overfitting y variación estadística (Pregunta 4).

Implementa:
  1. Visualización de la evolución de hiperparámetros en GridSearchCV.
  2. Evaluación de fidelidad a DBSCAN en Train/Validation y UNA SOLA VEZ en
     ``test.csv`` únicamente para la variante de balanceo elegida en
     ``validation.csv`` + modelos base (Regresión Logística y Centroide).
  3. Curvas de aprendizaje, curvas de pérdida/validación y contraste con el
     experimento sin ``early_stopping``.
  4. Matriz de confusión 4x4, métricas multiclase y curvas ROC-AUC OvR en test.
  5. Evaluación fuera de distribución (OOD) sobre ``ruido_descartado.csv``
     mediante entropía de Shannon.
  6. Variación estadística estimada:
       - 20 semillas de inicialización de pesos (0..19) sobre validation.
       - RepeatedStratifiedKFold (5x10 = 50 corridas) + prueba t corregida de
         Nadeau-Bengio sobre train+validation (solo varianza/comparación).
       - Bootstrap estratificado (B = 1,000) sobre ``test.csv``.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, learning_curve
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import entrenamiento as ent
import preprocesameinto as prep

RANDOM_STATE: int = 42
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS: list[str] = prep.FEATURE_COLUMNS
CLASS_NAMES: list[str] = [
    "Clase 0 (Peq.-med.)",
    "Clase 1 (Med.-grandes)",
    "Clase 2 (Alargados)",
    "Clase 3 (Gigantes)",
]


def _cfg_signature(cfg: dict[str, Any]) -> tuple[Any, ...]:
    """Firma hashable para deduplicar configuraciones MLP."""
    return (
        tuple(int(x) for x in cfg["hidden_layer_sizes"]),
        str(cfg["activation"]),
        str(cfg["solver"]),
        round(float(cfg["alpha"]), 8),
        round(float(cfg["learning_rate_init"]), 8),
        str(cfg.get("learning_rate", "constant")),
        round(float(cfg.get("momentum", 0.9)), 4),
        bool(cfg.get("nesterovs_momentum", True)),
        int(cfg.get("batch_size", 128)),
    )


def _cfg_short_label(cfg: dict[str, Any]) -> str:
    """Etiqueta compacta legible de una configuración MLP."""
    arch = str(tuple(int(x) for x in cfg["hidden_layer_sizes"]))
    sol = str(cfg["solver"]).upper()
    if sol == "SGD":
        m = float(cfg.get("momentum", 0.9))
        nest = "N" if cfg.get("nesterovs_momentum", True) else "sN"
        sol_s = f"SGD(m={m:g},{nest})"
    else:
        sol_s = "Adam"
    return f"{arch} {cfg['activation']} {sol_s} a={cfg['alpha']:g}"


def plot_hyperparameter_evolution(
    results_df: pd.DataFrame, out_path: Path
) -> None:
    """Grafica la evolución de F1-Macro CV según cada hiperparámetro en Etapa A."""
    df_a = results_df[results_df["etapa_busqueda"] == "A_gruesa"].copy()
    if df_a.empty:
        df_a = results_df.copy()

    def _opt_branch(row: pd.Series) -> str:
        if row["param_clf__solver"] == "adam":
            return "Adam"
        m = float(row["param_clf__momentum"])
        nest = bool(row["param_clf__nesterovs_momentum"])
        if abs(m - 0.5) < 1e-6:
            return "SGD m=0.5\n(Nesterov)"
        return "SGD m=0.9\n(Nesterov)" if nest else "SGD m=0.9\n(Sin Nesterov)"

    df_a["opt_branch"] = df_a.apply(_opt_branch, axis=1)
    arch_order_df = (
        df_a[["param_clf__hidden_layer_sizes", "n_params"]]
        .drop_duplicates()
        .sort_values("n_params")
    )
    arch_order = list(arch_order_df["param_clf__hidden_layer_sizes"])
    arch_labels = [
        f"{a}\n({int(p):,} p)"
        for a, p in zip(arch_order_df["param_clf__hidden_layer_sizes"], arch_order_df["n_params"])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.5))

    # Panel 1: Arquitectura
    ax = axes[0, 0]
    sns.boxplot(
        data=df_a,
        x="param_clf__hidden_layer_sizes",
        y="mean_test_f1_macro",
        order=arch_order,
        color="#93c5fd",
        width=0.55,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_a,
        x="param_clf__hidden_layer_sizes",
        y="mean_test_f1_macro",
        order=arch_order,
        color="#1d4ed8",
        alpha=0.45,
        size=3.5,
        ax=ax,
    )
    ax.set_xticks(range(len(arch_order)))
    ax.set_xticklabels(arch_labels, fontsize=8.8)
    ax.set_title("Efecto de Arquitectura y Número de Parámetros", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Capas ocultas (parámetros totales)")
    ax.set_ylabel("F1-Macro CV (5-fold)")
    ax.grid(axis="y", alpha=0.35)

    # Panel 2: Optimizador / Momentum / Nesterov
    ax = axes[0, 1]
    opt_order = ["Adam", "SGD m=0.9\n(Nesterov)", "SGD m=0.9\n(Sin Nesterov)", "SGD m=0.5\n(Nesterov)"]
    sns.boxplot(
        data=df_a,
        x="opt_branch",
        y="mean_test_f1_macro",
        order=opt_order,
        color="#86efac",
        width=0.55,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_a,
        x="opt_branch",
        y="mean_test_f1_macro",
        order=opt_order,
        color="#15803d",
        alpha=0.45,
        size=3.5,
        ax=ax,
    )
    ax.set_title("Efecto del Optimizador y Momentum (Adam vs. SGD)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Rama de optimización")
    ax.set_ylabel("F1-Macro CV (5-fold)")
    ax.grid(axis="y", alpha=0.35)

    # Panel 3: Heatmap activation x learning_rate_init
    ax = axes[1, 0]
    piv = df_a.pivot_table(
        index="param_clf__activation",
        columns="param_clf__learning_rate_init",
        values="mean_test_f1_macro",
        aggfunc="mean",
    )
    piv_max = df_a.pivot_table(
        index="param_clf__activation",
        columns="param_clf__learning_rate_init",
        values="mean_test_f1_macro",
        aggfunc="max",
    )
    annot_txt = np.empty(piv.shape, dtype=object)
    for r_i in range(piv.shape[0]):
        for c_i in range(piv.shape[1]):
            annot_txt[r_i, c_i] = f"Media: {piv.iloc[r_i, c_i]:.4f}\nMáx: {piv_max.iloc[r_i, c_i]:.4f}"
    sns.heatmap(
        piv,
        annot=annot_txt,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.8,
        cbar_kws={"label": "F1-Macro CV medio"},
        ax=ax,
    )
    ax.set_title("Interacción Activación × Tasa de Aprendizaje Inicial", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("learning_rate_init")
    ax.set_ylabel("activation")

    # Panel 4: Regularización L2 (alpha) y brecha Train - CV
    ax = axes[1, 1]
    alpha_grp = (
        df_a.groupby("param_clf__alpha")
        .agg(
            cv_mean=("mean_test_f1_macro", "mean"),
            cv_std=("mean_test_f1_macro", "std"),
            tr_mean=("mean_train_f1_macro", "mean"),
            tr_std=("mean_train_f1_macro", "std"),
            cv_max=("mean_test_f1_macro", "max"),
        )
        .reset_index()
    )
    x_pos = np.arange(len(alpha_grp))
    ax.errorbar(
        x_pos - 0.08,
        alpha_grp["tr_mean"],
        yerr=alpha_grp["tr_std"],
        fmt="o-",
        capsize=5,
        color="#f59e0b",
        label="Train F1-Macro (media ± DE)",
    )
    ax.errorbar(
        x_pos + 0.08,
        alpha_grp["cv_mean"],
        yerr=alpha_grp["cv_std"],
        fmt="s-",
        capsize=5,
        color="#2563eb",
        label="CV F1-Macro (media ± DE)",
    )
    ax.plot(x_pos, alpha_grp["cv_max"], "D--", color="#059669", label="CV F1-Macro máximo")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"alpha={a:g}" for a in alpha_grp["param_clf__alpha"]], fontsize=9.5)
    ax.set_title("Efecto de Regularización L2 (alpha) sobre Generalización", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Coeficiente de regularización L2 (alpha)")
    ax.set_ylabel("F1-Macro")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax.grid(True, alpha=0.35)

    fig.suptitle(
        "Evolución de Resultados en GridSearchCV (Etapa A Factorial, 240 Configuraciones)",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def evaluate_fidelity_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    top5_payload: dict[str, Any],
    decision: dict[str, Any],
    base_dir: Path,
    table_dir: Path,
    fig_dir: Path,
) -> tuple[pd.DataFrame, Pipeline]:
    """Evalúa modelos en Train/Val y UNA SOLA VEZ en Test (solo variante elegida)."""
    models_dir = base_dir / "models"
    data_dir = base_dir / "data" / "processed"
    chosen_arm = str(decision["variante_elegida"])
    model_final: Pipeline = joblib.load(models_dir / "mejor_modelo_mlp_final.joblib")
    model_1se: Pipeline = joblib.load(models_dir / "mejor_modelo_mlp.joblib")
    model_top1: Pipeline = joblib.load(models_dir / "mejor_modelo_mlp_top1.joblib")
    model_logreg: Pipeline = joblib.load(models_dir / "baseline_logreg.joblib")
    model_nc: Pipeline = joblib.load(models_dir / "baseline_nearest_centroid.joblib")

    arm_files = {
        "sin_balanceo": ("Sin balanceo", data_dir / "train.csv"),
        "duplicacion_simple": ("Duplicación simple", data_dir / "train_dup.csv"),
        "smote": ("SMOTE k=5", data_dir / "train_smote.csv"),
    }

    def _eval_split(pipe: Pipeline, X_s: np.ndarray, y_s: np.ndarray) -> dict[str, float]:
        pred = pipe.predict(X_s)
        roc = (
            float(roc_auc_score(y_s, pipe.predict_proba(X_s), multi_class="ovr"))
            if hasattr(pipe.named_steps["clf"], "predict_proba")
            else np.nan
        )
        return {
            "acc": float(accuracy_score(y_s, pred)),
            "bacc": float(balanced_accuracy_score(y_s, pred)),
            "f1_macro": float(f1_score(y_s, pred, average="macro")),
            "f1_weighted": float(f1_score(y_s, pred, average="weighted")),
            "roc_auc_ovr": roc,
        }

    rows: list[dict[str, Any]] = []

    def _add_row(
        name: str, role: str, pipe: Pipeline, eval_test: bool, n_params: int | float = np.nan
    ) -> None:
        m_tr = _eval_split(pipe, X_train, y_train)
        m_va = _eval_split(pipe, X_val, y_val)
        m_te = (
            _eval_split(pipe, X_test, y_test)
            if eval_test
            else {"acc": np.nan, "bacc": np.nan, "f1_macro": np.nan, "f1_weighted": np.nan, "roc_auc_ovr": np.nan}
        )
        rows.append(
            {
                "modelo": name,
                "rol": role,
                "n_params": n_params,
                "train_f1_macro": m_tr["f1_macro"],
                "train_accuracy": m_tr["acc"],
                "val_f1_macro": m_va["f1_macro"],
                "val_accuracy": m_va["acc"],
                "val_balanced_accuracy": m_va["bacc"],
                "val_roc_auc_ovr": m_va["roc_auc_ovr"],
                "gap_f1_train_val": m_tr["f1_macro"] - m_va["f1_macro"],
                "evaluado_en_test": eval_test,
                "test_f1_macro": m_te["f1_macro"],
                "test_accuracy": m_te["acc"],
                "test_balanced_accuracy": m_te["bacc"],
                "test_f1_weighted": m_te["f1_weighted"],
                "test_roc_auc_ovr": m_te["roc_auc_ovr"],
            }
        )

    one_se_cfg = top5_payload["ganador_1se"]
    top1_cfg = top5_payload["top1_absoluto"]

    # 1. Modelos evaluados en Test (solo variante elegida en Val + Top1 contraste + Baselines)
    _add_row(
        f"MLP Ganador 1-SE ({arm_files[chosen_arm][0]} — elegido en Val)",
        "Modelo Final Elegido",
        model_final,
        eval_test=True,
        n_params=int(one_se_cfg["n_params"]),
    )
    if _cfg_signature(top1_cfg) != _cfg_signature(one_se_cfg) or chosen_arm != "sin_balanceo":
        _add_row(
            f"MLP Top-1 Absoluto (sin balanceo, contraste)",
            "Contraste Top-1",
            model_top1,
            eval_test=True,
            n_params=int(top1_cfg["n_params"]),
        )
    _add_row(
        "Base: Regresión Logística (L2, C=1.0)",
        "Baseline Lineal",
        model_logreg,
        eval_test=True,
        n_params=16 * 4 + 4,
    )
    _add_row(
        "Base: Centroide Más Cercano",
        "Baseline Distancia",
        model_nc,
        eval_test=True,
        n_params=16 * 4,
    )

    # 2. Variantes de balanceo NO elegidas en validation (NUNCA evaluadas en test)
    for arm_key, (arm_desc, arm_path) in arm_files.items():
        if arm_key == chosen_arm:
            continue
        df_arm = pd.read_csv(arm_path)
        pipe_arm = ent.build_mlp_pipeline_from_params(
            one_se_cfg, random_state=RANDOM_STATE, early_stopping=True, max_iter=1000
        ).fit(df_arm[FEATURE_COLUMNS].to_numpy(dtype=float), df_arm["cluster"].to_numpy(dtype=np.int64))
        _add_row(
            f"Ablación 1-SE ({arm_desc} — descartado en Val)",
            "Ablación No Elegida",
            pipe_arm,
            eval_test=False,
            n_params=int(one_se_cfg["n_params"]),
        )

    # 3. Resto del Top-5 de la rejilla (solo Train y Validation, no Test)
    seen_sigs = {_cfg_signature(top1_cfg)}
    if chosen_arm == "sin_balanceo":
        seen_sigs.add(_cfg_signature(one_se_cfg))
    for cfg in top5_payload["top5"]:
        sig = _cfg_signature(cfg)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        pipe_t = ent.build_mlp_pipeline_from_params(
            cfg, random_state=RANDOM_STATE, early_stopping=True, max_iter=1000
        ).fit(X_train, y_train)
        _add_row(
            f"Top-{cfg['rank_grid']} Rejilla ({_cfg_short_label(cfg)})",
            "Top-5 Rejilla",
            pipe_t,
            eval_test=False,
            n_params=int(cfg["n_params"]),
        )

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(table_dir / "comparativa_modelos_fidelidad.csv", index=False)

    # Figura comparativa de generalización Train / Val / Test
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    ax = axes[0]
    df_t = df_comp[df_comp["evaluado_en_test"]].copy()
    short_names = []
    for m in df_t["modelo"]:
        if "Ganador 1-SE" in m:
            short_names.append("MLP 1-SE\n(Final)")
        elif "Top-1" in m:
            short_names.append("MLP Top-1\n(Contraste)")
        elif "Regresión" in m:
            short_names.append("Base:\nLogReg")
        else:
            short_names.append("Base:\nCentroide")
    x = np.arange(len(df_t))
    w = 0.25
    ax.bar(x - w, df_t["train_f1_macro"], width=w, color="#60a5fa", label="Train")
    ax.bar(x, df_t["val_f1_macro"], width=w, color="#2563eb", label="Validation")
    ax.bar(x + w, df_t["test_f1_macro"], width=w, color="#059669", label="Test (única eval.)")
    for idx_b, row in enumerate(df_t.itertuples()):
        ax.text(x[idx_b] - w, row.train_f1_macro + 0.004, f"{row.train_f1_macro:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
        ax.text(x[idx_b], row.val_f1_macro + 0.004, f"{row.val_f1_macro:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
        ax.text(x[idx_b] + w, row.test_f1_macro + 0.004, f"{row.test_f1_macro:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    ax.set_ylim(max(0.5, float(df_t["test_f1_macro"].min()) - 0.12), 1.07)
    ax.set_ylabel("F1-Macro (fidelidad a partición DBSCAN)")
    ax.set_title("Generalización Train / Validation / Test vs. Modelos Base", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    y_pos = np.arange(len(df_comp))
    labels_r = []
    for m in df_comp["modelo"]:
        if "Ganador 1-SE" in m:
            labels_r.append("MLP 1-SE (Sin balanceo, final)")
        elif "Top-1 Absoluto" in m:
            labels_r.append("MLP Top-1 Absoluto (contraste)")
        elif "Regresión Logística" in m:
            labels_r.append("Base: Regresión Logística")
        elif "Centroide" in m:
            labels_r.append("Base: Centroide Más Cercano")
        elif "Duplicación" in m:
            labels_r.append("Ablación 1-SE: Duplicación (no test)")
        elif "SMOTE" in m:
            labels_r.append("Ablación 1-SE: SMOTE k=5 (no test)")
        else:
            labels_r.append(m.split(" (")[0] + " (no test)")

    ax.barh(y_pos - 0.18, df_comp["train_f1_macro"], height=0.34, color="#93c5fd", label="Train F1-Macro")
    ax.barh(y_pos + 0.18, df_comp["val_f1_macro"], height=0.34, color="#1d4ed8", label="Validation F1-Macro")
    for i, r in enumerate(df_comp.itertuples()):
        ax.text(
            max(r.train_f1_macro, r.val_f1_macro) + 0.0025,
            y_pos[i],
            f"Δ={r.gap_f1_train_val:+.4f}",
            va="center",
            fontsize=8,
            color="#374151",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_r, fontsize=8.5)
    ax.set_ylim(len(df_comp) - 0.3, -1.2)
    ax.set_xlim(max(0.55, float(df_comp["val_f1_macro"].min()) - 0.04), 1.045)
    ax.set_xlabel("F1-Macro")
    ax.set_title("Brecha Train vs. Validation (Prevención de Overfitting)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper center", ncol=2, fontsize=8.5, frameon=True)
    ax.grid(axis="x", alpha=0.35)

    fig.suptitle(
        "Comparativa de Fidelidad a DBSCAN y Control de Sobreajuste",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "comparativa_generalizacion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Reporte detallado en test.csv del modelo ganador final
    y_pred_test = model_final.predict(X_test)
    y_prob_test = model_final.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

    prec, rec, f1_c, sup = precision_recall_fscore_support(
        y_test, y_pred_test, labels=[0, 1, 2, 3]
    )
    cls_rows: list[dict[str, Any]] = []
    for c in range(4):
        auc_c = float(roc_auc_score(y_test_bin[:, c], y_prob_test[:, c]))
        cls_rows.append(
            {
                "clase": CLASS_NAMES[c],
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1_score": float(f1_c[c]),
                "soporte_test": int(sup[c]),
                "roc_auc_ovr": auc_c,
            }
        )
    pd.DataFrame(cls_rows).to_csv(table_dir / "metricas_test_por_clase.csv", index=False)

    resumen_test = pd.DataFrame(
        [
            {
                "modelo": f"MLP Ganador 1-SE ({chosen_arm})",
                "n_test": int(len(y_test)),
                "accuracy": float(accuracy_score(y_test, y_pred_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_test)),
                "precision_macro": float(np.mean(prec)),
                "recall_macro": float(np.mean(rec)),
                "f1_macro": float(f1_score(y_test, y_pred_test, average="macro")),
                "f1_weighted": float(f1_score(y_test, y_pred_test, average="weighted")),
                "mcc": float(matthews_corrcoef(y_test, y_pred_test)),
                "roc_auc_ovr_macro": float(roc_auc_score(y_test, y_prob_test, multi_class="ovr", average="macro")),
                "roc_auc_ovr_weighted": float(roc_auc_score(y_test, y_prob_test, multi_class="ovr", average="weighted")),
            }
        ]
    )
    resumen_test.to_csv(table_dir / "metricas_test_resumen.csv", index=False)

    # Matriz de confusión 4x4 (absoluta + normalizada)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    short_cls = ["C0\nPeq-med", "C1\nMed-gde", "C2\nAlargado", "C3\nGigante"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_cls,
        yticklabels=short_cls,
        cbar=False,
        annot_kws={"size": 11, "weight": "bold"},
        ax=axes[0],
    )
    axes[0].set_title("Matriz de Confusión Absoluta (Test, n=1,767)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Predicción MLP")
    axes[0].set_ylabel("Etiqueta DBSCAN")

    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0.0,
        vmax=100.0,
        xticklabels=short_cls,
        yticklabels=short_cls,
        cbar_kws={"label": "% por clase real"},
        annot_kws={"size": 10.5, "weight": "bold"},
        ax=axes[1],
    )
    axes[1].set_title("Matriz de Confusión Normalizada (% Recall)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Predicción MLP")
    axes[1].set_ylabel("Etiqueta DBSCAN")

    fig.suptitle(
        f"Desempeño en Test Independiente — {chosen_arm} (Acc={resumen_test['accuracy'].iloc[0]:.4f}, F1-Macro={resumen_test['f1_macro'].iloc[0]:.4f})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "matriz_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Curvas ROC-AUC One-vs-Rest
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    colors_roc = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea"]
    fpr_dict: dict[int, np.ndarray] = {}
    tpr_dict: dict[int, np.ndarray] = {}
    for c in range(4):
        fpr, tpr, _ = roc_curve(y_test_bin[:, c], y_prob_test[:, c])
        fpr_dict[c] = fpr
        tpr_dict[c] = tpr
        roc_c = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors_roc[c], lw=2.0, label=f"{CLASS_NAMES[c]} (AUC = {roc_c:.4f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob_test.ravel())
    roc_micro = auc(fpr_micro, tpr_micro)
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(4):
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
    mean_tpr /= 4.0
    roc_macro = auc(all_fpr, mean_tpr)

    ax.plot(fpr_micro, tpr_micro, ":", color="#dc2626", lw=2.2, label=f"Micro-promedio (AUC = {roc_micro:.4f})")
    ax.plot(all_fpr, mean_tpr, "--", color="#111827", lw=2.2, label=f"Macro-promedio (AUC = {roc_macro:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC-AUC One-vs-Rest en Conjunto Test (4 Clases)", fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "curvas_roc_ovr.png", dpi=300)
    plt.close(fig)

    return df_comp, model_final


def generate_learning_and_loss_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    top5_payload: dict[str, Any],
    results_df: pd.DataFrame,
    model_1se: Pipeline,
    table_dir: Path,
    fig_dir: Path,
) -> None:
    """Genera Learning Curves y curvas de pérdida/épocas (con y sin early stopping)."""
    one_se_cfg = top5_payload["ganador_1se"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    sizes, tr_scores, va_scores = learning_curve(
        ent.build_mlp_pipeline_from_params(one_se_cfg, random_state=RANDOM_STATE),
        X_train,
        y_train,
        train_sizes=np.linspace(0.10, 1.0, 8),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    lc_df = pd.DataFrame(
        {
            "n_train": sizes.astype(int),
            "train_f1_mean": tr_scores.mean(axis=1),
            "train_f1_std": tr_scores.std(axis=1, ddof=1),
            "cv_f1_mean": va_scores.mean(axis=1),
            "cv_f1_std": va_scores.std(axis=1, ddof=1),
        }
    )
    lc_df.to_csv(table_dir / "learning_curves_mlp.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.fill_between(
        lc_df["n_train"],
        lc_df["train_f1_mean"] - lc_df["train_f1_std"],
        lc_df["train_f1_mean"] + lc_df["train_f1_std"],
        color="#3b82f6",
        alpha=0.15,
    )
    ax.fill_between(
        lc_df["n_train"],
        lc_df["cv_f1_mean"] - lc_df["cv_f1_std"],
        lc_df["cv_f1_mean"] + lc_df["cv_f1_std"],
        color="#059669",
        alpha=0.15,
    )
    ax.plot(lc_df["n_train"], lc_df["train_f1_mean"], "o-", color="#2563eb", lw=2.0, label="Entrenamiento (media ± DE)")
    ax.plot(lc_df["n_train"], lc_df["cv_f1_mean"], "s-", color="#059669", lw=2.0, label="Validación Cruzada 5-fold (media ± DE)")
    ax.set_xlabel("Número de instancias de entrenamiento")
    ax.set_ylabel("F1-Macro")
    ax.set_title(
        f"Curva de Aprendizaje del MLP Ganador 1-SE {_cfg_short_label(one_se_cfg)}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves_mlp.png", dpi=300)
    plt.close(fig)

    # Curvas de pérdida y validación por época (early_stopping=True vs False)
    sgd_rows = results_df[results_df["param_clf__solver"] == "sgd"]
    best_sgd_cfg = ent.row_to_clean_params(sgd_rows.iloc[0])
    sgd_pipe = ent.build_mlp_pipeline_from_params(
        best_sgd_cfg, random_state=RANDOM_STATE, early_stopping=True, max_iter=1000
    ).fit(X_train, y_train)

    traj_df = pd.read_csv(table_dir / "curva_epocas_sin_early_stopping.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    # Panel izquierdo: early_stopping=True (loss_curve_ y validation_scores_)
    ax = axes[0]
    mlp_1se = model_1se.named_steps["clf"]
    mlp_sgd = sgd_pipe.named_steps["clf"]
    l1, = ax.plot(np.arange(1, len(mlp_1se.loss_curve_) + 1), mlp_1se.loss_curve_, color="#2563eb", lw=2.0, label=f"Loss 1-SE ({one_se_cfg['solver'].upper()}, stop ep={mlp_1se.n_iter_})")
    l2, = ax.plot(np.arange(1, len(mlp_sgd.loss_curve_) + 1), mlp_sgd.loss_curve_, color="#f59e0b", lw=1.8, linestyle="--", label=f"Loss Mejor SGD (stop ep={mlp_sgd.n_iter_})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (Cross-Entropy + L2)")
    ax.set_title("Convergencia con Early Stopping (validation_fraction=0.15)", fontsize=11.2, fontweight="bold")
    ax.grid(True, alpha=0.35)

    ax2 = ax.twinx()
    l3, = ax2.plot(np.arange(1, len(mlp_1se.validation_scores_) + 1), mlp_1se.validation_scores_, color="#059669", lw=1.8, alpha=0.85, label="Val Score interno (1-SE)")
    l4, = ax2.plot(np.arange(1, len(mlp_sgd.validation_scores_) + 1), mlp_sgd.validation_scores_, color="#9333ea", lw=1.6, linestyle=":", alpha=0.85, label="Val Score interno (SGD)")
    ax2.set_ylabel("Accuracy en validación interna")
    ax2.set_ylim(max(0.45, min(min(mlp_1se.validation_scores_), min(mlp_sgd.validation_scores_)) - 0.03), 1.008)
    lines = [l1, l2, l3, l4]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=8.2, frameon=True)

    # Panel derecho: trayectoria sin early_stopping hasta 400 épocas
    ax = axes[1]
    ax.plot(traj_df["epoch"], traj_df["train_f1_macro"], color="#2563eb", lw=2.0, label="Train F1-Macro (sin early stopping)")
    ax.plot(traj_df["epoch"], traj_df["val_f1_macro"], color="#059669", lw=2.0, label="Validation F1-Macro (sin early stopping)")
    ax.axvline(mlp_1se.n_iter_, color="#dc2626", linestyle="--", lw=1.6, label=f"Parada temprana elegida (época {mlp_1se.n_iter_})")
    for chk in [50, 100, 200, 400]:
        ax.axvline(chk, color="#9ca3af", linestyle=":", lw=1.0)
    ax.set_xlabel("Época (early_stopping = False)")
    ax.set_ylabel("F1-Macro")
    ax.set_title("Trayectoria Sin Early Stopping: Efecto de #Épocas (50..400)", fontsize=11.2, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(True, alpha=0.35)

    fig.suptitle(
        "Dinámica de Entrenamiento por Época y Control de Sobreajuste",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "curvas_perdida_y_epocas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_ood_noise(
    model_final: Pipeline,
    test_df: pd.DataFrame,
    noise_path: Path,
    table_dir: Path,
    fig_dir: Path,
) -> pd.DataFrame:
    """Evalúa incertidumbre (entropía de Shannon) y clase asignada en el ruido descartado."""
    noise_df = pd.read_csv(noise_path)

    m_n1 = noise_df["origen_ruido"] == "etapa1_global"
    m_bridge = (noise_df["origen_ruido"] == "etapa2_interno") & (
        noise_df["subcategoria_ruido"].str.contains("Puente/frontera", regex=False)
    )
    m_periph = (noise_df["origen_ruido"] == "etapa2_interno") & (~m_bridge)

    groups: list[tuple[str, pd.DataFrame]] = [
        ("Test (En distribución)", test_df),
        ("Ruido Etapa 1: Global 16D", noise_df.loc[m_n1]),
        ("Ruido Etapa 2: Puente/frontera", noise_df.loc[m_bridge]),
        ("Ruido Etapa 2: Periferia/baja dens.", noise_df.loc[m_periph]),
    ]

    rows: list[dict[str, Any]] = []
    ent_samples: list[tuple[str, np.ndarray]] = []
    for gname, gdf in groups:
        Xg = gdf[FEATURE_COLUMNS].to_numpy(dtype=float)
        probs = model_final.predict_proba(Xg)
        preds = model_final.predict(Xg)
        h_norm = -np.sum(probs * np.log(np.clip(probs, 1e-15, 1.0)), axis=1) / np.log(4.0)
        max_p = np.max(probs, axis=1)
        ent_samples.append((gname, h_norm))

        row: dict[str, Any] = {
            "grupo": gname,
            "n": int(len(gdf)),
            "entropia_media": float(np.mean(h_norm)),
            "entropia_mediana": float(np.median(h_norm)),
            "entropia_p90": float(np.quantile(h_norm, 0.90)),
            "prob_max_media": float(np.mean(max_p)),
            "pct_alta_incertidumbre_h_gt_03": float(100.0 * np.mean(h_norm > 0.30)),
        }
        for c in range(4):
            row[f"pct_pred_c{c}"] = float(100.0 * np.mean(preds == c))
        rows.append(row)

    df_ood = pd.DataFrame(rows)
    df_ood.to_csv(table_dir / "evaluacion_ruido_ood.csv", index=False)

    # Gráfica OOD
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5))

    ax = axes[0]
    box_data = [h for _, h in ent_samples]
    box_labels = [
        f"Test\n(En distr.)\n(n={len(ent_samples[0][1])})",
        f"Ruido Etapa 1\n(Global 16D)\n(n={len(ent_samples[1][1])})",
        f"Ruido Etapa 2\n(Puente/front.)\n(n={len(ent_samples[2][1])})",
        f"Ruido Etapa 2\n(Perif./baja dens.)\n(n={len(ent_samples[3][1])})",
    ]
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
    box_colors = ["#10b981", "#dc2626", "#f59e0b", "#64748b"]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    for i, (_, h) in enumerate(ent_samples, start=1):
        ax.scatter(i, np.mean(h), color="#111827", marker="D", s=38, zorder=4, label="Media" if i == 1 else None)
    ax.axhline(0.30, color="#dc2626", linestyle="--", lw=1.2, label="Umbral alta incertidumbre (0.30)")
    ax.set_ylabel("Entropía de Shannon normalizada [0, 1]")
    ax.set_title("Incertidumbre Predictiva en Test vs. Ruido Descartado", fontsize=11.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1]
    x = np.arange(len(df_ood))
    bottom = np.zeros(len(df_ood))
    cls_cols = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea"]
    for c in range(4):
        vals = df_ood[f"pct_pred_c{c}"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.55, color=cls_cols[c], label=CLASS_NAMES[c])
        for i, v in enumerate(vals):
            if v >= 6.0:
                ax.text(x[i], bottom[i] + v / 2.0, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(box_labels, fontsize=8.5)
    ax.set_ylabel("% asignado a cada clase")
    ax.set_ylim(0, 114)
    ax.set_title("Distribución de Clases Asignadas Fuera de Distribución", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper center", fontsize=8, ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle(
        "Evaluación Fuera de Distribución (OOD) sobre Ruido Global y de Frontera",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "entropia_ruido_ood.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df_ood


def _unique_top_models(top5_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Construye lista deduplicada de modelos (1-SE, Top-1 y Top-5)."""
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("Ganador 1-SE", top5_payload["ganador_1se"]),
        ("Top-1 Absoluto", top5_payload["top1_absoluto"]),
    ] + [(f"Top-{cfg['rank_grid']}", cfg) for cfg in top5_payload["top5"]]

    seen: set[tuple[Any, ...]] = set()
    unique_list: list[tuple[str, dict[str, Any]]] = []
    for label, cfg in candidates:
        sig = _cfg_signature(cfg)
        if sig not in seen:
            seen.add(sig)
            unique_list.append((f"{label}: {_cfg_short_label(cfg)}", cfg))
    return unique_list


def _summarize_multiseed_raw(
    df_raw: pd.DataFrame, model_order: list[str], n_seeds: int
) -> pd.DataFrame:
    """Resume variación multi-semilla con IC 95 % t-Student acotado a [0, 1]."""
    t_crit = float(stats.t.ppf(0.975, df=n_seeds - 1))
    summary_rows: list[dict[str, Any]] = []
    for name in model_order:
        sub = df_raw[df_raw["modelo"] == name]
        f1s = sub["val_f1_macro"].to_numpy(dtype=float)
        accs = sub["val_accuracy"].to_numpy(dtype=float)
        mu_f1 = float(np.mean(f1s))
        sd_f1 = float(np.std(f1s, ddof=1))
        se_f1 = sd_f1 / np.sqrt(n_seeds)
        summary_rows.append(
            {
                "modelo": name,
                "n_semillas": n_seeds,
                "f1_macro_media": mu_f1,
                "f1_macro_std": sd_f1,
                "f1_macro_var": float(np.var(f1s, ddof=1)),
                "f1_macro_ic95_inf": float(max(0.0, mu_f1 - t_crit * se_f1)),
                "f1_macro_ic95_sup": float(min(1.0, mu_f1 + t_crit * se_f1)),
                "f1_macro_min": float(np.min(f1s)),
                "f1_macro_max": float(np.max(f1s)),
                "accuracy_media": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs, ddof=1)),
            }
        )
    return pd.DataFrame(summary_rows)


def _summarize_rskf_raw(
    df_raw: pd.DataFrame, model_order: list[str], n_splits: int = 5, n_runs: int = 50
) -> pd.DataFrame:
    """Resume RSKF con IC 95 % corregido por Nadeau-Bengio y acotado a [0, 1]."""
    t_crit = float(stats.t.ppf(0.975, df=n_runs - 1))
    corr_factor_ci = (1.0 / n_runs) + (1.0 / (n_splits - 1))
    sum_rows: list[dict[str, Any]] = []
    for name in model_order:
        sub = df_raw[df_raw["modelo"] == name].sort_values("run_idx")
        f1s = sub["f1_macro"].to_numpy(dtype=float)
        accs = sub["accuracy"].to_numpy(dtype=float)
        mu_f1 = float(np.mean(f1s))
        sd_f1 = float(np.std(f1s, ddof=1))
        se_f1 = sd_f1 * np.sqrt(corr_factor_ci)
        sum_rows.append(
            {
                "modelo": name,
                "n_corridas": n_runs,
                "f1_macro_media": mu_f1,
                "f1_macro_std": sd_f1,
                "f1_macro_var": float(np.var(f1s, ddof=1)),
                "f1_macro_se_nb": float(se_f1),
                "f1_macro_ic95_inf": float(max(0.0, mu_f1 - t_crit * se_f1)),
                "f1_macro_ic95_sup": float(min(1.0, mu_f1 + t_crit * se_f1)),
                "accuracy_media": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs, ddof=1)),
            }
        )
    return pd.DataFrame(sum_rows)


def run_multiseed_variation(
    model_specs: list[tuple[str, dict[str, Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    table_dir: Path,
    fig_dir: Path,
    n_seeds: int = 20,
) -> pd.DataFrame:
    """Evalúa variación por inicialización de pesos con 20 semillas en validation."""
    def _eval_one(name: str, cfg: dict[str, Any], seed: int) -> tuple[str, int, float, float]:
        pipe = ent.build_mlp_pipeline_from_params(cfg, random_state=seed).fit(X_train, y_train)
        pred = pipe.predict(X_val)
        return (
            name,
            seed,
            float(f1_score(y_val, pred, average="macro")),
            float(accuracy_score(y_val, pred)),
        )

    jobs = [(name, cfg, seed) for name, cfg in model_specs for seed in range(n_seeds)]
    raw_res = Parallel(n_jobs=-1)(
        delayed(_eval_one)(name, cfg, seed) for name, cfg, seed in jobs
    )
    df_raw = pd.DataFrame(raw_res, columns=["modelo", "seed", "val_f1_macro", "val_accuracy"])
    df_raw.to_csv(table_dir / "variacion_multisementes_raw.csv", index=False)

    df_sum = _summarize_multiseed_raw(
        df_raw, [name for name, _ in model_specs], n_seeds=n_seeds
    )
    df_sum.to_csv(table_dir / "variacion_multisementes.csv", index=False)

    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    order = [name for name, _ in model_specs]
    short_labels = []
    for n in order:
        role, desc = n.split(": ")
        parts = desc.split(") ")
        arch_part = parts[0] + ")"
        rest_part = parts[1] if len(parts) > 1 else ""
        short_labels.append(f"{role}\n{arch_part}\n{rest_part}")
    sns.boxplot(
        data=df_raw,
        x="modelo",
        y="val_f1_macro",
        order=order,
        color="#bfdbfe",
        width=0.50,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_raw,
        x="modelo",
        y="val_f1_macro",
        order=order,
        color="#1d4ed8",
        alpha=0.70,
        size=4.5,
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short_labels, fontsize=8.2)
    ax.set_ylabel("F1-Macro en Validation")
    ax.set_title(
        f"Variación Estadística por Inicialización de Pesos ({n_seeds} semillas: 0..{n_seeds-1})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "variacion_multisementes.png", dpi=300)
    plt.close(fig)
    return df_sum


def run_rskf_and_nadeau_bengio(
    model_specs: list[tuple[str, dict[str, Any]]],
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    table_dir: Path,
    fig_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta RepeatedStratifiedKFold(5x10) y prueba t corregida de Nadeau-Bengio."""
    n_splits, n_repeats = 5, 10
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    splits = list(rskf.split(X_tv, y_tv))
    n_runs = len(splits)  # 50

    all_models: list[tuple[str, str, dict[str, Any] | None]] = [
        (name, "mlp", cfg) for name, cfg in model_specs
    ] + [
        ("Base: Regresión Logística", "logreg", None),
        ("Base: Centroide Más Cercano", "nc", None),
    ]

    def _run_one(model_name: str, m_type: str, cfg: dict[str, Any] | None, run_idx: int) -> tuple[str, int, float, float]:
        tr_idx, va_idx = splits[run_idx]
        if m_type == "mlp" and cfg is not None:
            pipe = ent.build_mlp_pipeline_from_params(cfg, random_state=100 + run_idx)
        elif m_type == "logreg":
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=100 + run_idx)),
                ]
            )
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", NearestCentroid())])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X_tv[tr_idx], y_tv[tr_idx])
        pred = pipe.predict(X_tv[va_idx])
        return (
            model_name,
            run_idx,
            float(f1_score(y_tv[va_idx], pred, average="macro")),
            float(accuracy_score(y_tv[va_idx], pred)),
        )

    jobs = [
        (name, m_type, cfg, r_i)
        for name, m_type, cfg in all_models
        for r_i in range(n_runs)
    ]
    res = Parallel(n_jobs=-1)(
        delayed(_run_one)(name, m_type, cfg, r_i) for name, m_type, cfg, r_i in jobs
    )
    df_raw = pd.DataFrame(res, columns=["modelo", "run_idx", "f1_macro", "accuracy"])
    df_raw.to_csv(table_dir / "variacion_rskf_raw.csv", index=False)

    df_sum = _summarize_rskf_raw(
        df_raw, [name for name, _, _ in all_models], n_splits=n_splits, n_runs=n_runs
    )
    df_sum.to_csv(table_dir / "variacion_rskf_resumen.csv", index=False)

    # Prueba t corregida de Nadeau-Bengio comparando Ganador 1-SE contra los demás
    ref_name = all_models[0][0]
    ref_scores = df_raw[df_raw["modelo"] == ref_name].sort_values("run_idx")["f1_macro"].to_numpy(dtype=float)
    # n_test / n_train = 1 / 4 en 5-fold CV
    corr_factor = (1.0 / n_runs) + (1.0 / 4.0)
    nb_rows: list[dict[str, Any]] = []
    for name, _, _ in all_models[1:]:
        other_scores = df_raw[df_raw["modelo"] == name].sort_values("run_idx")["f1_macro"].to_numpy(dtype=float)
        diff = ref_scores - other_scores
        mean_d = float(np.mean(diff))
        var_d = float(np.var(diff, ddof=1))
        se_nb = float(np.sqrt(corr_factor * var_d))
        if se_nb > 1e-12:
            t_nb = mean_d / se_nb
            p_nb = float(2.0 * stats.t.sf(abs(t_nb), df=n_runs - 1))
        else:
            t_nb = 0.0
            p_nb = 1.0
        nb_rows.append(
            {
                "modelo_referencia": ref_name,
                "modelo_comparado": name,
                "diff_media_f1_macro": mean_d,
                "std_diff": float(np.std(diff, ddof=1)),
                "se_nadeau_bengio": se_nb,
                "t_nadeau_bengio": t_nb,
                "p_valor_bilateral": p_nb,
                "significativo_005": bool(p_nb < 0.05),
            }
        )
    df_nb = pd.DataFrame(nb_rows)
    df_nb.to_csv(table_dir / "comparacion_nadeau_bengio.csv", index=False)

    # Gráfica RSKF
    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    order = [m[0] for m in all_models]
    short_lbls = []
    for m in order:
        if ": " in m:
            role, desc = m.split(": ")
            if ") " in desc:
                arch_p, rest_p = desc.split(") ", 1)
                short_lbls.append(f"{role}\n{arch_p})\n{rest_p}")
            else:
                short_lbls.append(f"{role}\n{desc.replace(' ', chr(10), 1)}")
        else:
            short_lbls.append(m)
    sns.boxplot(
        data=df_raw,
        x="modelo",
        y="f1_macro",
        order=order,
        color="#a7f3d0",
        width=0.52,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_raw,
        x="modelo",
        y="f1_macro",
        order=order,
        color="#047857",
        alpha=0.42,
        size=3.2,
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short_lbls, fontsize=8.0)
    ax.set_ylabel("F1-Macro (50 corridas: datos + pesos)")
    ax.set_title(
        "RepeatedStratifiedKFold (5x10 = 50 corridas sobre Train+Val: Estimación de Varianza)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "variacion_estadistica_rskf.png", dpi=300)
    plt.close(fig)

    return df_sum, df_nb


def run_bootstrap_test(
    base_dir: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chosen_arm: str,
    table_dir: Path,
    fig_dir: Path,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Bootstrap estratificado (B=1,000) sobre test.csv para estimar IC 95 %."""
    models_dir = base_dir / "models"
    models: list[tuple[str, Pipeline]] = [
        (f"MLP Ganador 1-SE ({chosen_arm})", joblib.load(models_dir / "mejor_modelo_mlp_final.joblib")),
        ("Base: Regresión Logística", joblib.load(models_dir / "baseline_logreg.joblib")),
        ("Base: Centroide Más Cercano", joblib.load(models_dir / "baseline_nearest_centroid.joblib")),
    ]

    rng = np.random.default_rng(RANDOM_STATE)
    cls_indices = [np.flatnonzero(y_test == c) for c in range(4)]
    boot_indices = [
        np.concatenate([rng.choice(idx_c, size=len(idx_c), replace=True) for idx_c in cls_indices])
        for _ in range(n_boot)
    ]

    rows: list[dict[str, Any]] = []
    boot_f1_dist: dict[str, np.ndarray] = {}

    for m_name, pipe in models:
        y_pred = pipe.predict(X_test)
        has_prob = hasattr(pipe.named_steps["clf"], "predict_proba")
        y_prob = pipe.predict_proba(X_test) if has_prob else None

        accs = np.empty(n_boot)
        baccs = np.empty(n_boot)
        f1ms = np.empty(n_boot)
        f1ws = np.empty(n_boot)
        rocs = np.full(n_boot, np.nan)

        for b, b_idx in enumerate(boot_indices):
            yb = y_test[b_idx]
            pb = y_pred[b_idx]
            accs[b] = accuracy_score(yb, pb)
            baccs[b] = balanced_accuracy_score(yb, pb)
            f1ms[b] = f1_score(yb, pb, average="macro")
            f1ws[b] = f1_score(yb, pb, average="weighted")
            if y_prob is not None:
                rocs[b] = roc_auc_score(yb, y_prob[b_idx], multi_class="ovr")

        boot_f1_dist[m_name] = f1ms
        for metric_name, vals in [
            ("accuracy", accs),
            ("balanced_accuracy", baccs),
            ("f1_macro", f1ms),
            ("f1_weighted", f1ws),
            ("roc_auc_ovr", rocs),
        ]:
            if np.isnan(vals).all():
                continue
            rows.append(
                {
                    "modelo": m_name,
                    "metrica": metric_name,
                    "n_bootstrap": n_boot,
                    "media": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "varianza": float(np.var(vals, ddof=1)),
                    "ic95_p25": float(np.quantile(vals, 0.025)),
                    "ic95_p975": float(np.quantile(vals, 0.975)),
                }
            )

    df_boot = pd.DataFrame(rows)
    df_boot.to_csv(table_dir / "bootstrap_test_ci.csv", index=False)

    # Gráfica Bootstrap F1-Macro e IC 95% en Test (manejando varianza cero limpiamente)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))

    ax = axes[0]
    colors = ["#2563eb", "#059669", "#f59e0b"]
    styles = ["-", "--", "-."]
    for (m_name, vals), col, ls in zip(boot_f1_dist.items(), colors, styles):
        ci_l, ci_h = np.quantile(vals, [0.025, 0.975])
        mu_v = float(np.mean(vals))
        sd_v = float(np.std(vals, ddof=1))
        lbl = f"{m_name}: μ={mu_v:.4f} [IC95%: {ci_l:.4f}, {ci_h:.4f}]"
        if sd_v > 1e-8:
            sns.kdeplot(vals, fill=True, alpha=0.25, lw=2.0, color=col, ax=ax, label=lbl, warn_singular=False)
            ax.axvline(mu_v, color=col, linestyle="--", lw=1.3)
        else:
            ax.axvline(mu_v, color=col, linestyle=ls, lw=2.6, label=lbl + " (σ=0)")
            ax.scatter([mu_v], [5.0 if col == "#2563eb" else 12.0], color=col, s=55, zorder=5)
    ax.set_xlim(0.945, 1.005)
    ax.set_xlabel("F1-Macro en Bootstrap Estratificado (Test, B=1,000)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución Bootstrap de F1-Macro en Test", fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.3, frameon=True)
    ax.grid(True, alpha=0.35)

    ax = axes[1]
    sub_b = df_boot[df_boot["metrica"].isin(["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"])].copy()
    m_order = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"]
    m_labels = ["Accuracy", "Balanced Acc.", "F1-Macro", "ROC-AUC OvR"]
    y_base = np.arange(len(m_order))
    offsets = [-0.20, 0.0, 0.20]
    for (m_name, _), col, off in zip(models, colors, offsets):
        sub_m = sub_b[sub_b["modelo"] == m_name].set_index("metrica").reindex(m_order)
        means = sub_m["media"].to_numpy(dtype=float)
        err_low = means - sub_m["ic95_p25"].to_numpy(dtype=float)
        err_high = sub_m["ic95_p975"].to_numpy(dtype=float) - means
        ax.errorbar(
            means,
            y_base + off,
            xerr=[err_low, err_high],
            fmt="o",
            color=col,
            capsize=4,
            markersize=6,
            lw=1.8,
            label=m_name,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(m_labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0.945, 1.005)
    ax.set_xlabel("Estimación media e IC 95% percentil")
    ax.set_title("Intervalos de Confianza 95% Bootstrap (B=1,000)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.3, frameon=True)
    ax.grid(True, alpha=0.35)

    fig.suptitle(
        "Estabilidad Estadística en Test Independiente (Bootstrap Estratificado B = 1,000)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_test_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return df_boot


PRIMARY_COLUMNS: list[str] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "Extent",
]


def _recompute_derived_features(df_prim: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye las 16 variables preservando identidades geométricas exactas."""
    A = np.maximum(df_prim["Area"].to_numpy(dtype=float), 1000.0)
    L = np.maximum(df_prim["MajorAxisLength"].to_numpy(dtype=float), 20.0)
    l = np.clip(df_prim["MinorAxisLength"].to_numpy(dtype=float), 10.0, L * 0.9999)
    CA = np.maximum(df_prim["ConvexArea"].to_numpy(dtype=float), A * 1.0001)
    # Desigualdad isoperimétrica (4*pi*A / P^2 < 1) para garantizar roundness física válida
    P = np.maximum(df_prim["Perimeter"].to_numpy(dtype=float), np.sqrt(4.0 * np.pi * A) * 1.0005)
    Ex = np.clip(df_prim["Extent"].to_numpy(dtype=float), 0.45, 0.92)

    ar = L / l
    ecc = np.sqrt(np.maximum(0.0, 1.0 - (l / L) ** 2))
    ed = np.sqrt(4.0 * A / np.pi)
    sol = A / CA
    rnd = (4.0 * np.pi * A) / (P ** 2)
    cmp_v = ed / L
    sf1 = L / A
    sf2 = A / (L ** 3)
    sf3 = cmp_v ** 2
    sf4 = (4.0 * A) / (np.pi * L * l)

    out = pd.DataFrame(
        {
            "Area": A,
            "Perimeter": P,
            "MajorAxisLength": L,
            "MinorAxisLength": l,
            "AspectRation": ar,
            "Eccentricity": ecc,
            "ConvexArea": CA,
            "EquivDiameter": ed,
            "Extent": Ex,
            "Solidity": sol,
            "roundness": rnd,
            "Compactness": cmp_v,
            "ShapeFactor1": sf1,
            "ShapeFactor2": sf2,
            "ShapeFactor3": sf3,
            "ShapeFactor4": sf4,
        }
    )
    return out[FEATURE_COLUMNS]


def _predict_with_entropy(
    pipe: Pipeline, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Ejecuta inferencia sin ajuste y calcula entropía normalizada si hay probabilidades."""
    preds = pipe.predict(X)
    if not hasattr(pipe.named_steps["clf"], "predict_proba"):
        return preds, None, None, None
    probs = pipe.predict_proba(X)
    h_norm = -np.sum(probs * np.log(np.clip(probs, 1e-15, 1.0)), axis=1) / np.log(4.0)
    max_p = np.max(probs, axis=1)
    return preds, probs, h_norm, max_p


def _compute_dbscan_manifold_coverage(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    nn_c3: NearestNeighbors,
    nn_main: NearestNeighbors,
    scaler1: Any,
    scaler2: Any,
    pca2: Any,
) -> float:
    """Porcentaje de muestras sintéticas dentro del radio epsilon de su estrato DBSCAN."""
    in_manifold = np.zeros(len(X_syn), dtype=bool)
    m3 = y_syn == 3
    if np.any(m3):
        d3, _ = nn_c3.kneighbors(scaler1.transform(X_syn[m3]), n_neighbors=1)
        in_manifold[m3] = d3[:, 0] <= 1.50
    m012 = ~m3
    if np.any(m012):
        xp2 = pca2.transform(scaler2.transform(X_syn[m012]))
        d2, _ = nn_main.kneighbors(xp2, n_neighbors=1)
        in_manifold[m012] = d2[:, 0] <= 0.38
    return float(100.0 * np.mean(in_manifold))


def _row_keys_array(df_feats: pd.DataFrame) -> list[tuple[float, ...]]:
    """Llave numérica canónica por fila (9 decimales) para cruce exacto con el crudo."""
    arr = np.round(df_feats[FEATURE_COLUMNS].to_numpy(dtype=float), 9)
    return [tuple(row) for row in arr.tolist()]


def _generate_synthetic_cohorts(
    test_df: pd.DataFrame, raw_key_set: set[tuple[float, ...]]
) -> list[tuple[str, float | None, np.ndarray, np.ndarray]]:
    """Genera cohorte interpolada intra-clase (n=2,000) y perturbaciones gaussianas."""
    rng_interp = np.random.default_rng(RANDOM_STATE)
    n_per_class = 500
    interp_frames: list[pd.DataFrame] = []
    interp_labels: list[np.ndarray] = []

    for c in range(4):
        sub_c = test_df[test_df["cluster"] == c].reset_index(drop=True)
        X_c_scaled = StandardScaler().fit_transform(sub_c[PRIMARY_COLUMNS].to_numpy(dtype=float))
        k_nn = min(6, len(sub_c))
        _, nn_idx = NearestNeighbors(n_neighbors=k_nn).fit(X_c_scaled).kneighbors(X_c_scaled)
        neighbor_pool = nn_idx[:, 1:]

        base_i = rng_interp.integers(0, len(sub_c), size=n_per_class)
        nbr_j = rng_interp.integers(0, neighbor_pool.shape[1], size=n_per_class)
        lam = rng_interp.uniform(0.15, 0.85, size=(n_per_class, 1))

        prim_arr = sub_c[PRIMARY_COLUMNS].to_numpy(dtype=float)
        syn_prim = (1.0 - lam) * prim_arr[base_i] + lam * prim_arr[neighbor_pool[base_i, nbr_j]]
        interp_frames.append(pd.DataFrame(syn_prim, columns=PRIMARY_COLUMNS))
        interp_labels.append(np.full(n_per_class, c, dtype=np.int64))

    df_interp = _recompute_derived_features(pd.concat(interp_frames, ignore_index=True))
    y_interp = np.concatenate(interp_labels)

    # Garantía de cero colisiones con datos reales e invariantes físicos estrictos
    assert (df_interp["Area"] <= df_interp["ConvexArea"]).all()
    assert (df_interp["Solidity"] <= 1.0).all()
    assert (df_interp["roundness"] <= 1.0).all()
    interp_keys = set(_row_keys_array(df_interp))
    assert len(interp_keys & raw_key_set) == 0, "Colisión inesperada entre sintéticos y reales."

    cohorts: list[tuple[str, float | None, np.ndarray, np.ndarray]] = [
        (
            "Interpolación geométrica intra-clase (n=2,000)",
            np.nan,
            df_interp[FEATURE_COLUMNS].to_numpy(dtype=float),
            y_interp,
        )
    ]

    # Perturbación gaussiana controlada por desviación estándar intra-clase de test
    X_test_orig = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["cluster"].to_numpy(dtype=np.int64)
    prim_test = test_df[PRIMARY_COLUMNS].to_numpy(dtype=float)
    std_by_class = {
        c: np.std(prim_test[y_test == c], axis=0, ddof=1) for c in range(4)
    }

    for sigma in [0.00, 0.01, 0.02, 0.05, 0.10, 0.15]:
        if sigma == 0.0:
            X_noisy = X_test_orig
        else:
            rng_g = np.random.default_rng(RANDOM_STATE + int(round(sigma * 1000)))
            noisy_prim = prim_test.copy()
            for c in range(4):
                mc = y_test == c
                noise = rng_g.normal(0.0, sigma * std_by_class[c], size=(int(mc.sum()), len(PRIMARY_COLUMNS)))
                noisy_prim[mc] = prim_test[mc] + noise
            df_noisy = _recompute_derived_features(pd.DataFrame(noisy_prim, columns=PRIMARY_COLUMNS))
            X_noisy = df_noisy[FEATURE_COLUMNS].to_numpy(dtype=float)
        cohorts.append((f"Perturbación gaussiana (sigma={sigma:.2f})", float(sigma), X_noisy, y_test))

    return cohorts


def _plot_synthetic_and_raw_figure(
    df_syn: pd.DataFrame, df_raw_eval: pd.DataFrame, out_path: Path
) -> None:
    """Genera figura doble: generalización sintética y evaluación sobre el dataset crudo."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    # Panel izquierdo: F1-Macro en interpolación + perturbación gaussiana sigma 0..15%
    ax = axes[0]
    regimes = [
        "Interpolación geométrica intra-clase (n=2,000)",
        "Perturbación gaussiana (sigma=0.00)",
        "Perturbación gaussiana (sigma=0.01)",
        "Perturbación gaussiana (sigma=0.02)",
        "Perturbación gaussiana (sigma=0.05)",
        "Perturbación gaussiana (sigma=0.10)",
        "Perturbación gaussiana (sigma=0.15)",
    ]
    x_labels = ["Interp.\n(n=2,000)", "σ=0%", "σ=1%", "σ=2%", "σ=5%", "σ=10%", "σ=15%"]
    x = np.arange(len(regimes))

    model_styles = [
        ("MLP Ganador 1-SE", "#2563eb", "o-", 2.2),
        ("MLP Top-1 Absoluto", "#9333ea", "^--", 1.8),
        ("Base: Regresión Logística", "#059669", "s-.", 1.9),
        ("Base: Centroide Más Cercano", "#f59e0b", "D:", 1.8),
    ]
    for m_name, col, fmt, lw in model_styles:
        sub = df_syn[df_syn["modelo"] == m_name].set_index("regimen").reindex(regimes)
        ax.plot(x, sub["f1_macro"].to_numpy(dtype=float), fmt, color=col, lw=lw, label=m_name)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8.8)
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("F1-Macro (fidelidad sintética)")
    ax.set_title("Generalización con Datos Sintéticos Coherentes y Estrés Gaussiano", fontsize=11.0, fontweight="bold")
    ax.grid(True, alpha=0.35)

    ax2 = ax.twinx()
    sub_1se = df_syn[df_syn["modelo"] == "MLP Ganador 1-SE"].set_index("regimen").reindex(regimes)
    l_ent, = ax2.plot(
        x,
        sub_1se["entropia_media"].to_numpy(dtype=float),
        "X--",
        color="#dc2626",
        lw=1.7,
        alpha=0.85,
        label="Entropía media H_norm (MLP 1-SE)",
    )
    ax2.set_ylabel("Entropía normalizada media (MLP 1-SE)", color="#dc2626")
    ax2.tick_params(axis="y", colors="#dc2626")
    ax2.set_ylim(0.0, 0.35)

    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(
        h1 + [l_ent],
        l1 + [l_ent.get_label()],
        loc="center left",
        bbox_to_anchor=(0.02, 0.45),
        fontsize=7.8,
        frameon=True,
    )

    # Panel derecho: Inferencia del MLP Ganador 1-SE sobre Dataset.csv crudo (13,611 filas)
    ax = axes[1]
    sub_r = df_raw_eval[df_raw_eval["modelo"] == "MLP Ganador 1-SE"].copy()
    sub_order = [
        ("Crudo con Estrato Válido (únicos + dup.)", "Crudo\nVálido"),
        ("Crudo Válido No en Train (Val+Test+dup.)", "Crudo\nNo Train"),
        ("Crudo: Duplicados Válidos (n exactos)", "Duplicados\nVálidos"),
        ("Crudo Total (Dataset.csv 13,611)", "Crudo\nTotal"),
        ("Crudo Ruido E1: Global 16D", "Ruido E1\nGlobal"),
        ("Crudo Ruido E2: Puente/Frontera", "Ruido E2\nPuente"),
        ("Crudo Ruido E2: Periferia/Baja Dens.", "Ruido E2\nPerif."),
    ]
    sub_r = sub_r.set_index("subconjunto_crudo").reindex([k for k, _ in sub_order])
    xr = np.arange(len(sub_order))
    w = 0.36

    acc_pct = sub_r["accuracy"].fillna(0.0).to_numpy(dtype=float) * 100.0
    unc_pct = sub_r["pct_alta_incertidumbre_h_gt_03"].to_numpy(dtype=float)
    n_vals = sub_r["n_registros"].to_numpy(dtype=int)

    b1 = ax.bar(xr[:3] - w / 2, acc_pct[:3], width=w, color="#2563eb", label="Concordancia c/ DBSCAN (%)")
    b2 = ax.bar(xr + w / 2, unc_pct, width=w, color="#f59e0b", label="% Alta incertidumbre (H > 0.30)")
    for bar, v in zip(b1, acc_pct[:3]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    for bar, v in zip(b2, unc_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.6)

    ax.set_xticks(xr)
    ax.set_xticklabels([f"{lbl}\n(n={n:,})" for (_, lbl), n in zip(sub_order, n_vals)], fontsize=8.0)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Inferencia Directa sobre Dataset Crudo (Dataset.csv, 13,611 filas)", fontsize=11.0, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.2, frameon=True)
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle(
        "Validación Externa sin Reentrenamiento: Datos Sintéticos Geométricos y Dataset Crudo",
        fontsize=13.2,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_synthetic_and_raw_dataset(
    base_dir: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa modelos congelados sobre datos sintéticos y sobre Dataset.csv crudo."""
    models_dir = base_dir / "models"
    data_dir = base_dir / "data" / "processed"
    raw_csv = base_dir / "data" / "raw" / "Dataset.csv"
    table_dir = base_dir / "results" / "tablas"
    fig_dir = base_dir / "results" / "figuras"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Carga exclusivamente en modo lectura de los modelos ya entrenados y congelados
    frozen_models: list[tuple[str, Pipeline]] = [
        ("MLP Ganador 1-SE", joblib.load(models_dir / "mejor_modelo_mlp_final.joblib")),
        ("MLP Top-1 Absoluto", joblib.load(models_dir / "mejor_modelo_mlp_top1.joblib")),
        ("Base: Regresión Logística", joblib.load(models_dir / "baseline_logreg.joblib")),
        ("Base: Centroide Más Cercano", joblib.load(models_dir / "baseline_nearest_centroid.joblib")),
    ]
    scaler1 = joblib.load(models_dir / "scaler_etapa1_dbscan.joblib")
    scaler2 = joblib.load(models_dir / "scaler_etapa2_dbscan.joblib")
    pca2 = joblib.load(models_dir / "pca_etapa2_dbscan.joblib")

    clean_df = pd.read_csv(data_dir / "dataset_limpio_clusters.csv")
    noise_df = pd.read_csv(data_dir / "ruido_descartado.csv")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Referencia espacial DBSCAN para medir permanencia dentro del manifold de densidad
    X_c3_clean = clean_df.loc[clean_df["cluster"] == 3, FEATURE_COLUMNS].to_numpy(dtype=float)
    X_m_clean = clean_df.loc[clean_df["cluster"] != 3, FEATURE_COLUMNS].to_numpy(dtype=float)
    nn_c3 = NearestNeighbors(n_neighbors=1).fit(scaler1.transform(X_c3_clean))
    nn_main = NearestNeighbors(n_neighbors=1).fit(pca2.transform(scaler2.transform(X_m_clean)))

    # Dataset crudo completo (13,611 x 17 -> elimina solo columna vacía final)
    df_raw_full = pd.read_csv(raw_csv, encoding="utf-8-sig")
    all_nan_cols = [c for c in df_raw_full.columns if df_raw_full[c].isna().all()]
    df_raw = df_raw_full.drop(columns=all_nan_cols).copy()
    raw_keys = _row_keys_array(df_raw)
    raw_key_set = set(raw_keys)

    # Experimento A: Cohortes sintéticas geométricas y perturbaciones gaussianas
    cohorts = _generate_synthetic_cohorts(test_df, raw_key_set)
    syn_rows: list[dict[str, Any]] = []
    for regimen, sigma_val, X_syn, y_syn in cohorts:
        manifold_pct = _compute_dbscan_manifold_coverage(
            X_syn, y_syn, nn_c3, nn_main, scaler1, scaler2, pca2
        )
        for m_name, pipe in frozen_models:
            preds, probs, h_norm, max_p = _predict_with_entropy(pipe, X_syn)
            roc = (
                float(roc_auc_score(y_syn, probs, multi_class="ovr"))
                if probs is not None
                else np.nan
            )
            syn_rows.append(
                {
                    "regimen": regimen,
                    "nivel_ruido_sigma": sigma_val,
                    "modelo": m_name,
                    "n_muestras": int(len(y_syn)),
                    "pct_dentro_manifold_dbscan": manifold_pct,
                    "accuracy": float(accuracy_score(y_syn, preds)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_syn, preds)),
                    "f1_macro": float(f1_score(y_syn, preds, average="macro")),
                    "roc_auc_ovr": roc,
                    "entropia_media": float(np.mean(h_norm)) if h_norm is not None else np.nan,
                    "prob_max_media": float(np.mean(max_p)) if max_p is not None else np.nan,
                    "pct_alta_incertidumbre_h_gt_03": (
                        float(100.0 * np.mean(h_norm > 0.30)) if h_norm is not None else np.nan
                    ),
                }
            )
    df_syn = pd.DataFrame(syn_rows)
    df_syn.to_csv(table_dir / "evaluacion_datos_sinteticos.csv", index=False)

    # Experimento B: Inferencia directa sobre Dataset.csv crudo completo (13,611 filas)
    clean_map = dict(zip(_row_keys_array(clean_df), clean_df["cluster"].to_numpy(dtype=np.int64)))
    noise_origin_map: dict[tuple[float, ...], str] = {}
    for k, orig, subc in zip(
        _row_keys_array(noise_df),
        noise_df["origen_ruido"],
        noise_df["subcategoria_ruido"],
    ):
        if orig == "etapa1_global":
            noise_origin_map[k] = "ruido_e1"
        elif "Puente/frontera" in str(subc):
            noise_origin_map[k] = "ruido_e2_puente"
        else:
            noise_origin_map[k] = "ruido_e2_periferia"
    train_key_set = set(_row_keys_array(train_df))

    dest_cluster = np.array([clean_map.get(k, -1) for k in raw_keys], dtype=np.int64)
    dest_kind = np.array(
        [
            "estrato_valido" if clean_map.get(k, -1) != -1 else noise_origin_map[k]
            for k in raw_keys
        ],
        dtype=object,
    )
    in_train = np.array([k in train_key_set for k in raw_keys], dtype=bool)
    is_dup = df_raw[FEATURE_COLUMNS].duplicated(keep="first").to_numpy(dtype=bool)

    m_valid = dest_cluster != -1
    raw_subsets: list[tuple[str, np.ndarray]] = [
        ("Crudo Total (Dataset.csv 13,611)", np.ones(len(df_raw), dtype=bool)),
        ("Crudo con Estrato Válido (únicos + dup.)", m_valid),
        ("Crudo Válido No en Train (Val+Test+dup.)", m_valid & (~in_train)),
        ("Crudo: Duplicados Válidos (n exactos)", m_valid & is_dup),
        ("Crudo Ruido E1: Global 16D", dest_kind == "ruido_e1"),
        ("Crudo Ruido E2: Puente/Frontera", dest_kind == "ruido_e2_puente"),
        ("Crudo Ruido E2: Periferia/Baja Dens.", dest_kind == "ruido_e2_periferia"),
    ]

    X_raw_all = df_raw[FEATURE_COLUMNS].to_numpy(dtype=float)
    raw_eval_models = [frozen_models[0], frozen_models[2]]  # MLP 1-SE + Regresión Logística
    raw_rows: list[dict[str, Any]] = []
    for m_name, pipe in raw_eval_models:
        preds_all, probs_all, h_all, maxp_all = _predict_with_entropy(pipe, X_raw_all)
        assert probs_all is not None and h_all is not None and maxp_all is not None
        for s_name, mask in raw_subsets:
            n_sub = int(mask.sum())
            sub_valid = mask & m_valid
            n_val_sub = int(sub_valid.sum())
            p_sub = preds_all[mask]
            h_sub = h_all[mask]
            mp_sub = maxp_all[mask]

            if n_val_sub > 0 and s_name != "Crudo Total (Dataset.csv 13,611)":
                y_sv = dest_cluster[sub_valid]
                p_sv = preds_all[sub_valid]
                pr_sv = probs_all[sub_valid]
                n_hits = int(np.sum(y_sv == p_sv))
                n_err = int(n_val_sub - n_hits)
                acc_v = float(accuracy_score(y_sv, p_sv))
                f1_v = float(f1_score(y_sv, p_sv, average="macro"))
                roc_v = (
                    float(roc_auc_score(y_sv, pr_sv, multi_class="ovr"))
                    if len(np.unique(y_sv)) == 4
                    else np.nan
                )
            else:
                n_hits = int(np.sum(dest_cluster[sub_valid] == preds_all[sub_valid])) if n_val_sub > 0 else 0
                n_err = int(n_val_sub - n_hits) if n_val_sub > 0 else 0
                acc_v = np.nan
                f1_v = np.nan
                roc_v = np.nan

            raw_rows.append(
                {
                    "modelo": m_name,
                    "subconjunto_crudo": s_name,
                    "n_registros": n_sub,
                    "n_con_estrato_valido": n_val_sub,
                    "n_aciertos_estrato": n_hits if n_val_sub > 0 else np.nan,
                    "n_errores_estrato": n_err if n_val_sub > 0 else np.nan,
                    "accuracy": acc_v,
                    "f1_macro": f1_v,
                    "roc_auc_ovr": roc_v,
                    "entropia_media": float(np.mean(h_sub)),
                    "prob_max_media": float(np.mean(mp_sub)),
                    "pct_alta_incertidumbre_h_gt_03": float(100.0 * np.mean(h_sub > 0.30)),
                    "pct_pred_c0": float(100.0 * np.mean(p_sub == 0)),
                    "pct_pred_c1": float(100.0 * np.mean(p_sub == 1)),
                    "pct_pred_c2": float(100.0 * np.mean(p_sub == 2)),
                    "pct_pred_c3": float(100.0 * np.mean(p_sub == 3)),
                }
            )
    df_raw_eval = pd.DataFrame(raw_rows)
    df_raw_eval.to_csv(table_dir / "evaluacion_dataset_crudo.csv", index=False)

    _plot_synthetic_and_raw_figure(
        df_syn, df_raw_eval, fig_dir / "evaluacion_sinteticos_y_crudo.png"
    )
    return df_syn, df_raw_eval


def run_all_evaluations(base_dir: Path = PROJECT_ROOT) -> None:
    """Orquesta toda la Fase 4 de evaluación y variación estadística."""
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    table_dir = results_dir / "tablas"
    fig_dir = results_dir / "figuras"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    results_df = pd.read_csv(results_dir / "resultados_gridsearch.csv")
    top5_payload = json.loads((models_dir / "top5_configs.json").read_text(encoding="utf-8"))
    decision = json.loads((models_dir / "decision_balanceo.json").read_text(encoding="utf-8"))

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["cluster"].to_numpy(dtype=np.int64)
    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val_df["cluster"].to_numpy(dtype=np.int64)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["cluster"].to_numpy(dtype=np.int64)

    plot_hyperparameter_evolution(results_df, fig_dir / "evolucion_hiperparametros.png")

    df_comp, model_final = evaluate_fidelity_and_test(
        X_train, y_train, X_val, y_val, X_test, y_test,
        top5_payload, decision, base_dir, table_dir, fig_dir,
    )

    model_1se = joblib.load(models_dir / "mejor_modelo_mlp.joblib")
    generate_learning_and_loss_curves(
        X_train, y_train, top5_payload, results_df, model_1se, table_dir, fig_dir
    )

    df_ood = evaluate_ood_noise(
        model_final, test_df, data_dir / "ruido_descartado.csv", table_dir, fig_dir
    )

    model_specs = _unique_top_models(top5_payload)
    df_ms = run_multiseed_variation(
        model_specs, X_train, y_train, X_val, y_val, table_dir, fig_dir, n_seeds=20
    )

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    df_rskf, df_nb = run_rskf_and_nadeau_bengio(
        model_specs, X_tv, y_tv, table_dir, fig_dir
    )

    df_boot = run_bootstrap_test(
        base_dir, X_test, y_test, str(decision["variante_elegida"]), table_dir, fig_dir, n_boot=1000
    )

    df_syn, df_raw_eval = evaluate_synthetic_and_raw_dataset(base_dir)

    print("=" * 72)
    print("FASE 4 — EVALUACIÓN, OVERFITTING Y VARIACIÓN ESTADÍSTICA COMPLETADA")
    print("=" * 72)
    print("\n[Comparativa Fidelidad Train / Val / Test]")
    print(df_comp[["modelo", "train_f1_macro", "val_f1_macro", "test_f1_macro", "gap_f1_train_val", "evaluado_en_test"]].to_string(index=False))
    print("\n[Resumen Test Único]")
    print(pd.read_csv(table_dir / "metricas_test_resumen.csv").to_string(index=False))
    print("\n[OOD Ruido Descartado]")
    print(df_ood[["grupo", "n", "entropia_media", "prob_max_media", "pct_alta_incertidumbre_h_gt_03"]].to_string(index=False))
    print("\n[Variación 20 Semillas en Validation]")
    print(df_ms[["modelo", "f1_macro_media", "f1_macro_std", "f1_macro_ic95_inf", "f1_macro_ic95_sup"]].to_string(index=False))
    print("\n[RSKF 5x10 (Train+Val) + Prueba Nadeau-Bengio]")
    print(df_rskf[["modelo", "f1_macro_media", "f1_macro_std", "f1_macro_ic95_inf", "f1_macro_ic95_sup"]].to_string(index=False))
    print(df_nb[["modelo_comparado", "diff_media_f1_macro", "t_nadeau_bengio", "p_valor_bilateral", "significativo_005"]].to_string(index=False))
    print("\n[Bootstrap Test B=1,000 (F1-Macro y Accuracy)]")
    print(df_boot[df_boot["metrica"].isin(["accuracy", "f1_macro", "roc_auc_ovr"])].to_string(index=False))
    print("\n[Datos Sintéticos (Interpolación + Perturbación Gaussiana)]")
    print(df_syn[df_syn["modelo"] == "MLP Ganador 1-SE"][["regimen", "n_muestras", "pct_dentro_manifold_dbscan", "f1_macro", "accuracy", "entropia_media"]].to_string(index=False))
    print("\n[Dataset Crudo Completo (Dataset.csv, 13,611 filas)]")
    print(df_raw_eval[df_raw_eval["modelo"] == "MLP Ganador 1-SE"][["subconjunto_crudo", "n_registros", "n_errores_estrato", "f1_macro", "accuracy", "entropia_media", "pct_alta_incertidumbre_h_gt_03"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Configura argumentos de línea de comandos para ``evaluacion_experimentos.py``."""
    parser = argparse.ArgumentParser(
        description="Evaluación experimental, control de overfitting y variación estadística."
    )
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    run_all_evaluations(base_dir=args.base_dir)


if __name__ == "__main__":
    main()
