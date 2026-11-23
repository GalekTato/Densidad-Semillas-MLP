"""Optimización en rejilla de MLPClassifier, modelos base y ablación (Pregunta 3).

Implementa:
  1. Modelos base (LogisticRegression y NearestCentroid) en Pipeline sin fuga.
  2. GridSearchCV en dos etapas sobre ``train.csv`` original únicamente:
       - Etapa A (gruesa): 240 combinaciones x 5 folds = 1,200 entrenamientos.
       - Etapa B (fina): 24 combinaciones x 5 folds = 120 entrenamientos.
  3. Selección parsimoniosa mediante regla de 1 Error Estándar (1-SE) frente al
     Top-1 absoluto, serializando ``mejor_modelo_mlp.joblib`` y
     ``mejor_modelo_mlp_top1.joblib``.
  4. Experimento dedicado de número de épocas sin parada temprana
     (``early_stopping=False``).
  5. Ablación de balanceo de 3 brazos (sin balanceo / duplicación simple / SMOTE)
     evaluada y decidida EXCLUSIVAMENTE sobre ``validation.csv`` (``test.csv``
     no se carga ni participa en ninguna decisión).
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import ast
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import preprocesameinto as prep

RANDOM_STATE: int = 42
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS: list[str] = prep.FEATURE_COLUMNS

PARAM_COLS: list[str] = [
    "param_clf__hidden_layer_sizes",
    "param_clf__activation",
    "param_clf__solver",
    "param_clf__alpha",
    "param_clf__learning_rate_init",
    "param_clf__learning_rate",
    "param_clf__momentum",
    "param_clf__nesterovs_momentum",
    "param_clf__batch_size",
]


def count_mlp_params(
    hidden_layer_sizes: tuple[int, ...], n_in: int = 16, n_out: int = 4
) -> int:
    """Cuenta pesos y sesgos entrenables totales de un MLP denso."""
    dims = [n_in, *hidden_layer_sizes, n_out]
    return int(sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)))


def _parse_hidden(val: Any) -> tuple[int, ...]:
    """Convierte representación de ``hidden_layer_sizes`` a tupla de enteros."""
    if isinstance(val, tuple):
        return tuple(int(v) for v in val)
    if isinstance(val, list):
        return tuple(int(v) for v in val)
    if isinstance(val, str):
        parsed = ast.literal_eval(val)
        if isinstance(parsed, int):
            return (int(parsed),)
        return tuple(int(v) for v in parsed)
    return (int(val),)


def build_mlp_pipeline_from_params(
    params: dict[str, Any],
    random_state: int = RANDOM_STATE,
    early_stopping: bool = True,
    max_iter: int = 1000,
) -> Pipeline:
    """Construye un Pipeline(StandardScaler, MLPClassifier) desde un dict de parámetros."""
    hidden = _parse_hidden(params["hidden_layer_sizes"])
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=str(params["activation"]),
        solver=str(params["solver"]),
        alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]),
        learning_rate=str(params.get("learning_rate", "constant")),
        momentum=float(params.get("momentum", 0.9)),
        nesterovs_momentum=bool(params.get("nesterovs_momentum", True)),
        batch_size=int(params.get("batch_size", 128)),
        max_iter=int(max_iter),
        early_stopping=bool(early_stopping),
        validation_fraction=0.15,
        n_iter_no_change=15,
        tol=1e-4,
        shuffle=True,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", mlp)])


def row_to_clean_params(row: pd.Series) -> dict[str, Any]:
    """Extrae diccionario JSON-serializable de hiperparámetros desde una fila CV."""
    hidden = _parse_hidden(row["param_clf__hidden_layer_sizes"])
    return {
        "hidden_layer_sizes": list(hidden),
        "n_params": int(row["n_params"]),
        "activation": str(row["param_clf__activation"]),
        "solver": str(row["param_clf__solver"]),
        "alpha": float(row["param_clf__alpha"]),
        "learning_rate_init": float(row["param_clf__learning_rate_init"]),
        "learning_rate": str(row["param_clf__learning_rate"]),
        "momentum": float(row["param_clf__momentum"]),
        "nesterovs_momentum": bool(row["param_clf__nesterovs_momentum"]),
        "batch_size": int(row["param_clf__batch_size"]),
    }


def evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cv: StratifiedKFold,
    out_dir: Path,
    table_dir: Path,
) -> pd.DataFrame:
    """Entrena y evalúa Regresión Logística y Centroide Más Cercano en validation."""
    baselines: list[tuple[str, Pipeline, str]] = [
        (
            "Regresion Logistica",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0, max_iter=1000, random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            "baseline_logreg.joblib",
        ),
        (
            "Centroide Mas Cercano",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", NearestCentroid(metric="euclidean")),
                ]
            ),
            "baseline_nearest_centroid.joblib",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, pipe, fname in baselines:
        has_proba = hasattr(pipe.named_steps["clf"], "predict_proba")
        scoring = {
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
        if has_proba:
            scoring["roc_auc_ovr"] = "roc_auc_ovr"

        cv_res = cross_validate(
            clone(pipe), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        fitted = clone(pipe).fit(X_train, y_train)
        joblib.dump(fitted, out_dir / fname)

        pred_tr = fitted.predict(X_train)
        pred_va = fitted.predict(X_val)
        roc_va = (
            float(
                roc_auc_score(
                    y_val, fitted.predict_proba(X_val), multi_class="ovr"
                )
            )
            if has_proba
            else np.nan
        )
        rows.append(
            {
                "modelo": name,
                "cv_f1_macro_mean": float(np.mean(cv_res["test_f1_macro"])),
                "cv_f1_macro_std": float(np.std(cv_res["test_f1_macro"], ddof=1)),
                "cv_accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
                "train_f1_macro": float(f1_score(y_train, pred_tr, average="macro")),
                "train_accuracy": float(accuracy_score(y_train, pred_tr)),
                "val_f1_macro": float(f1_score(y_val, pred_va, average="macro")),
                "val_accuracy": float(accuracy_score(y_val, pred_va)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, pred_va)),
                "val_f1_weighted": float(f1_score(y_val, pred_va, average="weighted")),
                "val_roc_auc_ovr": roc_va,
            }
        )
    df_base = pd.DataFrame(rows)
    df_base.to_csv(table_dir / "modelos_base_validation.csv", index=False)
    return df_base


def _normalize_cv_results(
    cv_results: dict[str, Any], stage_label: str
) -> pd.DataFrame:
    """Convierte ``cv_results_`` de GridSearchCV en DataFrame limpio y normalizado."""
    df = pd.DataFrame(cv_results)
    defaults: dict[str, Any] = {
        "param_clf__learning_rate": "constant",
        "param_clf__momentum": 0.9,
        "param_clf__nesterovs_momentum": True,
        "param_clf__batch_size": 128,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    df["param_clf__hidden_layer_sizes"] = df["param_clf__hidden_layer_sizes"].apply(
        lambda v: str(_parse_hidden(v))
    )
    df["param_clf__alpha"] = df["param_clf__alpha"].astype(float)
    df["param_clf__learning_rate_init"] = df["param_clf__learning_rate_init"].astype(
        float
    )
    df["param_clf__momentum"] = df["param_clf__momentum"].astype(float)
    df["param_clf__nesterovs_momentum"] = df["param_clf__nesterovs_momentum"].astype(
        bool
    )
    df["param_clf__batch_size"] = df["param_clf__batch_size"].astype(int)
    df["n_params"] = df["param_clf__hidden_layer_sizes"].apply(
        lambda s: count_mlp_params(_parse_hidden(s))
    )
    df["etapa_busqueda"] = stage_label
    return df


def run_two_stage_grid_search(
    X_train: np.ndarray, y_train: np.ndarray, cv: StratifiedKFold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta GridSearchCV en dos etapas (240 gruesa + 24 fina) sobre ``train.csv``."""
    base_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    max_iter=1000,
                    batch_size=128,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    tol=1e-4,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr": "roc_auc_ovr",
    }

    common_a = {
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (64, 32, 16)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [0.001, 0.01],
    }
    param_grid_a = [
        {**common_a, "clf__solver": ["adam"]},
        {
            **common_a,
            "clf__solver": ["sgd"],
            "clf__momentum": [0.5],
            "clf__nesterovs_momentum": [True],
            "clf__learning_rate": ["constant"],
        },
        {
            **common_a,
            "clf__solver": ["sgd"],
            "clf__momentum": [0.9],
            "clf__nesterovs_momentum": [True],
            "clf__learning_rate": ["constant"],
        },
        {
            **common_a,
            "clf__solver": ["sgd"],
            "clf__momentum": [0.9],
            "clf__nesterovs_momentum": [False],
            "clf__learning_rate": ["constant"],
        },
    ]

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs_a = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid_a,
            scoring=scoring,
            refit=False,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=0,
        )
        gs_a.fit(X_train, y_train)
    t_a = time.perf_counter() - t0
    df_a = _normalize_cv_results(gs_a.cv_results_, "A_gruesa")
    print(f"Etapa A completada: {len(df_a)} combinaciones en {t_a:.1f} s")

    # Etapa B fina alrededor del mejor absoluto y del mejor SGD de Etapa A
    best_a = df_a.sort_values("mean_test_f1_macro", ascending=False).iloc[0]
    sgd_a = df_a[df_a["param_clf__solver"] == "sgd"].sort_values(
        "mean_test_f1_macro", ascending=False
    )
    best_sgd_a = sgd_a.iloc[0]

    alpha_best = float(best_a["param_clf__alpha"])
    if alpha_best <= 1e-4:
        alpha_fine = [1e-4, 5e-4]
    elif alpha_best >= 1e-2:
        alpha_fine = [5e-3, 1e-2]
    else:
        alpha_fine = [5e-4, 2e-3]

    lr_best = float(best_a["param_clf__learning_rate_init"])
    lr_fine = [0.001, 0.003] if lr_best <= 0.001 else [0.005, 0.01]

    rama_b1: dict[str, Any] = {
        "clf__hidden_layer_sizes": [_parse_hidden(best_a["param_clf__hidden_layer_sizes"])],
        "clf__activation": [str(best_a["param_clf__activation"])],
        "clf__solver": [str(best_a["param_clf__solver"])],
        "clf__alpha": alpha_fine,
        "clf__learning_rate_init": lr_fine,
        "clf__batch_size": [64, 128, 256],
    }
    if str(best_a["param_clf__solver"]) == "sgd":
        rama_b1["clf__momentum"] = [float(best_a["param_clf__momentum"])]
        rama_b1["clf__nesterovs_momentum"] = [bool(best_a["param_clf__nesterovs_momentum"])]
        rama_b1["clf__learning_rate"] = [str(best_a["param_clf__learning_rate"])]

    rama_b2: dict[str, Any] = {
        "clf__hidden_layer_sizes": [_parse_hidden(best_sgd_a["param_clf__hidden_layer_sizes"])],
        "clf__activation": [str(best_sgd_a["param_clf__activation"])],
        "clf__solver": ["sgd"],
        "clf__alpha": [float(best_sgd_a["param_clf__alpha"])],
        "clf__learning_rate_init": [float(best_sgd_a["param_clf__learning_rate_init"])],
        "clf__learning_rate": ["constant", "adaptive"],
        "clf__momentum": [0.9, 0.95],
        "clf__nesterovs_momentum": [True],
        "clf__batch_size": [64, 128, 256],
    }

    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs_b = GridSearchCV(
            estimator=base_pipe,
            param_grid=[rama_b1, rama_b2],
            scoring=scoring,
            refit=False,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=0,
        )
        gs_b.fit(X_train, y_train)
    t_b = time.perf_counter() - t1
    df_b = _normalize_cv_results(gs_b.cv_results_, "B_fina")
    print(f"Etapa B completada: {len(df_b)} combinaciones en {t_b:.1f} s")

    combined = pd.concat([df_a, df_b], ignore_index=True)
    combined = combined.drop_duplicates(subset=PARAM_COLS, keep="first").reset_index(
        drop=True
    )
    combined = combined.sort_values(
        by=["mean_test_f1_macro", "n_params"], ascending=[False, True]
    ).reset_index(drop=True)
    return combined, df_b.sort_values("mean_test_f1_macro", ascending=False).reset_index(
        drop=True
    )


def apply_one_se_selection(
    combined_df: pd.DataFrame, n_splits: int = 5
) -> tuple[pd.DataFrame, pd.Series, pd.Series, float, float]:
    """Aplica la regla de 1 Error Estándar (1-SE) para elegir el modelo más simple."""
    top1_row = combined_df.iloc[0]
    s_star = float(top1_row["mean_test_f1_macro"])
    se_star = float(top1_row["std_test_f1_macro"]) / np.sqrt(float(n_splits))
    threshold = s_star - se_star

    df = combined_df.copy()
    df["dentro_1se"] = df["mean_test_f1_macro"] >= threshold

    candidates = df[df["dentro_1se"]].sort_values(
        by=["n_params", "param_clf__alpha", "mean_test_f1_macro"],
        ascending=[True, False, False],
    )
    one_se_row = candidates.iloc[0]
    df["seleccionado_1se"] = False
    df.loc[one_se_row.name, "seleccionado_1se"] = True
    df["seleccionado_top1"] = False
    df.loc[top1_row.name, "seleccionado_top1"] = True
    return df, one_se_row, top1_row, s_star, se_star


def run_epochs_experiment(
    one_se_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    table_dir: Path,
) -> None:
    """Estudia el efecto de ``max_iter`` y épocas sin parada temprana."""
    rows: list[dict[str, Any]] = []
    for m_iter in [50, 100, 200, 400, 800]:
        pipe = build_mlp_pipeline_from_params(
            one_se_params, random_state=RANDOM_STATE, early_stopping=False, max_iter=m_iter
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X_train, y_train)
        pred_tr = pipe.predict(X_train)
        pred_va = pipe.predict(X_val)
        f1_tr = float(f1_score(y_train, pred_tr, average="macro"))
        f1_va = float(f1_score(y_val, pred_va, average="macro"))
        rows.append(
            {
                "max_iter": m_iter,
                "n_iter_real": int(pipe.named_steps["clf"].n_iter_),
                "final_loss_train": float(pipe.named_steps["clf"].loss_),
                "train_f1_macro": f1_tr,
                "val_f1_macro": f1_va,
                "train_accuracy": float(accuracy_score(y_train, pred_tr)),
                "val_accuracy": float(accuracy_score(y_val, pred_va)),
                "gap_f1_train_val": f1_tr - f1_va,
            }
        )
    pd.DataFrame(rows).to_csv(table_dir / "experimento_epocas.csv", index=False)

    # Trayectoria detallada época a época (warm_start=True) hasta 400 épocas
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    epoch_rng = np.random.RandomState(RANDOM_STATE)
    mlp_ep = MLPClassifier(
        hidden_layer_sizes=_parse_hidden(one_se_params["hidden_layer_sizes"]),
        activation=str(one_se_params["activation"]),
        solver=str(one_se_params["solver"]),
        alpha=float(one_se_params["alpha"]),
        learning_rate_init=float(one_se_params["learning_rate_init"]),
        learning_rate=str(one_se_params["learning_rate"]),
        momentum=float(one_se_params["momentum"]),
        nesterovs_momentum=bool(one_se_params["nesterovs_momentum"]),
        batch_size=int(one_se_params["batch_size"]),
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        shuffle=True,
        random_state=epoch_rng,
    )
    traj_rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for ep in range(1, 401):
            mlp_ep.fit(X_tr_s, y_train)
            if ep == 1 or ep % 5 == 0:
                p_tr = mlp_ep.predict(X_tr_s)
                p_va = mlp_ep.predict(X_va_s)
                prob_va = mlp_ep.predict_proba(X_va_s)
                traj_rows.append(
                    {
                        "epoch": ep,
                        "train_loss": float(mlp_ep.loss_),
                        "val_log_loss": float(log_loss(y_val, prob_va)),
                        "train_f1_macro": float(f1_score(y_train, p_tr, average="macro")),
                        "val_f1_macro": float(f1_score(y_val, p_va, average="macro")),
                        "train_accuracy": float(accuracy_score(y_train, p_tr)),
                        "val_accuracy": float(accuracy_score(y_val, p_va)),
                    }
                )
    pd.DataFrame(traj_rows).to_csv(
        table_dir / "curva_epocas_sin_early_stopping.csv", index=False
    )


def run_balancing_ablation_on_validation(
    one_se_params: dict[str, Any],
    X_train_orig: np.ndarray,
    y_train_orig: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_dup_path: Path,
    train_smote_path: Path,
    out_dir: Path,
    table_dir: Path,
) -> dict[str, Any]:
    """Compara 3 variantes (sin balanceo / duplicación / SMOTE) SOLO en validation."""
    df_dup = pd.read_csv(train_dup_path)
    df_smote = pd.read_csv(train_smote_path)

    arms: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        ("sin_balanceo", "Sin balanceo (train.csv)", X_train_orig, y_train_orig),
        (
            "duplicacion_simple",
            "Duplicacion simple (train_dup.csv)",
            df_dup[FEATURE_COLUMNS].to_numpy(dtype=float),
            df_dup["cluster"].to_numpy(dtype=np.int64),
        ),
        (
            "smote",
            "SMOTE k=5 (train_smote.csv)",
            df_smote[FEATURE_COLUMNS].to_numpy(dtype=float),
            df_smote["cluster"].to_numpy(dtype=np.int64),
        ),
    ]

    fitted_arms: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []
    for key, label, X_tr_arm, y_tr_arm in arms:
        pipe = build_mlp_pipeline_from_params(
            one_se_params, random_state=RANDOM_STATE, early_stopping=True, max_iter=1000
        )
        pipe.fit(X_tr_arm, y_tr_arm)
        fitted_arms[key] = pipe

        # Comparación justa en Train original y en Validation intacto
        pred_tr_orig = pipe.predict(X_train_orig)
        pred_va = pipe.predict(X_val)
        prob_va = pipe.predict_proba(X_val)

        rec_per_cls = recall_score(y_val, pred_va, average=None, labels=[0, 1, 2, 3])
        f1_per_cls = f1_score(y_val, pred_va, average=None, labels=[0, 1, 2, 3])
        rows.append(
            {
                "variante": key,
                "descripcion": label,
                "n_train_usado": int(len(X_tr_arm)),
                "n_iter_real": int(pipe.named_steps["clf"].n_iter_),
                "train_orig_f1_macro": float(
                    f1_score(y_train_orig, pred_tr_orig, average="macro")
                ),
                "train_orig_accuracy": float(accuracy_score(y_train_orig, pred_tr_orig)),
                "val_accuracy": float(accuracy_score(y_val, pred_va)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, pred_va)),
                "val_f1_macro": float(f1_score(y_val, pred_va, average="macro")),
                "val_f1_weighted": float(f1_score(y_val, pred_va, average="weighted")),
                "val_roc_auc_ovr": float(roc_auc_score(y_val, prob_va, multi_class="ovr")),
                "val_recall_c3": float(rec_per_cls[3]),
                "val_f1_c3": float(f1_per_cls[3]),
            }
        )

    df_ab = pd.DataFrame(rows)
    base_f1 = float(df_ab.loc[df_ab["variante"] == "sin_balanceo", "val_f1_macro"].iloc[0])
    best_row = df_ab.sort_values("val_f1_macro", ascending=False).iloc[0]
    if (
        str(best_row["variante"]) != "sin_balanceo"
        and (float(best_row["val_f1_macro"]) - base_f1) < 0.0005
    ):
        chosen_key = "sin_balanceo"
        motivo = (
            "Empate técnico en validation (diferencia < 0.0005 en F1-Macro); "
            "se conserva 'sin_balanceo' por parsimonia y ausencia de distorsión sintética."
        )
    else:
        chosen_key = str(best_row["variante"])
        motivo = f"Mayor val_f1_macro ({float(best_row['val_f1_macro']):.6f}) en validation.csv."

    df_ab["elegida_en_validation"] = df_ab["variante"] == chosen_key
    df_ab.to_csv(table_dir / "ablacion_balanceo_validation.csv", index=False)

    joblib.dump(fitted_arms[chosen_key], out_dir / "mejor_modelo_mlp_final.joblib")
    decision = {
        "variante_elegida": chosen_key,
        "criterio": "Mayor val_f1_macro en validation.csv (empate < 0.0005 favorece sin_balanceo)",
        "motivo": motivo,
        "metricas_validation": df_ab.to_dict(orient="records"),
    }
    (out_dir / "decision_balanceo.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


def run_training(base_dir: Path = PROJECT_ROOT) -> None:
    """Ejecuta toda la Fase 3 sin tocar ``test.csv``."""
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    table_dir = results_dir / "tablas"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["cluster"].to_numpy(dtype=np.int64)
    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val_df["cluster"].to_numpy(dtype=np.int64)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Modelos base
    df_base = evaluate_baselines(X_train, y_train, X_val, y_val, cv, models_dir, table_dir)

    # 2. Búsqueda en rejilla en dos etapas (siempre sobre train.csv original)
    combined_df, stage_b_df = run_two_stage_grid_search(X_train, y_train, cv)

    # 3. Regla 1-SE vs Top-1 absoluto
    results_df, one_se_row, top1_row, s_star, se_star = apply_one_se_selection(
        combined_df, n_splits=5
    )
    results_df.to_csv(results_dir / "resultados_gridsearch.csv", index=False)
    stage_b_df.to_csv(results_dir / "resultados_gridsearch_etapaB.csv", index=False)

    one_se_params = row_to_clean_params(one_se_row)
    top1_params = row_to_clean_params(top1_row)

    model_1se = build_mlp_pipeline_from_params(
        one_se_params, random_state=RANDOM_STATE, early_stopping=True, max_iter=1000
    ).fit(X_train, y_train)
    model_top1 = build_mlp_pipeline_from_params(
        top1_params, random_state=RANDOM_STATE, early_stopping=True, max_iter=1000
    ).fit(X_train, y_train)

    joblib.dump(model_1se, models_dir / "mejor_modelo_mlp.joblib")
    joblib.dump(model_top1, models_dir / "mejor_modelo_mlp_top1.joblib")

    # Top-5 configuraciones de la rejilla + referencia 1-SE
    top5_list = [row_to_clean_params(results_df.iloc[i]) for i in range(min(5, len(results_df)))]
    for idx_t, cfg in enumerate(top5_list, start=1):
        cfg["rank_grid"] = idx_t
        cfg["cv_f1_macro"] = float(results_df.iloc[idx_t - 1]["mean_test_f1_macro"])
        cfg["cv_f1_macro_std"] = float(results_df.iloc[idx_t - 1]["std_test_f1_macro"])

    one_se_params["cv_f1_macro"] = float(one_se_row["mean_test_f1_macro"])
    one_se_params["cv_f1_macro_std"] = float(one_se_row["std_test_f1_macro"])
    top1_params["cv_f1_macro"] = float(top1_row["mean_test_f1_macro"])
    top1_params["cv_f1_macro_std"] = float(top1_row["std_test_f1_macro"])

    top5_payload = {
        "s_star": s_star,
        "se_star": se_star,
        "umbral_1se": s_star - se_star,
        "ganador_1se": one_se_params,
        "top1_absoluto": top1_params,
        "top5": top5_list,
    }
    (models_dir / "top5_configs.json").write_text(
        json.dumps(top5_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. Experimento dedicado de #épocas (early_stopping=False)
    run_epochs_experiment(one_se_params, X_train, y_train, X_val, y_val, table_dir)

    # 5. Ablación de balanceo 3 brazos en validation.csv
    decision = run_balancing_ablation_on_validation(
        one_se_params,
        X_train,
        y_train,
        X_val,
        y_val,
        data_dir / "train_dup.csv",
        data_dir / "train_smote.csv",
        models_dir,
        table_dir,
    )

    print("=" * 72)
    print("FASE 3 — OPTIMIZACIÓN MLP Y ABLACIÓN EN VALIDATION COMPLETADA")
    print("=" * 72)
    print("\n[Modelos Base en Validation]")
    print(df_base.to_string(index=False))
    print(
        f"\n[Regla 1-SE] Mejor CV F1-Macro = {s_star:.6f} | SE = {se_star:.6f} | "
        f"Umbral 1-SE = {s_star - se_star:.6f}"
    )
    print(f"Top-1 absoluto: {top1_params} (n_iter={model_top1.named_steps['clf'].n_iter_})")
    print(f"Ganador 1-SE:   {one_se_params} (n_iter={model_1se.named_steps['clf'].n_iter_})")
    print(
        f"Validation Ganador 1-SE -> F1-Macro={f1_score(y_val, model_1se.predict(X_val), average='macro'):.6f}, "
        f"Acc={accuracy_score(y_val, model_1se.predict(X_val)):.6f}"
    )
    print("\n[Ablación de 3 Brazos en Validation]")
    print(pd.read_csv(table_dir / "ablacion_balanceo_validation.csv").to_string(index=False))
    print(f"Decisión fijada antes de test: {decision['variante_elegida']} — {decision['motivo']}")


def parse_args() -> argparse.Namespace:
    """Configura argumentos de línea de comandos para ``entrenamiento.py``."""
    parser = argparse.ArgumentParser(
        description="Optimización en rejilla de MLPClassifier, modelos base y ablación."
    )
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    """Punto de entrada CLI."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    run_training(base_dir=args.base_dir)


if __name__ == "__main__":
    main()
