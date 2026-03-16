from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .utils import dump_json


@dataclass
class TrainConfig:
    data_dir: Path
    dataset_file: str = "weekly_model_dataset.parquet"
    model_dir: Path = Path("models")
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "tricura_claims"
    random_state: int = 42


def _time_split(df: pd.DataFrame) -> pd.DataFrame:
    weeks = np.array(sorted(df["week_start"].dropna().unique()))
    if len(weeks) < 10:
        raise ValueError("Not enough weekly periods for train/val/test split")

    train_end = int(len(weeks) * 0.70)
    val_end = int(len(weeks) * 0.85)

    train_weeks = weeks[:train_end]
    val_weeks = weeks[train_end:val_end]

    out = df.copy()
    out["split"] = np.where(
        out["week_start"].isin(train_weeks),
        "train",
        np.where(out["week_start"].isin(val_weeks), "val", "test"),
    )
    return out


def _select_features(df: pd.DataFrame) -> list[str]:
    explicit_drop = {
        "resident_id",
        "week_start",
        "week_end",
        "split",
        "target_claim_next_week",
        "target_claim_type_next_week",
        "claim_count",
        "incident_type_primary",
        "incident_type_list",
        "injury_count_sum",
        "admission_count_sum",
        "transfer_count_sum",
        "had_injury_week",
        "had_admission_week",
        "had_transfer_week",
        "is_claim_week",
    }

    pattern_drop_prefixes = (
        "claim_",
        "incident_type_",
        "injury_count_",
        "admission_count_",
        "transfer_count_",
        "had_",
        "is_claim_",
    )

    leak_like = [c for c in df.columns if c.startswith(pattern_drop_prefixes) and c not in {"target_claim_next_week", "target_claim_type_next_week"}]
    drop_cols = explicit_drop.union(set(leak_like))
    return [c for c in df.columns if c not in drop_cols]


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return prep, num_cols, cat_cols


def _tune_threshold_f1(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 37)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return best_t, best_f1


def train_and_log(config: TrainConfig) -> dict:
    dataset_path = config.data_dir / config.dataset_file
    df = pd.read_parquet(dataset_path).copy()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["resident_id", "week_start"]).sort_values(["resident_id", "week_start"]).reset_index(drop=True)

    df = _time_split(df)
    feature_cols = _select_features(df)

    X = df[feature_cols]
    y_claim = df["target_claim_next_week"].astype(int)

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train_claim, y_val_claim, y_test_claim = y_claim[train_mask], y_claim[val_mask], y_claim[test_mask]

    preprocessor, num_cols, cat_cols = _build_preprocessor(X_train)

    claim_model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    claim_model.fit(X_train, y_train_claim)

    p_val_claim = claim_model.predict_proba(X_val)[:, 1]
    p_test_claim = claim_model.predict_proba(X_test)[:, 1]
    best_t, _ = _tune_threshold_f1(y_val_claim.values, p_val_claim)

    type_train_mask = train_mask & (y_claim == 1)
    type_val_mask = val_mask & (y_claim == 1)
    type_test_mask = test_mask & (y_claim == 1)

    X_train_type, X_val_type, X_test_type = X[type_train_mask], X[type_val_mask], X[type_test_mask]
    y_train_type_raw = df.loc[type_train_mask, "target_claim_type_next_week"].astype(str)
    y_val_type_raw = df.loc[type_val_mask, "target_claim_type_next_week"].astype(str)
    y_test_type_raw = df.loc[type_test_mask, "target_claim_type_next_week"].astype(str)

    if len(y_train_type_raw) == 0:
        raise ValueError("No positive claim rows in training for type model")

    label_encoder = LabelEncoder()
    y_train_type = label_encoder.fit_transform(y_train_type_raw)
    y_val_type = label_encoder.transform(y_val_type_raw) if len(y_val_type_raw) else np.array([], dtype=int)
    y_test_type = label_encoder.transform(y_test_type_raw) if len(y_test_type_raw) else np.array([], dtype=int)

    type_model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    multi_class="auto",
                    solver="lbfgs",
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    type_model.fit(X_train_type, y_train_type)

    # Baseline (for reference)
    baseline_claim = Pipeline([("prep", preprocessor), ("clf", DummyClassifier(strategy="prior", random_state=config.random_state))])
    baseline_claim.fit(X_train, y_train_claim)

    metrics = {
        "claim_val_roc_auc": float(roc_auc_score(y_val_claim, p_val_claim) if y_val_claim.nunique() > 1 else np.nan),
        "claim_val_pr_auc": float(average_precision_score(y_val_claim, p_val_claim)),
        "claim_test_roc_auc": float(roc_auc_score(y_test_claim, p_test_claim) if y_test_claim.nunique() > 1 else np.nan),
        "claim_test_pr_auc": float(average_precision_score(y_test_claim, p_test_claim)),
        "claim_threshold": float(best_t),
    }

    if len(y_val_type):
        p_val_type = type_model.predict_proba(X_val_type)
        pred_val_type = np.argmax(p_val_type, axis=1)
        metrics["type_val_accuracy"] = float(accuracy_score(y_val_type, pred_val_type))
        metrics["type_val_f1_macro"] = float(f1_score(y_val_type, pred_val_type, average="macro", zero_division=0))
        metrics["type_val_balanced_accuracy"] = float(balanced_accuracy_score(y_val_type, pred_val_type))
        metrics["type_val_log_loss"] = float(log_loss(y_val_type, p_val_type, labels=np.arange(len(label_encoder.classes_))))

    if len(y_test_type):
        p_test_type = type_model.predict_proba(X_test_type)
        pred_test_type = np.argmax(p_test_type, axis=1)
        metrics["type_test_accuracy"] = float(accuracy_score(y_test_type, pred_test_type))
        metrics["type_test_f1_macro"] = float(f1_score(y_test_type, pred_test_type, average="macro", zero_division=0))
        metrics["type_test_balanced_accuracy"] = float(balanced_accuracy_score(y_test_type, pred_test_type))
        metrics["type_test_log_loss"] = float(log_loss(y_test_type, p_test_type, labels=np.arange(len(label_encoder.classes_))))

    bundle = {
        "claim_model": claim_model,
        "type_model": type_model,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "threshold": best_t,
        "train_feature_count": len(feature_cols),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dataset_file": config.dataset_file,
    }

    config.model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = config.model_dir / "model_bundle.pkl"
    joblib.dump(bundle, bundle_path)

    metrics_path = config.model_dir / "metrics.json"
    dump_json(metrics_path, metrics)

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name="logit_champion"):
        mlflow.log_params(
            {
                "dataset_file": config.dataset_file,
                "model_family": "logistic_regression",
                "feature_count": len(feature_cols),
                "num_cols": len(num_cols),
                "cat_cols": len(cat_cols),
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)})
        mlflow.log_artifact(str(bundle_path), artifact_path="bundle")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        mlflow.sklearn.log_model(claim_model, artifact_path="claim_model")
        mlflow.sklearn.log_model(type_model, artifact_path="type_model")

    return {
        "bundle_path": str(bundle_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }
