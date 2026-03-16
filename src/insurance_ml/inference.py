from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class PredictConfig:
    model_bundle: Path
    input_parquet: Path
    output_parquet: Path


def predict_with_bundle(config: PredictConfig) -> Path:
    bundle = joblib.load(config.model_bundle)

    claim_model = bundle["claim_model"]
    type_model = bundle["type_model"]
    label_encoder = bundle["label_encoder"]
    feature_cols = bundle["feature_cols"]
    threshold = float(bundle.get("threshold", 0.5))

    df = pd.read_parquet(config.input_parquet).copy()
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

    # Ensure all trained features exist
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    X = df[feature_cols]

    p_claim = claim_model.predict_proba(X)[:, 1]
    y_claim_pred = (p_claim >= threshold).astype(int)

    y_type_pred = np.array(["NoClaim"] * len(X), dtype=object)
    claim_idx = np.flatnonzero(y_claim_pred == 1)
    if len(claim_idx) > 0:
        X_sub = X.iloc[claim_idx]
        p_type_sub = type_model.predict_proba(X_sub)
        y_type_idx = np.argmax(p_type_sub, axis=1)

        classes = np.array(type_model.classes_) if hasattr(type_model, "classes_") else np.arange(p_type_sub.shape[1])
        if np.issubdtype(classes.dtype, np.integer):
            y_type_lbl = label_encoder.inverse_transform(classes[y_type_idx])
        else:
            y_type_lbl = classes[y_type_idx]
        y_type_pred[claim_idx] = pd.Series(y_type_lbl).astype(str).values

    out = df[[c for c in ["resident_id", "week_start"] if c in df.columns]].copy()
    out["p_claim_pred"] = p_claim
    out["y_claim_pred"] = y_claim_pred
    out["y_type_pred"] = y_type_pred
    out["threshold_used"] = threshold

    config.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.output_parquet, index=False)
    return config.output_parquet
