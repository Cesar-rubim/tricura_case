from __future__ import annotations

from pathlib import Path

from insurance_ml.inference import PredictConfig, predict_with_bundle


def main() -> None:
    model_bundle = Path("models/model_bundle.pkl")
    input_parquet = Path("data/weekly_model_dataset.parquet")
    output_parquet = Path("data/predictions.parquet")

    cfg = PredictConfig(
        model_bundle=model_bundle,
        input_parquet=input_parquet,
        output_parquet=output_parquet,
    )

    out = predict_with_bundle(cfg)
    print(f"Predictions saved to: {out}")


if __name__ == "__main__":
    main()
