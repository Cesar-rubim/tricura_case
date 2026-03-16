from __future__ import annotations

from pathlib import Path

from insurance_ml.feature_store import FeatureStoreConfig, build_weekly_model_dataset


def main() -> None:
    data_dir = Path("data")
    date_start = "2023-10-01"
    date_end = "2024-12-30"
    output_file = "weekly_model_dataset.parquet"

    cfg = FeatureStoreConfig(
        data_dir=data_dir,
        date_start=date_start,
        date_end=date_end,
        output_file=output_file,
    )
    out = build_weekly_model_dataset(cfg)
    print(f"Feature store dataset saved to: {out}")


if __name__ == "__main__":
    main()
