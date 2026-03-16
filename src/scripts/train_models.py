from __future__ import annotations

from pathlib import Path

from insurance_ml.training import TrainConfig, train_and_log


def main() -> None:
    data_dir = Path("data")
    dataset_file = "weekly_model_dataset.parquet"
    model_dir = Path("models")
    mlflow_tracking_uri = ""
    mlflow_experiment = "tricura_claims"
    random_state = 42

    cfg = TrainConfig(
        data_dir=data_dir,
        dataset_file=dataset_file,
        model_dir=model_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        random_state=random_state,
    )

    out = train_and_log(cfg)
    print("Training done")
    print(out)


if __name__ == "__main__":
    main()
