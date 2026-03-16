# Prototype Notebooks

This folder contains iterative notebook-based work for feature engineering, model training, and business validation.

## Data Convention

All notebook artifacts in this folder read/write from `../data`.


## Notebook Guide

### `feature_engineering.ipynb`
Builds the weekly feature-store dataset from base parquet tables (residents, vitals, incidents, injuries, admissions, transfers) and exports `weekly_model_dataset.parquet`.

### `feature_store_eda.ipynb`
Performs exploratory analysis on `weekly_model_dataset.parquet` to inspect distribution, quality, and feature behavior.

### `weekly_claims_ml.ipynb`
Trains and evaluates the weekly ML models:
- claim occurrence prediction (binary)
- claim type prediction (multiclass, conditional workflow)

It also exports prediction datasets and metric snapshots used by downstream validation.

### `business_validation.ipynb`
Evaluates model-driven intervention strategies in business-cost terms (best strategy vs baseline vs do-nothing) using exported prediction files.

## Model Conclusions (Prototype)

From the current notebook outputs in this repo snapshot:
- The business validation notebook selected `logit_balanced` as best strategy and `xgb_main` as baseline in validation output.
- The prototype demonstrates an end-to-end path from weekly feature-store generation to business-impact comparison.
