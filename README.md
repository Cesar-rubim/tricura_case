# Tricura Claims Risk Prototype

## Core: Problem, Solution, Data, Weekly Design, Impact

### Problem We Solved

Tricura insures skilled nursing facilities and is exposed to
resident‑incident claims such as:

-   Falls
-   Medication errors
-   Wounds / pressure injuries
-   Return‑to‑hospital events
-   Elopement / wandering
-   Altercations

The business challenge is **not only predicting claims**, but **reducing
expected claim costs through early intervention**, without relying on
premium increases.

------------------------------------------------------------------------

# Our Solution

We frame the problem as a **weekly resident‑level risk prediction
pipeline** with two machine‑learning stages.

### Stage 1 --- Claim Occurrence Prediction

Predict whether a resident will generate **any claim next week**.

### Stage 2 --- Claim Type Prediction

If a claim is predicted, estimate the **most likely claim type**.

This produces two operational signals:

-   `p_claim_pred`: probability of a claim next week
-   `y_type_pred`: predicted claim type

These signals support:

-   **triage / prioritization**
-   **type‑specific interventions**

------------------------------------------------------------------------

# Core Modeling Equations

For each resident‑week observation ( i ) with features ( x_i ):

------------------------------------------------------------------------

## 1. Claim Occurrence Model (Binary)

Target variable

$$
y_i^{\text{claim}} \in \{0,1\}
$$

where (1) indicates that at least one claim occurs in the following
week.

Predicted probability

$$
p_i = P(y_i^{\text{claim}} = 1 \mid x_i)
$$

Decision rule with threshold ( `\tau `{=tex})

$$
\hat{y}_i^{\text{claim}} =
\begin{cases}
1 & \text{if } p_i \ge \tau \\
0 & \text{otherwise}
\end{cases}
$$

The threshold ( `\tau `{=tex}) is tuned on validation data to maximize
**F1 score**.

------------------------------------------------------------------------

## 2. Claim Type Model (Multiclass Conditional Model)

Let the set of claim types be

$$
k \in \mathcal{K}
$$

Conditional probabilities are estimated as

$$
q_{i,k} =
P(y_i^{\text{type}} = k \mid x_i, y_i^{\text{claim}} = 1)
$$

Type prediction for rows flagged as claims

$$
\hat{y}_i^{\text{type}} =
\underset{k \in \mathcal{K}}{\arg\max} \; q_{i,k}
$$

If

$$
\hat{y}_i^{\text{claim}} = 0
$$

the output type is `"NoClaim"`.

------------------------------------------------------------------------

# Data Sources

Feature engineering uses the following tables:

-   `residents.parquet`
-   `vitals.parquet`
-   `incidents.parquet`
-   `injuries.parquet`
-   `hospital_admissions.parquet`
-   `hospital_transfers.parquet`

Conceptually:

-   **Residents** define stay windows and demographics
-   **Vitals** provide physiological trends
-   **Incidents and clinical outcomes** define claim behavior

------------------------------------------------------------------------

# Weekly Aggregation Design

All modeling occurs at **resident‑week granularity**.

We construct one row per:

    resident × week

during the resident's active stay.

### Weekly feature construction

-   Aggregate vitals into weekly statistics
-   Aggregate incident signals
-   Create contextual indicators for injury, admission, transfer

### Forward targets

Labels are defined for the **next week**:

-   `target_claim_next_week`
-   `target_claim_type_next_week`

### Leakage prevention

Current‑week outcome signals are removed before training, including:

-   incident type fields
-   claim counts
-   injury / admission / transfer outcomes

This ensures the model predicts **forward risk**, not current outcomes.

------------------------------------------------------------------------

# Business Impact Evaluation

Model performance is evaluated at two levels.

------------------------------------------------------------------------

## 1. ML Metrics

Claim occurrence model:

-   ROC‑AUC
-   PR‑AUC
-   F1

Claim type model:

-   Accuracy
-   Macro F1
-   Balanced accuracy
-   Log‑loss

------------------------------------------------------------------------

## 2. Operational Cost Simulation

Strategies are evaluated as **decision policies over resident‑weeks**.

------------------------------------------------------------------------

# Cost Framework

For each observation ( i ) and strategy ( s ):

  Symbol       Meaning
  ------------ --------------------------
  (c_i)        realized claim cost
  (a\_{i,s})   intervention triggered
  (m\_{i,s})   mitigation effectiveness
  (k\_{i,s})   intervention cost

------------------------------------------------------------------------

## Mitigated Claim Cost

$$
\tilde{c}_{i,s} =
c_i \cdot (1 - a_{i,s} \cdot m_{i,s})
$$

------------------------------------------------------------------------

## Total Row Cost

$$
t_{i,s} =
\tilde{c}_{i,s} + a_{i,s} \cdot k_{i,s}
$$

------------------------------------------------------------------------

# Strategy Aggregation

Baseline (no intervention)

$$
\text{BaselineCost}_s =
\sum_i c_i
$$

Strategy cost

$$
\text{StrategyCost}_s =
\sum_i t_{i,s}
$$

Savings

$$
\text{Savings}_s =
\text{BaselineCost}_s - \text{StrategyCost}_s
$$

Savings rate

$$
\text{SavingsRate}_s =
\frac{\text{Savings}_s}{\text{BaselineCost}_s}
$$

Operational indicators

-   `intervened_rows = Σ a_{i,s}`
-   `mitigated_claim_rows`

------------------------------------------------------------------------

# Model → Decision Link

The ML predictions drive operational policies.

  Prediction       Operational Role
  ---------------- ----------------------------
  `p_claim_pred`   prioritization signal
  `y_type_pred`    type‑specific intervention

Different **decision strategies** correspond to:

-   thresholds
-   ranking rules
-   intervention policies

------------------------------------------------------------------------

# Strategy Selection

Strategies are evaluated on **validation predictions**.

Selection rule:

1.  Rank strategies by minimum total cost
2.  Choose

-   **best strategy**
-   **baseline comparator**
-   **do‑nothing policy**

All strategies are reported on:

-   validation
-   test data

------------------------------------------------------------------------

# Why This Metric Matters

A model with higher ROC‑AUC can still perform worse operationally if:

-   it triggers too many expensive interventions
-   it misses high‑severity claims

Therefore the **primary evaluation metric is total operational cost**,
not only ML metrics.

------------------------------------------------------------------------

# Prototype Outputs

Saved artifacts include:

-   `weekly_model_dataset.parquet`
-   `val_claim_predictions.parquet`
-   `test_claim_predictions.parquet`
-   `val_claim_type_predictions.parquet`
-   `test_claim_type_predictions.parquet`
-   `metrics_claim.parquet`
-   `metrics_claim_type.parquet`
-   `business_validation_summary_3_approaches.parquet`

------------------------------------------------------------------------

# Key Ideas

-   Weekly resident‑level modeling supports operational planning.
-   Two‑stage prediction enables **who to prioritize** and **what
    intervention to apply**.
-   Leakage‑safe feature engineering is essential.
-   Business value must be evaluated through **cost simulation**, not
    only predictive metrics.

------------------------------------------------------------------------

# Notebook Workflow

Run notebooks in the following order.

### 1 --- Feature Engineering

    prototype/feature_engineering.ipynb

Produces

    data/weekly_model_dataset.parquet

### 2 --- Dataset Exploration (optional)

    prototype/feature_store_eda.ipynb

### 3 --- Model Training

    prototype/weekly_claims_ml.ipynb

Outputs predictions and metrics.

### 4 --- Business Validation

    prototype/business_validation.ipynb

Computes operational cost impact.

------------------------------------------------------------------------

# Production Track (Work in Progress)

The repository is transitioning toward a reproducible pipeline under:

    src/

Components include:

-   `feature_store.py`
-   `training.py`
-   `inference.py`

Scripts:

-   `build_feature_store.py`
-   `train_models.py`
-   `predict.py`


This structure supports **CI/CD pipelines and reproducible training**,
while notebooks remain useful for exploration and business
communication.