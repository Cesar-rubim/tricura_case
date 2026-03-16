# Tricura Claims Risk Prototype

## Core: Problem, Solution, Data, Weekly Design, Impact

### Problem We Solved
Tricura insures skilled nursing facilities and is exposed to resident-incident claims (falls, medication errors, wounds, return-to-hospital events, elopement, altercations).

The business challenge is not only to predict claims, but to reduce expected claims cost through earlier intervention, without relying on premium increases.

### Our Solution (High Level)
We framed the work as a **weekly resident-level risk pipeline** with two ML stages:

1. **Will this resident have a claim next week?** (binary classification)
2. **If yes, what is the most likely claim type next week?** (multiclass classification)

This gives both:
- a probability signal (`p_claim_pred`) for triage/intervention prioritization
- a likely incident-type signal (`y_type_pred`) to tailor intervention actions

### Our Solution (Core Equations)
For each resident-week record $i$, with features $x_i$:

1. **Claim occurrence model (binary)**
- Target: $y_i^{claim} \in \{0,1\}$, where 1 means at least one claim next week.
- Predicted probability:
  ```math
  p_i = P(y_i^{claim}=1 \mid x_i)
  ```
- Decision rule with threshold $\tau$:
  ```math
  \hat{y}_i^{claim} = \mathbb{1}[p_i \ge \tau]
  ```
- $\tau$ is tuned on validation data to maximize F1.

2. **Claim type model (multiclass, conditional)**
- Classes $k \in \mathcal{K}$ (incident types).
- Conditional probabilities:
  ```math
  q_{i,k} = P(y_i^{type}=k \mid x_i,\ y_i^{claim}=1)
  ```
- Type prediction for cases flagged as claim:
  ```math
  \hat{y}_i^{type} = \arg\max_{k \in \mathcal{K}} q_{i,k}
  ```
- For $\hat{y}_i^{claim}=0$, output `"NoClaim"`.

### Data We Considered
Primary tables used in feature engineering:
- `residents.parquet`
- `vitals.parquet`
- `incidents.parquet`
- `injuries.parquet`
- `hospital_admissions.parquet`
- `hospital_transfers.parquet`

Conceptually:
- Residents define enrollment windows and demographics (e.g., age proxy over time).
- Vitals provide physiological trends/statistics.
- Incidents and linked clinical context (injuries/admissions/transfers) define claim behavior.

### Weekly Aggregation (Core Design Choice)
We modeled everything on **resident-week granularity**:
- Build one row per resident per week (`week_start`, `week_end`) during active stay windows.
- Aggregate vitals into weekly summary stats (means/stds where applicable).
- Aggregate incidents and related outcomes into weekly claim-context columns.

Then create labels as **next-week targets**:
- `target_claim_next_week`: whether next week has any claim
- `target_claim_type_next_week`: primary incident type next week

Important leakage control:
- Current-week incident outcome fields (claim counts, incident type fields, injury/admission/transfer indicators) are dropped before training so the model predicts forward, not “reads the answer” from current-week outcomes.

### How We Measured Impact
We measured impact at two levels:

1. **Model quality metrics** (ML-level)
- Binary claim model: ROC-AUC, PR-AUC, threshold-tuned F1 behavior
- Claim type model: accuracy, macro F1, balanced accuracy, log-loss

2. **Business simulation metrics** (decision-level)
- Compare strategy outputs in cost terms (best strategy vs baseline vs do-nothing)
- Evaluate mitigated claim rows, strategy cost, baseline cost, and savings ratio

This is the key idea: success is not only classification performance, but whether model-driven actions improve expected cost outcomes versus baseline operational behavior.

### Business Impact Method (Detailed)
We evaluate policies as **decision strategies** over resident-weeks, not just classifiers.

#### 1. Per-row business quantities
For each row $i$:
- $c_i$: realized claim cost (0 if no claim)
- $a_{i,s} \in \{0,1\}$: whether strategy $s$ triggers intervention for row $i$
- $m_{i,s} \in [0,1]$: assumed mitigation effectiveness for that intervention under strategy $s$
- $k_{i,s} \ge 0$: intervention/operational cost for acting on row $i$

Mitigated claim cost under strategy $s$:
```math
\tilde{c}_{i,s} = c_i \cdot (1 - a_{i,s} \cdot m_{i,s})
```

Total per-row strategy cost:
```math
t_{i,s} = \tilde{c}_{i,s} + a_{i,s}\cdot k_{i,s}
```

#### 2. Aggregate by strategy and split
For each strategy $s$, we aggregate:
```math
\text{BaselineDoNothingCost}_s = \sum_i c_i
```
```math
\text{StrategyCost}_s = \sum_i t_{i,s}
```
```math
\text{Savings}_s = \text{BaselineDoNothingCost}_s - \text{StrategyCost}_s
```
```math
\text{SavingsRate}_s = \frac{\text{Savings}_s}{\text{BaselineDoNothingCost}_s}
```

Operational activity indicators:
- `intervened_rows`: $\sum_i a_{i,s}$
- `mitigated_claim_rows`: rows where intervention applies and claim impact is reduced

#### 3. Model-to-decision link
- `p_claim_pred` controls prioritization (who gets intervention first / above threshold).
- `y_type_pred` enables type-specific interventions/cost assumptions (e.g., different playbooks by incident type).
- Different strategy definitions correspond to different thresholds/ranking/assignment rules.

#### 4. Selection protocol
- Compute strategy costs on **validation** predictions.
- Rank strategies by minimum `strategy_cost` (or maximum savings).
- Select:
  - `best` = lowest validation strategy cost
  - `baseline` = next-best operational comparator
  - plus explicit `do_nothing`
- Report all three on validation and test summaries.

#### 5. Why this is the business metric
A model with better ROC-AUC can still be worse operationally if it over-triggers expensive interventions or misses high-cost events.  
Therefore, final acceptance is based on total-cost outcomes (`strategy_cost`, `savings_vs_do_nothing`, `savings_rate`), with ML metrics used as supporting diagnostics.

### Outputs
Main prototype artifacts (saved to `data/`):
- `weekly_model_dataset.parquet`
- `val_claim_predictions.parquet`, `test_claim_predictions.parquet`
- `val_claim_type_predictions.parquet`, `test_claim_type_predictions.parquet`
- `val_business_predictions.parquet`, `test_business_predictions.parquet` (if exported)
- `metrics_claim.parquet`, `metrics_claim_type.parquet`
- `business_validation_summary_3_approaches.parquet`

### Core Ideas to Keep
- Weekly resident-level framing enables operational planning cadence.
- Two-stage modeling supports both *who to prioritize* and *what kind of prevention to apply*.
- Leakage-aware feature design is essential for realistic forward prediction.
- Business-value validation is required; pure ML metrics are insufficient.

---

## Setup and Notebook-First Run

### Environment
Use Python 3.8 and a virtual environment.

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data and Artifacts Folder Convention
- Raw source data: `data/`
- Prototype notebook outputs/artifacts: `data/`

Prototype notebooks currently read/write directly in `data/`.
If you do not have the dataset locally, follow the instructions in [`data/README.md`](./data/README.md) to download and place the parquet files.

### Run Flow (Notebooks)
Open notebooks in `prototype/` and execute in this order:

1. `prototype/feature_engineering.ipynb`
- Builds weekly feature store dataset
- Produces `data/weekly_model_dataset.parquet`

2. `prototype/feature_store_eda.ipynb` (optional but recommended)
- Validates/inspects weekly dataset behavior

3. `prototype/weekly_claims_ml.ipynb`
- Trains/evaluates claim and claim-type models
- Exports prediction and metrics parquet artifacts

4. `prototype/business_validation.ipynb`
- Computes strategy-level cost impact
- Produces business validation summary output

---

## Work In Progress: `src/` Package + MLflow Track

This repository is actively migrating from notebook-first experimentation to a script/package pipeline under `src/`.

Current WIP components:
- `src/insurance_ml/feature_store.py`
  - Reproducible weekly feature-store generation
- `src/insurance_ml/training.py`
  - Train binary + multiclass models
  - Feature handling and threshold selection
  - MLflow logging of params/metrics/artifacts/models
- `src/insurance_ml/inference.py`
  - Bundle-based batch inference output
- `src/scripts/`
  - `build_feature_store.py`
  - `train_models.py`
  - `predict.py`

MLflow is already integrated in training code and documented in `docs/CI_PROCESS.md` for local tracking/server workflows.

This `src` track is intended to become the production-grade path (CI/CD-friendly, reproducible, and less notebook-dependent) while notebook artifacts remain useful for exploration and business communication.
