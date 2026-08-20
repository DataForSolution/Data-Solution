# Automobile Insurance Fraud Classification — Portfolio Reconstruction

This project reconstructs 2024 DATA 430 coursework that used `fraud_oracle.csv` for Naive Bayes and Logistic Regression experiments.

The public version focuses on the defensible lesson in the historical work: **class imbalance makes overall accuracy a poor fraud-detection metric, and model ranking quality must be separated from the decision threshold used to flag claims.**

## Dataset and provenance

The audited CSV contains:

- 15,420 automobile insurance claims,
- 33 columns including the binary target `FraudFound_P`,
- 923 fraud-positive rows (5.99%),
- years 1994–1996,
- no conventional missing values,
- no duplicate rows,
- a unique sequential `PolicyNumber` per row.

A current public copy is available on Figshare as `fraud_oracle.csv`, authored by Mohamed Ibrahim and licensed CC BY 4.0:

https://figshare.com/articles/dataset/fraud_oracle_csv/24994233

The dataset is **not redistributed** in this repository. See [`data/README.md`](data/README.md).

## Historical audit highlights

- The Logistic Regression notebook reports about 94% accuracy while predicting essentially no fraud cases at the default threshold.
- One historical ROC-AUC calculation incorrectly uses hard class predictions and produces a value near 0.50; a later probability-based calculation gives about 0.823 on the reused random test split.
- The GridSearchCV mixes `penalty='l1'` with the default `lbfgs` solver, causing half of the cross-validation fits to fail.
- The notebook creates an SMOTE pipeline, but later evaluation follows separate preprocessing/model paths, making the final estimator lineage ambiguous.
- The Naive Bayes notebook ordinal-encodes nominal categorical variables with `LabelEncoder` and feeds those arbitrary integer codes to `MultinomialNB`.
- Weighted averages obscure the minority-class weakness: the saved Naive Bayes fraud precision is about 6.9% with F1 about 12.2%.
- Random splitting mixes claims from 1994–1996. The reconstruction uses a chronological holdout to better reflect forward deployment.

See [`docs/AUDIT.md`](docs/AUDIT.md).

## Reconstructed evaluation design

1. Validate the expected schema and target.
2. Drop `PolicyNumber` from predictors because it is a unique row identifier.
3. Use `Year` only for chronological partitioning:
   - 1994 → train,
   - 1995 → validation,
   - 1996 → final test.
4. Learn preprocessing only from the training year:
   - standardize numeric fields,
   - one-hot encode nominal categorical fields with unknown-category handling.
5. Compare a small, declared Logistic Regression candidate set on **validation PR-AUC**.
6. Select a classification threshold on validation data only.
7. Freeze model + threshold.
8. Evaluate once on 1996 with:
   - ROC-AUC,
   - PR-AUC / average precision,
   - precision,
   - fraud recall,
   - F1,
   - balanced accuracy,
   - specificity,
   - confusion counts.

## Audited reconstruction result

Using the local audited CSV and the deterministic configuration in this project:

| 1996 holdout metric | Value |
| --- | ---: |
| Fraud prevalence | 5.22% |
| ROC-AUC | 0.7442 |
| PR-AUC / average precision | 0.1106 |
| Default-threshold fraud recall | 0.0% |
| Validation-selected threshold | 0.0737 |
| Fraud recall at selected threshold | 67.1% |
| Fraud precision at selected threshold | 10.1% |
| F1 at selected threshold | 0.1749 |
| Balanced accuracy at selected threshold | 0.6704 |

The low absolute precision remains important. This is an educational imbalanced-classification project, not evidence of a production-ready fraud model.

## Repository structure

```text
src/fraud_portfolio/
  data.py         # schema audit + chronological split
  modeling.py     # leakage-safe preprocessing + Logistic Regression
  evaluation.py   # imbalance-aware binary metrics
  thresholds.py   # deterministic threshold selection

tests/
  test_data.py
  test_modeling.py
  test_evaluation.py
  test_thresholds.py

docs/
  AUDIT.md

data/
  README.md

notebooks/
  temporal_fraud_evaluation.ipynb
```

## Install and test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest
```

To rerun against the public dataset, download `fraud_oracle.csv` from the cited source into a local `data/` directory and execute the notebook or equivalent source functions.

## Interpretation boundary

A model that ranks fraud cases better than chance can still be unusable at a default threshold. Fraud operations require explicit investigation capacity, false-positive cost, missed-fraud cost, drift monitoring, calibration, and threshold governance. This project demonstrates evaluation mechanics; it does not certify the dataset or model for real insurance decisions.
