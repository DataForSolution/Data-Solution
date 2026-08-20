# Automobile insurance fraud reconstruction audit

## Source inventory

Reviewed historical artifacts:

- `Data430BayesianClassification.ipynb`
- `DATA 430 – Assignment 1: Logistic Regression.ipynb`
- Drive file `fraud_oracle.csv`

The dataset is also available from a current Figshare record under CC BY 4.0.

## Local data audit

The connected Drive copy contains:

- 15,420 rows,
- 33 columns,
- 14,497 non-fraud claims,
- 923 fraud claims,
- fraud prevalence 5.9857%,
- years 1994, 1995, and 1996,
- no duplicate rows,
- no conventional null values,
- 15,420 unique `PolicyNumber` values.

Local SHA-256 of the audited Drive copy:

`8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6`

The repository does not redistribute that CSV.

## Historical Logistic Regression findings

### Overall accuracy hid complete minority-class failure

Several saved reports show about 93.5–94% accuracy while fraud recall is between 0% and about 1%.

One report shows:

- non-fraud support: 2,887,
- fraud support: 197,
- fraud precision: 0.25,
- fraud recall: 0.01,
- fraud F1: 0.01.

Other cells predict no fraud examples at all and emit undefined-metric warnings.

Because fraud prevalence is low, overall accuracy is not an adequate primary metric.

### Inconsistent model lineage

The notebook builds an imbalanced-learn pipeline with:

- numeric scaling,
- one-hot categorical encoding,
- SMOTE,
- Logistic Regression.

Later cells construct and evaluate separate scikit-learn pipelines and manually preprocessed matrices. Therefore the final saved metrics cannot be attributed to one clean estimator lineage.

### Invalid GridSearch combinations

The parameter grid tries both `l1` and `l2` penalties with default Logistic Regression solver behavior. Saved output shows 25 of 50 fits failing because `lbfgs` does not support `l1`.

### ROC-AUC was computed two different ways

One cell computes ROC-AUC from hard predictions and reports about 0.4998, which does not measure ranking quality.

A later probability-based calculation reports about 0.8229 on the reused random test split.

The reconstruction always computes ROC-AUC and average precision from continuous positive-class scores.

## Historical Naive Bayes findings

The notebook:

1. label-encodes every object column,
2. feeds the resulting integer codes into `MultinomialNB`.

For nominal values such as vehicle make, marital status, policy type, and day/month, those integer codes do not have valid count or ordinal meaning. Multinomial Naive Bayes is therefore a poor fit for that representation.

The saved positive-class metrics include approximately:

- precision: 0.0691,
- recall: 0.5298,
- F1: 0.1222,
- accuracy: 0.5311.

The earlier weighted precision of about 0.89 is misleading for the minority fraud class.

The reconstruction does not preserve this Naive Bayes design as a recommended model.

## Chronological reconstruction

The dataset spans three years, so the portfolio uses:

- train: 1994 (6,142 rows; fraud prevalence 6.66%),
- validation: 1995 (5,195 rows; fraud prevalence 5.79%),
- test: 1996 (4,083 rows; fraud prevalence 5.22%).

`PolicyNumber` is removed as a unique row identifier. `Year` is used for splitting and not passed to the estimator.

A small Logistic Regression candidate set is compared on 1995 average precision. The best audited candidate is unweighted Logistic Regression with `C=0.01`.

On 1995:

- ROC-AUC ≈ 0.7462,
- PR-AUC ≈ 0.1113.

At threshold 0.5, it predicts no fraud claims. Selecting the threshold that maximizes F1 on 1995 gives approximately 0.0737.

Frozen evaluation on 1996:

- ROC-AUC ≈ 0.7442,
- PR-AUC ≈ 0.1106,
- precision ≈ 0.1006,
- recall ≈ 0.6714,
- F1 ≈ 0.1749,
- balanced accuracy ≈ 0.6704,
- TN 2,591,
- FP 1,279,
- FN 70,
- TP 143.

These results are intentionally modest. They show why threshold-aware evaluation matters and why this historical dataset should not be presented as a modern production fraud benchmark.

## Responsible interpretation

The dataset is historical and narrow. The reconstruction does not claim:

- modern fraud prevalence,
- business cost optimization,
- calibrated financial risk,
- production readiness,
- fairness across protected groups,
- or transferability to current insurance processes.

A real deployment would require current data, explicit cost matrices, temporal drift tests, calibration, fairness analysis, investigation-capacity constraints, and external validation.
