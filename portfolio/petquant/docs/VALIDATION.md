# PETQuant reconstruction validation notes

## Status

**Portfolio reconstruction: suitable for engineering demonstration after review.**  
**Historical ML performance: not clinically validated.**

This repository deliberately separates reproducible public engineering examples from the private/cloud-connected development environment used during the original PETQuant work.

## Historical source inventory

The reconstruction reviewed these Google Colab artifacts:

- `petquant_v2_training.ipynb`
- `PETQuant_Validation_v1.ipynb`
- `PETQuant_Synthetic_Validation_v1.ipynb`
- `petquant_ml_events_export.ipynb`
- `pet-quant_advisor_monitor.py` (stored by Colab as notebook JSON despite the `.py` name)

A dedicated Drive folder named `petquant-ml/notebooks` also contains a zero-byte `petquant_v2_training.ipynb`; that placeholder is **not** the canonical training source. The non-empty Colab notebook was used for this reconstruction.

## High-impact findings

### 1. Training sample was far too small for a performance claim

The historical training notebook contained four rows. Its split fell back to a random split with three training rows and one validation row. A 100% validation accuracy on one case is not a meaningful estimate of generalization and must not be presented as model performance.

### 2. Nested dose representation caused silent feature loss

`fdg_dose_mci` appeared as nested objects such as a mapping with `float` and `integer` fields. The historical notebook called `pd.to_numeric(..., errors="coerce")`, which converted those mappings to missing values. The public reconstruction includes an explicit parser and tests for this representation.

### 3. Entirely missing optional features produced warnings

`weight_kg` and `expected_dose_mci` were entirely missing in the four-row training frame. Median imputation therefore produced `NaN` and runtime warnings. The reconstructed code keeps all-missing optional fields explicit instead of pretending they were successfully imputed.

### 4. Historical validation used rule labels as the reference

The validation notebooks compared ML predictions with `rule_label`. That is useful for rule/ML agreement analysis, but `rule_label` is not independent clinical ground truth.

One historical snapshot contained 11 paired rows and only one reference-unreliable row, with zero false-reliable predictions in that tiny subset.

A later mixed event set contained 416 paired rows. Before the score gate, 41 of 140 rule-unreliable cases were predicted reliable (29.3% false-reliable rate). A post-hoc gate changed `reliable` predictions with scores below 90 to `caution`, reducing the same-sample false-reliable count to zero and raising same-sample accuracy from about 0.50 to 0.62.

That result is a useful safety-engineering observation, **not an independently validated clinical threshold**, because the threshold was evaluated on the same data used to motivate it.

## Public portfolio policy

The public version should therefore claim only that PETQuant demonstrates:

- safety-oriented model evaluation,
- explicit false-reassurance measurement,
- conservative confidence gating,
- data-contract normalization,
- reproducible testing,
- separation of deterministic rules from advisory ML,
- and awareness of clinical validation limits.

It must **not** claim diagnostic accuracy, clinical effectiveness, regulatory clearance, or validated patient-care performance.

## Next validation steps for a real model

1. Define independent, expert-reviewed outcome labels that are not derived from the same rules being evaluated.
2. Accumulate an adequately sized dataset with documented inclusion/exclusion criteria.
3. Freeze the feature contract and correct nested numeric extraction before training.
4. Use train/validation/test separation or nested cross-validation appropriate to the available sample size.
5. Pre-register safety thresholds before evaluating the held-out test set.
6. Report class support, confidence intervals, calibration, sensitivity for unsafe cases, and false-reliable rate.
7. Validate across scanner/site/protocol strata before any clinical-use claim.
