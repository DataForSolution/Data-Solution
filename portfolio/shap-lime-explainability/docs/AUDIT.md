# SHAP/LIME reconstruction audit

## Overall assessment

**Historical notebook: useful learning artifact, but not reliable as a clean reproducible reference.**  
**Reconstructed project: suitable for an explainable-AI engineering portfolio after verification.**

## Source inventory

The audit reviewed:

- Google Drive `shap-lime/shap_lime_v2 (1).ipynb` — 2,824,026 bytes;
- Google Drive `shap_lime_v2.ipynb` — 2,824,026 bytes;
- Google Drive `Data-450_shap_lime_v2.ipynb` — 4,719,499 bytes;
- public GitHub `DataForSolution/shap-lime/shap_lime_v2 (1).ipynb`.

The two 2.8 MB Drive files are byte-for-byte identical. Their Git blob SHA is
`fd7fa292c2de3d59ed0d12e81949bda998cba942`, which exactly matches the
notebook already stored in the public `DataForSolution/shap-lime` repository.
They are duplicate copies of one historical artifact, not separate projects.

The later 4.7 MB notebook contains more assignment prose and additional code,
but also duplicates much of the shorter notebook. It is treated as the richest
historical context, not as a file that should be copied directly into the
rebuilt portfolio.

## Provenance

The historical notebook explicitly states that code was inspired and partially
copied/derived from:

- Dario Radečić, *LIME vs. SHAP: Which is Better for Explaining Machine Learning Models?*
- Conor O'Sullivan, *Squeezing More out of LIME with Python*

It also cites the SHAP, LIME, and scikit-learn documentation.

The reconstruction retains those citations as provenance but implements the
public source independently.

## High-impact findings

### 1. Dataset narrative and executed analysis disagree

The later notebook contains assignment prose describing the Breast Cancer
Wisconsin dataset, but the executable cells load scikit-learn's Wine dataset.
The public project follows the actual executed Wine analysis and documents the
discrepancy.

### 2. MLP inputs were not standardized

The Wine features have very different numeric scales (for example, magnesium
and proline versus hue and phenols). The historical MLP was fit directly on
those raw features. One later MLP run reached 97.2% test accuracy but emitted a
convergence warning after 500 iterations.

The reconstruction places `StandardScaler` inside a scikit-learn pipeline before
the MLP.

### 3. Split was not stratified

The historical split used `train_test_split(..., random_state=42)` without
`stratify=y`. The resulting 36-row test split had class supports:

- `class_0`: 14
- `class_1`: 14
- `class_2`: 8

The reconstruction uses stratification so evaluation support is more stable.

### 4. Stored classification-report class names are wrong

The shorter notebook used:

```python
target_names = set(raw_data["target_names"])
classification_report(y_test, y_pred, target_names=target_names)
```

A set has no semantic class ordering contract. The stored report displays rows
as `class_2`, `class_1`, `class_0` with supports `14, 14, 8`. But the actual
support of that split is `class_0=14`, `class_1=14`, `class_2=8`.

Therefore at least the first and last row names are attached to the wrong class.
The aggregate accuracy can still be numerically correct while the per-class
interpretation is mislabeled.

The reconstruction always obtains ordered labels from `model.classes_`.

### 5. Mixed DataFrame/NumPy model input produced warnings

The historical MLP was fit with `X_train.values` but later predicted with a
pandas DataFrame, causing scikit-learn feature-name warnings. The reconstruction
keeps named DataFrames through the pipeline.

### 6. SHAP background was unnecessarily large

Historical `KernelExplainer` calls supplied all 142 training rows as background,
and SHAP warned that this could be slow. The reconstruction samples a bounded,
deterministic background before using a model-agnostic permutation explainer.

### 7. LIME global aggregation had two correctness defects

One helper was declared as:

```python
def return_weights(ep):
    exp_list = exp.as_map()[1]
```

It ignores `ep`, reads a global `exp`, and hard-codes class index `1`.

Another aggregation converts `explanation.as_list()` strings back to feature
names by splitting only on `" <= "`. LIME discretized terms can be ranges or
other condition strings, so this method can silently fail to map weights to the
original features.

The reconstruction reads LIME's `as_map()` feature **indices** directly and
requires an explicit class index.

### 8. SHAP/LIME global comparison was not consistently class-matched

Some historical SHAP calls produced multi-class attributions while LIME
aggregation defaulted or hard-coded a single class. A comparison is meaningful
only when both methods refer to the same model output/class and use a clearly
defined aggregation.

The reconstructed helper requires an explicit class for both methods and uses
mean absolute attribution for global ranking.

### 9. Warning suppression hid useful diagnostics

The historical notebook later called `warnings.filterwarnings("ignore")`. That
can conceal convergence, feature-name, and compatibility problems. The public
reconstruction does not globally suppress warnings.

## Current dependency context

As of August 2026:

- scikit-learn 1.9.0 is the current PyPI release (June 2, 2026);
- SHAP 0.52.0 is the current PyPI release (May 28, 2026);
- the original `lime` PyPI package remains at 0.2.0.1 (June 26, 2020).

Because LIME is old but still useful for reproducing the historical comparison,
it is an optional dependency.

## Responsible interpretation

SHAP and LIME answer questions about how a trained model's output changes around
an input or relative to a background/perturbation distribution. They do not
establish:

- causal feature effects,
- data quality,
- absence of leakage,
- model calibration,
- generalization,
- fairness,
- or domain validity.

Agreement between SHAP and LIME is interesting evidence about attribution
stability, not proof that the model itself is correct.
