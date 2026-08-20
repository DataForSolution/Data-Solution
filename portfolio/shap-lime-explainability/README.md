# SHAP + LIME Explainability — Portfolio Reconstruction

This project reconstructs a 2024 explainable-AI exercise comparing **SHAP** and **LIME** on a scikit-learn neural-network classifier.

The historical work is preserved as evidence of the original learning process, while this public reconstruction fixes reproducibility, class-label alignment, model preprocessing, and explanation-aggregation issues found during audit.

## What this project demonstrates

- model-agnostic local explanation with SHAP and LIME,
- class-specific attribution comparison,
- reproducible MLP classification with standardized features,
- stable label ordering and stratified evaluation,
- safe aggregation of LIME weights by feature index,
- and explicit separation between an explanation method and evidence that a model is correct.

## Dataset

The reconstruction uses scikit-learn's built-in **Wine recognition dataset**:

- 178 samples,
- 13 numeric features,
- 3 classes.

No external dataset file is committed.

> The later historical notebook contains assignment prose discussing the Breast Cancer Wisconsin dataset, but the executed analysis actually uses the Wine dataset. The public reconstruction follows the executed Wine workflow and documents that discrepancy rather than hiding it.

## Historical audit highlights

The recovered notebooks revealed several issues:

- the MLP was trained on unscaled features;
- one run emitted a convergence warning;
- the train/test split was not stratified;
- model fitting mixed NumPy arrays and pandas DataFrames, producing feature-name warnings;
- `classification_report(..., target_names=set(...))` made report labels nondeterministic and, in the stored output, misidentified class rows;
- SHAP `KernelExplainer` used the full 142-row training set as background and warned about runtime;
- one LIME helper ignored its function argument and read a global variable instead;
- global LIME aggregation hard-coded class index 1;
- another LIME aggregation parsed discretized condition strings, which can silently fail to map weights back to original features;
- SHAP and LIME global summaries were therefore not always comparing the same class or the same attribution definition;
- large sections of the later notebook duplicate earlier code.

See [`docs/AUDIT.md`](docs/AUDIT.md) for details.

## Reconstructed approach

1. Load the Wine dataset with stable class names.
2. Create a **stratified** train/test split.
3. Fit an `MLPClassifier` inside a `StandardScaler` pipeline.
4. Evaluate with accuracy, balanced accuracy, macro-F1, and a correctly ordered classification report.
5. Explain a chosen **explicit class** with both SHAP and LIME.
6. Aggregate global attributions as mean absolute contribution for that same class.
7. Compare top-feature overlap without claiming that agreement proves model validity.

## Repository structure

```text
src/xai_portfolio/
  data.py          # deterministic Wine loading/splitting
  modeling.py      # scaled MLP and evaluation
  attribution.py   # safe SHAP/LIME aggregation helpers

tests/
  test_data.py
  test_modeling.py
  test_attribution.py

docs/
  AUDIT.md

notebooks/
  shap_lime_comparison.ipynb
```

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest
```

To run the SHAP/LIME notebook:

```bash
python -m pip install -e ".[xai]"
jupyter notebook
```

The environment targets current maintained releases as of August 2026: scikit-learn 1.9.x and SHAP 0.52.x. The original `lime` package remains at 0.2.0.1, so it is isolated as an optional dependency rather than treated as a modern core dependency.

## Interpretation boundary

SHAP and LIME explain **model behavior**, not biological truth, causality, or model correctness. A persuasive explanation can still describe a wrong or poorly generalized model. This project is therefore an explainability engineering demonstration, not evidence that the model should be trusted in a high-stakes setting.

## Provenance

The historical notebook explicitly credited tutorial material from Dario Radečić and Conor O'Sullivan. Those citations are preserved in the audit. The reconstructed source code in this portfolio directory was written independently around the underlying SHAP/LIME APIs rather than copied from those tutorial fragments.
