# Fairness Evaluation — Portfolio Reconstruction

This project reconstructs the **fairness-evaluation lessons** from a 2024 DATA 450 assignment built on IBM/Trusted-AI's AIF360 `demo_optim_data_preproc.ipynb` example.

The historical notebooks are not presented as original AIF360 implementations. Their Colab metadata points directly to the official AIF360 demo. The meaningful portfolio contribution is the classifier/fairness comparison, the identification of tradeoffs, and the reconstruction of a safer evaluation contract.

## Historical experiment actually executed

Despite several filenames, the later saved notebooks execute the **Adult dataset with sex as the protected attribute**.

Two historical classifier runs are useful as observations:

| Classifier | Balanced accuracy before | After | Disparate impact before | After |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.7440 | 0.7221 | 0.2765 | 0.5369 |
| Logistic regression | 0.7437 | 0.7018 | 0.2794 | 0.7490 |

These saved outputs show the intended utility/fairness tradeoff: group-fairness metrics improved while balanced accuracy declined. They are historical notebook results, not newly reproduced benchmarks.

## Audit corrections

- `...RF_with_plots.ipynb` does **not** run a Random Forest. It is the MLP notebook with one extra comment mentioning Random Forest.
- `..._german.ipynb` still sets `dataset_used = "adult"`; it runs Logistic Regression on Adult rather than German Credit.
- The markdown summary table listing Adult/German/COMPAS is inherited from the official AIF360 demo and is not evidence that all three datasets were rerun in these modified notebooks.
- The notebook installs an old Matplotlib version inside the analysis, which is unnecessary for the public reconstruction.
- AIF360's Optimized Preprocessing can transform both features **and labels** (`transform_Y=True`). Therefore “before” and “after” accuracy can refer to different reference labels unless the estimand is stated explicitly.

See [`docs/AUDIT.md`](docs/AUDIT.md) for details.

## Reconstructed evaluation contract

The reusable public code intentionally separates **fairness measurement** from any particular mitigation algorithm:

1. Keep one explicit fixed `y_true` reference vector for a comparison.
2. Evaluate predictions/scores on aligned rows and one declared protected attribute.
3. Compute:
   - balanced accuracy,
   - statistical parity difference,
   - disparate impact,
   - equal opportunity difference,
   - average odds difference.
4. Fail closed (`None`) when a metric denominator is undefined instead of manufacturing a numeric value.
5. Sweep thresholds on validation data and select the best balanced-accuracy threshold deterministically.
6. Report utility and fairness together; no single metric defines “fair.”

## Repository structure

```text
src/fairness_portfolio/
  metrics.py       # group fairness + balanced accuracy
  thresholds.py    # threshold/fairness curves on fixed labels

tests/
  test_metrics.py
  test_thresholds.py

docs/
  AUDIT.md

notebooks/
  fairness_threshold_demo.ipynb
```

## Install and test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest
```

AIF360 is optional for reproducing framework-specific mitigation experiments:

```bash
python -m pip install -e ".[aif360]"
```

Current AIF360 documentation continues to describe `OptimPreproc` as a probabilistic preprocessing method that can edit features and labels under fairness/distortion/data-fidelity objectives.

## Interpretation boundary

Fairness is definition- and context-dependent. Statistical parity, disparate impact, equality of opportunity, average odds, calibration, and individual fairness can conflict. A portfolio metric improvement is not a statement that a deployed system is ethically or legally fair.
