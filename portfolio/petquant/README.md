# PETQuant Reliability — Portfolio Reconstruction

PETQuant is a safety-oriented medical-imaging engineering project exploring whether acquisition context can be used to estimate the **reliability of PET SUV comparability**. The project separates deterministic rules from advisory machine learning and gives special attention to avoiding false reassurance.

> **Important:** This repository is an engineering portfolio reconstruction. It is **not a diagnostic system**, is **not clinically validated**, and must not be used for patient-care decisions.

## Why this reconstruction exists

The original work was developed in Google Colab and used private/cloud-connected data sources. During reconstruction, the historical notebooks were audited for reproducibility, data handling, methodology, and claims. The public version intentionally excludes private operational data and cloud credentials/endpoints and instead preserves the engineering lessons in reproducible code and tests.

## Key engineering lessons

- Measure false reassurance explicitly, not just overall accuracy.
- Fail closed when a safety denominator is absent instead of reporting a misleading 0% error rate.
- Treat confidence thresholds as hypotheses until validated on independent data.
- Normalize nested export formats before numeric conversion.
- Do not silently median-impute a feature when the entire column is missing.
- Keep deterministic clinical rules authoritative unless an advisory model has sufficient independent validation.

## Historical validation context

The recovered training notebook used only four records, producing a three-row training set and one-row validation set. That is insufficient for a model-performance claim.

A later historical rule-vs-ML evaluation showed a 29.3% false-reliable rate (41/140 rule-unreliable cases predicted reliable) on a mixed event dataset. A transparent score gate reduced false-reliable predictions in that same dataset, but the threshold was chosen and evaluated on the same evidence and therefore is **not an independently validated clinical threshold**.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the full reconstruction assessment.

## Repository structure

```text
src/petquant_portfolio/
  features.py       # robust feature normalization
  safety.py         # false-reliable metric and conservative score gate
tests/
  test_features.py
  test_safety.py
docs/
  VALIDATION.md
notebooks/
  petquant_safety_demo.ipynb
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest
```

## What is intentionally not published

- patient data or PHI,
- private cloud data,
- credentials or secrets,
- model artifacts trained on non-public data,
- operational service configuration,
- or claims that exceed the evidence.

## Current portfolio status

The reconstruction is suitable for demonstrating healthcare-aware software engineering, data-quality review, model-safety thinking, and reproducible Python practices. A clinically deployable model would require a substantially larger independently labeled dataset and formal external validation.
