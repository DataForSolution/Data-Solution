# DataForSolution — Curated Data Science & Engineering Portfolio

This repository is a **curated portfolio of selected data science, machine learning, healthcare AI, responsible-AI, and data-engineering work**.

It is not a raw archive of every notebook I have created. Older coursework and experiments are reviewed individually. A project is promoted into [`portfolio/`](portfolio/) only when its provenance, methodology, reproducibility, and claims are strong enough to present publicly.

## Portfolio principles

Each reconstructed project aims to:

- preserve the historical learning context without overstating authorship;
- distinguish original work from course, tutorial, framework, or starter code;
- identify and document methodological defects found during audit;
- use reproducible environments and testable source modules instead of notebook state alone;
- keep datasets and external artifacts traceable to documented sources/licenses;
- avoid publishing credentials, private data, PHI, or unsupported claims;
- report limitations as clearly as results.

## Curated projects

### Healthcare and Medical Imaging

- **[PETQuant Reliability](portfolio/petquant/)** — reconstruction of a PET quantitative-ML workflow with explicit small-sample, data-quality, and validation boundaries.
- **[Chest CT Classification](portfolio/chest-ct-classification/)** — transfer-learning reconstruction focused on evaluation discipline, overfitting, split integrity, and deterministic inference.
- **[Pima Diabetes ML Evaluation](portfolio/pima-diabetes-ml/)** — leakage-safe SVM/MLP retrospective with clinically relevant missing-measurement handling and sensitivity/specificity reporting.

### Responsible and Explainable AI

- **[SHAP + LIME Explainability](portfolio/shap-lime-explainability/)** — class-aligned explainability workflow with corrected label ordering and attribution aggregation.
- **[Fairness Evaluation](portfolio/fairness-evaluation/)** — explicit group-fairness metrics and threshold analysis with fixed reference labels and documented AIF360 provenance.
- **[Adversarial Robustness](portfolio/adversarial-robustness/)** — bounded PGD and defense-evaluation reconstruction with preprocessing, threat-model, and denominator corrections.

### Applied Machine Learning and Generative AI

- **[CIFAR-10 Generated-Image Analysis](portfolio/cifar10-generated-analysis/)** — classifier-response diagnostics for generated images with a clear distinction between model confidence and perceptual realism.
- **[Automobile Insurance Fraud Classification](portfolio/insurance-fraud-classification/)** — temporal, imbalance-aware fraud evaluation showing why high accuracy can coexist with zero minority-class recall.
- **[Restaurant Review Sentiment Evaluation](portfolio/restaurant-sentiment-evaluation/)** — leakage-safe NLP model comparison using fold-local TF-IDF pipelines and a licensed UCI Yelp data source.

### Data Engineering

- **[AWS Glue + EMR PySpark Ingestion Validation](portfolio/spark-glue-ingestion/)** — catalog-resolution and ingestion-quality reconstruction based on a historical embedded-header defect discovered in an EMR/Glue run.

**[View the full portfolio index →](portfolio/README.md)**

## What is intentionally not promoted

Historical notebooks, tutorials, duplicate copies, projects with untraceable datasets, very small experiments, and analyses whose methodology cannot support their original claims are not shown as featured projects. They may remain in archival storage or Git history when useful as learning/provenance evidence, but they are intentionally absent from the current portfolio surface.

## Repository status

The current branch is intentionally minimal: `README.md` plus the canonical `portfolio/` directory. Legacy root notebooks and old website files have been removed from the current tree so the repository listing itself reflects the curated standard.

Each featured project maintains its own README, audit notes, source modules, tests, environment definition, and interpretation boundaries as appropriate.

## Use and interpretation

Several projects use healthcare or other high-stakes datasets. They are educational/research engineering projects and **are not clinical, diagnostic, financial, or security decision systems** unless a project explicitly states otherwise.
