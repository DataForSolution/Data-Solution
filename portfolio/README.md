# Curated Portfolio

These projects were selected from a larger archive of notebooks, coursework, experiments, and research work. Each featured project has been reviewed for **provenance, methodology, reproducibility, data boundaries, and claim quality** before being promoted here.

## Project index

| Project | Domain | Core technologies | What the reconstruction demonstrates |
| --- | --- | --- | --- |
| [PETQuant Reliability](petquant/) | Healthcare AI / PET | Python, NumPy, pytest | Small-data validation, data-quality contracts, conservative reliability gates |
| [Chest CT Classification](chest-ct-classification/) | Medical imaging | TensorFlow, scikit-learn, pytest | Transfer-learning discipline, overfitting analysis, deterministic validation/test handling |
| [Pima Diabetes ML Evaluation](pima-diabetes-ml/) | Healthcare ML | scikit-learn, pandas, OpenML | Leakage-safe preprocessing, missing-measurement handling, sensitivity/specificity |
| [SHAP + LIME Explainability](shap-lime-explainability/) | Explainable AI | SHAP, LIME, scikit-learn | Stable class alignment, attribution aggregation, interpretation limits |
| [Fairness Evaluation](fairness-evaluation/) | Responsible AI | Python, AIF360, pytest | Group-fairness metrics, fixed-label threshold evaluation, utility tradeoffs |
| [Adversarial Robustness](adversarial-robustness/) | ML security | PyTorch, ART, pytest | Bounded PGD, threat-model discipline, clean-vs-robust utility |
| [Automobile Insurance Fraud Classification](insurance-fraud-classification/) | Imbalanced ML | scikit-learn, pandas, PR-AUC | Temporal holdout, threshold selection, minority-class tradeoffs |
| [Restaurant Review Sentiment Evaluation](restaurant-sentiment-evaluation/) | NLP | TF-IDF, scikit-learn, VADER | Fold-local text modeling and stratified out-of-fold comparison |
| [CIFAR-10 Generated-Image Analysis](cifar10-generated-analysis/) | Generative AI evaluation | NumPy, classifier diagnostics, pytest | Probability validation, entropy/margin analysis, realism boundary |
| [AWS Glue + EMR PySpark Ingestion Validation](spark-glue-ingestion/) | Data engineering | PySpark, AWS Glue, EMR | Catalog resolution, embedded-header detection, fail-closed ingestion checks |

## Selection standard

A historical project is **not** promoted merely because it runs or once received a good grade. The reconstruction process asks:

1. **Provenance** — What was original work, what came from a tutorial/course/example, and can the distinction be documented?
2. **Data legitimacy** — Is the dataset traceable, appropriately licensed, de-identified where necessary, and suitable for the claim being made?
3. **Methodological validity** — Are train/validation/test boundaries, preprocessing, metrics, denominators, and evaluation procedures defensible?
4. **Reproducibility** — Can the important logic be expressed as source code with explicit dependencies and tests rather than relying on notebook state?
5. **Portfolio value** — Does the project demonstrate a distinct engineering or analytical skill that is not already represented better elsewhere?

Projects that fail this standard are archived, deferred, or used only as methodology-audit evidence.

## Reconstructed project pattern

Most projects follow a common structure:

```text
project/
  README.md
  pyproject.toml
  src/
  tests/
  docs/AUDIT.md
  notebooks/
  data/README.md   # when external data provenance is relevant
```

The `AUDIT.md` files are intentional. They document historical bugs, provenance boundaries, and interpretation limitations rather than rewriting history to make an older project look cleaner than it actually was.

## Current portfolio coverage

The selected projects collectively demonstrate:

- healthcare and medical-imaging technology,
- classical and neural-network model evaluation,
- explainable and responsible AI,
- adversarial robustness,
- generative-model diagnostics,
- imbalanced classification,
- natural-language processing,
- AWS/PySpark data engineering,
- reproducible Python packaging and regression testing.

The portfolio will remain selective. Additional historical work will be added only when it contributes a distinct, defensible capability.
