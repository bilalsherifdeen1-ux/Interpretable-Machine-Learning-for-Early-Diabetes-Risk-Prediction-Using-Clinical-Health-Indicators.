# Interpretable Machine Learning for Early Diabetes Risk Prediction Using Clinical Health Indicators

A reproducible, interpretability-first machine learning project for early diabetes risk
screening, built as a technical work sample. The project trains and compares three models
(Logistic Regression, Random Forest, XGBoost) on the Pima Indians Diabetes Dataset, evaluates
them with metrics suited to a clinical screening context, explains their predictions with
SHAP, and treats bias, uncertainty, and responsible deployment as first-class concerns rather
than an afterthought.

**Full technical write-up:** [`reports/paper.md`](reports/paper.md)

---

## Highlights

- **Leakage-safe pipeline** — train/test split performed *before* imputation and scaling;
  all preprocessing statistics fit on the training split only.
- **Clinically grounded feature engineering** — BMI/glucose/age risk buckets and interaction
  terms chosen for physiological plausibility, not just correlation with the target.
- **Three models across the interpretability spectrum** — Logistic Regression (fully
  transparent), Random Forest, and XGBoost, evaluated on accuracy, precision, recall, F1, and
  ROC-AUC.
- **Explainability at both the population and individual level** — SHAP summary plots,
  SHAP bar importance, a per-patient SHAP decision plot, and Logistic Regression coefficients.
- **An explicit AI Safety Discussion** covering bias/fairness, distribution shift, model
  uncertainty, human oversight, and responsible clinical deployment — see
  [`reports/paper.md`](reports/paper.md), Section 10.
- **Fully reproducible** from a single fixed random seed (`RANDOM_STATE = 42`).

## Results Summary

Evaluated on a held-out, stratified 20% test split (n = 154):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.792** | 0.672 | **0.796** | **0.729** | **0.846** |
| Random Forest | 0.747 | 0.612 | 0.759 | 0.678 | 0.829 |
| XGBoost | 0.747 | 0.642 | 0.630 | 0.636 | 0.833 |

Logistic Regression — the most transparent of the three models — achieved the best recall and
ROC-AUC, which we argue matters more than raw accuracy in a screening context where missing a
true positive is more costly than a false alarm. See [`reports/paper.md`](reports/paper.md),
Section 8, for the full discussion.

> **Disclaimer:** This is a research prototype and technical work sample built on a small,
> narrow, publicly available dataset (see [Limitations](#limitations)). It is **not** validated
> for, and must not be used for, real clinical decision-making.

## Repository Structure

```
diabetes-risk-project/
├── README.md                    # This file
├── data/
│   └── diabetes.csv             # Pima Indians Diabetes Dataset (768 patients, 8 features + label)
├── notebooks/
│   └── diabetes_risk_analysis.ipynb   # End-to-end, narrated, executable notebook
├── src/
│   ├── preprocessing.py         # Missing-value handling, leakage-safe imputation
│   ├── feature_engineering.py   # Clinically motivated derived features
│   └── run_pipeline.py          # Full pipeline: load -> clean -> EDA -> train -> evaluate -> explain
├── figures/                     # All generated plots (EDA, confusion matrices, ROC, SHAP, etc.)
├── reports/
│   ├── paper.md                 # Full technical paper (all required sections)
│   ├── metrics_summary.csv      # Final metrics table
│   └── run_summary.json         # Run configuration + results, for reproducibility
└── requirements.txt             # Pinned-style dependency list
```

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <this-repo-url>
cd diabetes-risk-project
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
cd src
python3 run_pipeline.py
```

This will regenerate every figure in `figures/` and every result in `reports/` from scratch,
using the fixed random seed `RANDOM_STATE = 42` for reproducibility.

### 3. Or explore interactively

```bash
jupyter notebook notebooks/diabetes_risk_analysis.ipynb
```

The notebook mirrors the pipeline step by step, with narrative markdown explaining the
reasoning behind each choice (why zeros are treated as missing, why the split happens before
imputation, why recall is emphasized, etc.).

## Dataset

The **Pima Indians Diabetes Dataset** (originally from the National Institute of Diabetes and
Digestive and Kidney Diseases, distributed via the UCI Machine Learning Repository and
Kaggle) contains 768 records for female patients of Pima Indian heritage, age 21+, with 8
clinical features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
DiabetesPedigreeFunction, Age) and a binary `Outcome` label. See
[`reports/paper.md`](reports/paper.md), Section 3, for full details, including a known
data-quality issue (implausible zero values used as missing-data placeholders) and how this
project handles it.

## Explainability

All explainability artifacts are generated in `figures/`:

- `rf_feature_importance.png` — Random Forest Gini-based feature importance
- `logreg_coefficients.png` — Logistic Regression standardized coefficients
- `shap_summary_beeswarm.png` — SHAP value distribution per feature, across all test patients
- `shap_feature_importance_bar.png` — mean absolute SHAP value per feature
- `shap_local_decision_plot_patient0.png` — a worked local explanation for one individual
  patient

## AI Safety & Responsible ML

This project treats trustworthiness as part of the technical deliverable, not a separate
concern. See [`reports/paper.md`](reports/paper.md), Section 10, for a full discussion of:

- **Bias and fairness** — the dataset's narrow demographic scope and what that does and does
  not license.
- **Distribution shift** — how clinical measurement practices can silently change model
  performance over time or across sites.
- **Model uncertainty** — why raw predicted probabilities are not automatically calibrated,
  and what a calibration-aware version of this project would add.
- **Human oversight** — this model is designed as a decision-support signal, not a
  diagnostic authority.
- **Responsible deployment** — intended-use documentation, monitoring plans, consent, and
  regulatory context (e.g., FDA Software as a Medical Device considerations) that a real
  deployment would require and this project does not attempt to satisfy.

## Limitations

- Single, small (768-patient), narrow (one sex, one ethnicity, age 21+) dataset.
- Substantial missingness in some fields (e.g., ~49% of Insulin values were imputed).
- No calibration analysis or external validation set in this iteration.
- Modest ceiling on achievable performance (~0.85 ROC-AUC), consistent with prior published
  work on this exact dataset.

See [`reports/paper.md`](reports/paper.md), Sections 11–12, for the full limitations and
future work discussion.

## Reproducibility

- Fixed random seed (`RANDOM_STATE = 42`) used throughout for the train/test split and all
  stochastic model components.
- All preprocessing statistics (imputation medians, feature scaler) are fit on the training
  split only and then applied to the test split — no test-set information leaks into training.
- `reports/run_summary.json` records the exact configuration and results of the run that
  produced the numbers in this README and in `reports/paper.md`.

## Requirements

See [`requirements.txt`](requirements.txt). Core dependencies: `pandas`, `numpy`,
`scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `jupyter`.

## License

This project is released for educational and portfolio purposes. The Pima Indians Diabetes
Dataset is publicly available via the UCI Machine Learning Repository / Kaggle; see the
dataset's original terms for reuse.

## References

Full APA-formatted references are provided in [`reports/paper.md`](reports/paper.md).
