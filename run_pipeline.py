"""
run_pipeline.py
-----------------
End-to-end, reproducible pipeline for the diabetes risk prediction project:

    load -> clean -> EDA figures -> feature engineering -> train/test split
    -> train models (Logistic Regression, Random Forest, XGBoost)
    -> evaluate (accuracy, precision, recall, F1, ROC-AUC)
    -> explain (SHAP, feature importance)
    -> write all figures to /figures and a metrics summary to /reports

Run with:  python3 src/run_pipeline.py
A fixed RANDOM_STATE is used throughout for reproducibility.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for figure generation
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from preprocessing import (
    load_raw_data,
    replace_implausible_zeros_with_nan,
    impute_missing_median,
    missingness_report,
)
from feature_engineering import engineer_features

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
FIG_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------------
section("1. Loading and cleaning data")
df_raw = load_raw_data(str(DATA_PATH))
print(f"Raw shape: {df_raw.shape}")

zero_report = missingness_report(df_raw)
print("Implausible-zero (i.e., missing) counts per clinical field:")
print(zero_report)

df_clean = replace_implausible_zeros_with_nan(df_raw)

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (figures saved before imputation biases them)
# ---------------------------------------------------------------------------
section("2. Exploratory Data Analysis")

# 2a. Class balance
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Outcome", data=df_raw, hue="Outcome", palette="Set2", legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
plt.title("Class Distribution of Outcome")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_balance.png", dpi=150)
plt.close()

class_counts = df_raw["Outcome"].value_counts()
class_ratio = class_counts[0] / class_counts[1]
print(f"Class counts -> No diabetes: {class_counts[0]}, Diabetes: {class_counts[1]} "
      f"(ratio ~{class_ratio:.2f}:1)")

# 2b. Correlation heatmap
plt.figure(figsize=(9, 7))
corr = df_clean.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature Correlation Matrix (after handling implausible zeros)")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# 2c. Distributions of key clinical variables by outcome
key_vars = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "DiabetesPedigreeFunction"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, var in zip(axes.flatten(), key_vars):
    sns.kdeplot(data=df_clean, x=var, hue="Outcome", fill=True, common_norm=False,
                alpha=0.4, ax=ax, palette="Set2")
    ax.set_title(f"{var} Distribution by Outcome")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_distributions.png", dpi=150)
plt.close()

# 2d. Boxplots to visualize outliers per feature
plt.figure(figsize=(12, 6))
df_melt = df_clean[key_vars].melt(var_name="Feature", value_name="Value")
sns.boxplot(data=df_melt, x="Feature", y="Value", hue="Feature", palette="Set3", legend=False)
plt.xticks(rotation=30)
plt.title("Outlier Inspection Across Key Clinical Features")
plt.tight_layout()
plt.savefig(FIG_DIR / "outlier_boxplots.png", dpi=150)
plt.close()

print(f"EDA figures written to: {FIG_DIR}")

# ---------------------------------------------------------------------------
# 3. Train/test split BEFORE imputation and feature engineering
#    (prevents any leakage of test-set statistics into training artifacts)
# ---------------------------------------------------------------------------
section("3. Train/test split")

X_cols = [c for c in df_clean.columns if c != "Outcome"]
X = df_clean[X_cols]
y = df_clean["Outcome"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train_raw.shape[0]}, Test size: {X_test_raw.shape[0]}")
print(f"Train class balance:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"Test class balance:\n{y_test.value_counts(normalize=True).round(3)}")

# ---------------------------------------------------------------------------
# 4. Median imputation, fit on TRAIN ONLY, applied to both splits
# ---------------------------------------------------------------------------
section("4. Median imputation (fit on training split only)")

X_train_imp = impute_missing_median(X_train_raw, reference=X_train_raw)
X_test_imp = impute_missing_median(X_test_raw, reference=X_train_raw)

# ---------------------------------------------------------------------------
# 5. Feature engineering, applied identically to both splits
# ---------------------------------------------------------------------------
section("5. Feature engineering")

train_full = X_train_imp.copy()
train_full["Outcome"] = y_train.values
test_full = X_test_imp.copy()
test_full["Outcome"] = y_test.values

train_fe = engineer_features(train_full.drop(columns=["Outcome"]))
test_fe = engineer_features(test_full.drop(columns=["Outcome"]))

# Align columns in case a rare category is present in only one split
train_fe, test_fe = train_fe.align(test_fe, join="left", axis=1, fill_value=0)

print(f"Engineered feature count: {train_fe.shape[1]} (from {X_train_raw.shape[1]} raw features)")
print("Engineered feature names:")
print(list(train_fe.columns))

# ---------------------------------------------------------------------------
# 6. Scale features (fit scaler on train only) for Logistic Regression
# ---------------------------------------------------------------------------
section("6. Feature scaling")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_fe)
X_test_scaled = scaler.transform(test_fe)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_fe.columns, index=train_fe.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=test_fe.columns, index=test_fe.index)

# ---------------------------------------------------------------------------
# 7. Model training
# ---------------------------------------------------------------------------
section("7. Model training")

models = {}

# Logistic Regression: chosen first as a transparent, coefficient-based
# baseline. class_weight='balanced' compensates for the ~1.9:1 class
# imbalance without synthetic oversampling.
log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
models["Logistic Regression"] = (log_reg, X_test_scaled)

# Random Forest: a nonlinear ensemble that still supports built-in
# feature-importance and SHAP TreeExplainer analysis.
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=5,
    class_weight="balanced", random_state=RANDOM_STATE
)
rf.fit(train_fe, y_train)
models["Random Forest"] = (rf, test_fe)

# XGBoost: gradient-boosted trees, generally the strongest tabular baseline;
# scale_pos_weight compensates for class imbalance analogous to class_weight.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", random_state=RANDOM_STATE
)
xgb.fit(train_fe, y_train)
models["XGBoost"] = (xgb, test_fe)

print("Trained models:", list(models.keys()))

# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
section("8. Model evaluation")

results = {}
roc_data = {}

for name, (model, X_te) in models.items():
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    results[name] = metrics

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, metrics["roc_auc"])

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # Confusion matrix figure per model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(FIG_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()

# Combined ROC curve
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curves_comparison.png", dpi=150)
plt.close()

# Metrics comparison bar chart
metrics_df = pd.DataFrame(results).T
plt.figure(figsize=(9, 5))
metrics_df.plot(kind="bar", ax=plt.gca(), colormap="viridis")
plt.title("Model Comparison Across Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(loc="lower right", ncol=3, fontsize=8)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "metrics_comparison.png", dpi=150)
plt.close()

print("\nSummary metrics table:")
print(metrics_df.round(4))

metrics_df.round(4).to_csv(REPORT_DIR / "metrics_summary.csv")

# ---------------------------------------------------------------------------
# 9. Explainability: feature importance + SHAP (Random Forest as primary
#    explainability model since tree ensembles support fast, exact SHAP)
# ---------------------------------------------------------------------------
section("9. Explainability (feature importance + SHAP)")

# 9a. Random Forest built-in feature importance
importances = pd.Series(rf.feature_importances_, index=train_fe.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
importances.head(12).sort_values().plot(kind="barh", color="teal")
plt.title("Random Forest — Top 12 Feature Importances")
plt.xlabel("Relative Importance (Gini-based)")
plt.tight_layout()
plt.savefig(FIG_DIR / "rf_feature_importance.png", dpi=150)
plt.close()
print("\nTop 10 Random Forest features:")
print(importances.head(10))

# 9b. Logistic Regression coefficients (directly interpretable log-odds)
coef_series = pd.Series(log_reg.coef_[0], index=train_fe.columns).sort_values()
plt.figure(figsize=(8, 6))
coef_series.plot(kind="barh", color=["firebrick" if c < 0 else "steelblue" for c in coef_series])
plt.title("Logistic Regression Coefficients (standardized features)")
plt.xlabel("Coefficient (log-odds contribution)")
plt.tight_layout()
plt.savefig(FIG_DIR / "logreg_coefficients.png", dpi=150)
plt.close()

# 9c. SHAP values for the Random Forest model
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(test_fe)

# shap_values can be a list (per-class) depending on sklearn/shap version;
# for binary classification we take the "positive class" (diabetes) values.
if isinstance(shap_values, list):
    shap_vals_pos = shap_values[1]
elif shap_values.ndim == 3:
    shap_vals_pos = shap_values[:, :, 1]
else:
    shap_vals_pos = shap_values

plt.figure()
shap.summary_plot(shap_vals_pos, test_fe, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_vals_pos, test_fe, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_feature_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Local explanation for a single representative patient (first test-set case)
plt.figure()
sample_idx = 0
shap.decision_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray))
    and np.ndim(explainer.expected_value) > 0 else explainer.expected_value,
    shap_vals_pos[sample_idx],
    test_fe.iloc[sample_idx],
    show=False,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_local_decision_plot_patient0.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"SHAP and feature-importance figures written to: {FIG_DIR}")

# ---------------------------------------------------------------------------
# 10. Persist final metrics + config for reproducibility
# ---------------------------------------------------------------------------
section("10. Saving run configuration and results")

run_summary = {
    "random_state": RANDOM_STATE,
    "train_size": int(X_train_raw.shape[0]),
    "test_size": int(X_test_raw.shape[0]),
    "n_features_after_engineering": int(train_fe.shape[1]),
    "class_balance_full_dataset": class_counts.to_dict(),
    "metrics": results,
}
with open(REPORT_DIR / "run_summary.json", "w") as f:
    json.dump(run_summary, f, indent=2, default=str)

print("Pipeline complete. Metrics saved to reports/metrics_summary.csv and reports/run_summary.json")
