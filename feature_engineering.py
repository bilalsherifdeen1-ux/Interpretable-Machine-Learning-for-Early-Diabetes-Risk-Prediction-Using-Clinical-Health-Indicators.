"""
feature_engineering.py
-----------------------
Derives a small number of clinically motivated features on top of the raw
Pima Indians Diabetes measurements. Every engineered feature is chosen for
clinical plausibility rather than for chasing marginal accuracy gains, in
keeping with the project's emphasis on interpretability.
"""

import pandas as pd
import numpy as np


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket BMI into standard WHO categories, which clinicians already use
    and can sanity-check, rather than relying on the model to rediscover
    well-known clinical thresholds from a continuous value alone.
    """
    df = df.copy()
    bins = [0, 18.5, 25, 30, np.inf]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMI_Category"] = pd.cut(df["BMI"], bins=bins, labels=labels)
    return df


def add_glucose_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket fasting glucose into normal / prediabetic / diabetic ranges
    per standard ADA (American Diabetes Association) screening thresholds.
    """
    df = df.copy()
    bins = [0, 99, 125, np.inf]
    labels = ["Normal", "Prediabetic", "Diabetic_Range"]
    df["Glucose_Category"] = pd.cut(df["Glucose"], bins=bins, labels=labels)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket age into decades-aligned risk bands consistent with clinical
    diabetes-risk screening guidelines.
    """
    df = df.copy()
    bins = [0, 30, 45, 60, np.inf]
    labels = ["<30", "30-44", "45-59", "60+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small number of interaction terms with a known physiological
    rationale (e.g., glucose and BMI jointly relate to insulin resistance),
    rather than an unconstrained polynomial expansion that would be harder
    to explain to a clinical reviewer.
    """
    df = df.copy()
    df["Glucose_BMI_Interaction"] = df["Glucose"] * df["BMI"]
    df["Age_Pregnancies_Interaction"] = df["Age"] * df["Pregnancies"]
    return df


def one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical bucket features created above."""
    cat_cols = [c for c in ["BMI_Category", "Glucose_Category", "Age_Group"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full, clinically-motivated feature engineering pipeline."""
    df = add_bmi_category(df)
    df = add_glucose_category(df)
    df = add_age_group(df)
    df = add_interaction_terms(df)
    df = one_hot_encode_categoricals(df)
    return df
