"""
preprocessing.py
-----------------
Data loading and cleaning utilities for the diabetes risk prediction project.

The Pima Indians Diabetes Dataset encodes missing clinical measurements as 0
for physiologically implausible fields (e.g., a blood pressure of 0 mmHg is
not survivable). We treat these zeros as missing values (NaN) and impute
them, rather than treating them as true zero readings, since naively
including them would bias the model toward spurious patterns.
"""

import pandas as pd
import numpy as np

# Columns where a recorded value of 0 is physiologically implausible and is
# therefore treated as a missing-data placeholder rather than a true reading.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def replace_implausible_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zero values with NaN so they can be
    handled explicitly by an imputation strategy instead of silently
    distorting feature distributions and downstream model coefficients.
    """
    df = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        df[col] = df[col].replace(0, np.nan)
    return df


def impute_missing_median(df: pd.DataFrame, reference: pd.DataFrame = None) -> pd.DataFrame:
    """Impute missing values with the median of each column.

    A `reference` DataFrame (e.g., the training split) may be supplied so
    that validation/test data is imputed using statistics learned only from
    training data. This avoids test-set information leaking into the
    imputation step, which is a common and easy-to-miss source of
    over-optimistic evaluation results.
    """
    df = df.copy()
    stats_source = reference if reference is not None else df
    for col in ZERO_AS_MISSING_COLUMNS:
        median_value = stats_source[col].median()
        df[col] = df[col].fillna(median_value)
    return df


def missingness_report(df: pd.DataFrame) -> pd.Series:
    """Return the count of implausible-zero (i.e., missing) values per column."""
    zero_counts = (df[ZERO_AS_MISSING_COLUMNS] == 0).sum()
    return zero_counts.sort_values(ascending=False)
