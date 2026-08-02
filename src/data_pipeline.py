"""
data_pipeline.py
=================
Phase 1 of the Energy Consumption Forecasting system.

Handles the full data-engineering lifecycle for the Fedesoriano "Electric
Power Consumption" (Tetouan City) dataset:
    https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption

Raw columns expected in the CSV:
    DateTime, Temperature, Humidity, Wind Speed,
    general diffuse flows, diffuse flows,
    Zone 1 Power Consumption, Zone 2 Power Consumption, Zone 3 Power Consumption

This module is intentionally dependency-light (pandas/numpy/sklearn only) so
it can be imported both by the offline training script (train.py) and,
indirectly, by the Streamlit app (for the fitted scalers).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Canonical column names used throughout the project (renamed from the messy
# raw Kaggle headers early, so every downstream module works with clean names)
# ----------------------------------------------------------------------------
RAW_TO_CLEAN = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
    "Zone 1 Power Consumption": "power_zone1",
    "Zone 2 Power Consumption": "power_zone2",
    "Zone 3 Power Consumption": "power_zone3",
}

TARGET_COL = "power_zone1"

# Final feature set fed into the ANN (per the spec: temp, humidity, wind,
# hour, day-of-week, is_holiday -- we also keep month/is_weekend available
# for EDA, but the model only trains on this exact list unless you extend it)
FEATURE_COLS = [
    "temperature",
    "humidity",
    "wind_speed",
    "hour",
    "day_of_week",
    "is_holiday",
]


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame


# ----------------------------------------------------------------------------
# 1. Loading + Cleaning
# ----------------------------------------------------------------------------
def load_raw_csv(path: str) -> pd.DataFrame:
    """Load the raw Kaggle/UCI CSV and rename columns to clean snake_case.

    NOTE: the real source file ships with inconsistent internal whitespace
    in a couple of headers (e.g. "Zone 2  Power Consumption" has a double
    space). We normalize whitespace before mapping so both the Kaggle and
    UCI mirrors of this dataset load identically.
    """
    df = pd.read_csv(path)
    df.columns = [" ".join(c.split()) for c in df.columns]  # collapse repeated whitespace
    df = df.rename(columns={k: v for k, v in RAW_TO_CLEAN.items() if k in df.columns})
    missing = [c for c in RAW_TO_CLEAN.values() if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing expected columns: {missing}. "
            "Make sure this is the Fedesoriano Electric Power Consumption dataset."
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning & validation:
      - parse/validate the datetime column, drop unparsable rows
      - sort chronologically (critical -- source file may not be sorted)
      - handle missing values via time-aware interpolation
      - clip statistical outliers (beyond 3 std) in weather + power columns
    """
    df = df.copy()

    # --- Corrupted timestamps -------------------------------------------------
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    n_bad_ts = df["datetime"].isna().sum()
    if n_bad_ts:
        logger.warning("Dropping %d rows with unparsable timestamps.", n_bad_ts)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # --- Duplicate timestamps ---------------------------------------------------
    dupes = df["datetime"].duplicated().sum()
    if dupes:
        logger.warning("Dropping %d duplicate-timestamp rows.", dupes)
        df = df.drop_duplicates(subset="datetime", keep="first")

    numeric_cols = [
        "temperature", "humidity", "wind_speed",
        "general_diffuse_flows", "diffuse_flows",
        "power_zone1", "power_zone2", "power_zone3",
    ]

    # --- Missing values: coerce to numeric, then time-aware interpolation -----
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing:
        logger.warning("Interpolating %d missing numeric values.", int(n_missing))
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # --- Physically impossible readings -> treat as missing, then re-interpolate
    df.loc[(df["humidity"] < 0) | (df["humidity"] > 100), "humidity"] = np.nan
    df.loc[df["wind_speed"] < 0, "wind_speed"] = np.nan
    df.loc[(df["temperature"] < -30) | (df["temperature"] > 55), "temperature"] = np.nan
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # --- Statistical outlier clipping (3-sigma) on weather + power columns ----
    for col in numeric_cols:
        mu, sigma = df[col].mean(), df[col].std()
        lower, upper = mu - 3 * sigma, mu + 3 * sigma
        n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_clipped:
            logger.info("Clipping %d outliers in '%s' to [%.2f, %.2f].", n_clipped, col, lower, upper)
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


# ----------------------------------------------------------------------------
# 2. EDA helpers (used by the "EDA" tab in the Streamlit app / notebooks)
# ----------------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of weather attributes + temporal features against the target."""
    cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    return df[cols].corr()


def target_distribution_summary(df: pd.DataFrame) -> dict:
    """Skewness / spike diagnostics for the target variable (Phase 2 step 1)."""
    from scipy import stats  # local import: only needed for this diagnostic

    y = df[TARGET_COL]
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    spikes = ((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum()
    return {
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std()),
        "skewness": float(stats.skew(y)),
        "kurtosis": float(stats.kurtosis(y)),
        "iqr_outlier_count": int(spikes),
        "iqr_outlier_pct": float(spikes / len(y) * 100),
    }


# ----------------------------------------------------------------------------
# 3. Feature engineering
# ----------------------------------------------------------------------------
def add_holiday_flags(df: pd.DataFrame, holiday_dates: Optional[set] = None) -> pd.DataFrame:
    """
    Adds `is_holiday`. Without an external holiday calendar the dataset has no
    holiday field, so by default we treat Sundays as the regional day-off
    proxy (Morocco/Tetouan work week runs Mon-Sat in many sectors) AND allow
    the caller to pass an explicit set of `date()` objects for real public
    holidays, which take priority.
    """
    df = df.copy()
    holiday_dates = holiday_dates or set()
    is_named_holiday = df["datetime"].dt.date.isin(holiday_dates)
    is_weekend_proxy = df["datetime"].dt.dayofweek == 6  # Sunday
    df["is_holiday"] = (is_named_holiday | is_weekend_proxy).astype(int)
    return df


def engineer_features(df: pd.DataFrame, holiday_dates: Optional[set] = None) -> pd.DataFrame:
    """
    Extracts temporal features from `datetime`:
        hour, day_of_week, month, is_weekend, is_holiday
    and assembles the final modeling table (features + target), fully
    time-ordered.
    """
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # Mon=0 ... Sun=6
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = add_holiday_flags(df, holiday_dates)
    return df


# ----------------------------------------------------------------------------
# 4 & 5. Scaling + chronological train/test split
# ----------------------------------------------------------------------------
def scale_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    feature_scaler: Optional[MinMaxScaler] = None,
    target_scaler: Optional[MinMaxScaler] = None,
) -> tuple[SplitData, MinMaxScaler, MinMaxScaler]:
    """
    Chronological (non-shuffled) 80/20 split, then MinMax scaling of features
    and target independently (each fit ONLY on the training partition to
    avoid leakage from the future into training statistics).
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train_raw = train_df[FEATURE_COLS].values
    X_test_raw = test_df[FEATURE_COLS].values
    y_train_raw = train_df[[TARGET_COL]].values
    y_test_raw = test_df[[TARGET_COL]].values

    feature_scaler = feature_scaler or MinMaxScaler(feature_range=(0, 1))
    target_scaler = target_scaler or MinMaxScaler(feature_range=(0, 1))

    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)
    y_train = target_scaler.fit_transform(y_train_raw)
    y_test = target_scaler.transform(y_test_raw)

    logger.info(
        "Chronological split -> train: %d rows (%.0f%%..%s), test: %d rows (%s..%.0f%%)",
        len(train_df), 0, train_df["datetime"].max(),
        len(test_df), test_df["datetime"].min(), 100,
    )

    return (
        SplitData(X_train, X_test, y_train, y_test, train_df, test_df),
        feature_scaler,
        target_scaler,
    )


def run_full_pipeline(csv_path: str, holiday_dates: Optional[set] = None):
    """Convenience entry point: raw CSV -> cleaned, engineered, scaled, split."""
    raw = load_raw_csv(csv_path)
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned, holiday_dates)
    split, fscaler, tscaler = scale_and_split(engineered)
    return engineered, split, fscaler, tscaler
