"""Step 1 - Data acquisition and preprocessing.

Loads the Kaggle Crop Recommendation CSV, checks its quality, encodes the
labels, splits the data (stratified) and scales the features.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
BASE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_data(csv_path):
    """Read the CSV and standardise column names."""
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download 'Crop_recommendation.csv' from Kaggle (Atharva Ingle's "
            "'Crop Recommendation Dataset') and put it in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    # Some uploads use different capitalisation or call the target 'crop'.
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"crop": "label"})
    df = df.rename(columns={f.lower(): f for f in BASE_FEATURES})
    missing = [c for c in BASE_FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns {missing}. Found: {list(df.columns)}"
        )
    return df[BASE_FEATURES + ["label"]]


def quality_report(df):
    """Return a dict summarising data quality (used for the notebook/README)."""
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "n_classes": int(df["label"].nunique()),
        "rows_per_class_min": int(df["label"].value_counts().min()),
        "rows_per_class_max": int(df["label"].value_counts().max()),
    }
    outliers = {}
    for col in BASE_FEATURES:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        outliers[col] = int(mask.sum())
    report["iqr_outliers_per_feature"] = outliers
    return report


def clean_data(df):
    """Drop duplicates and rows with missing values."""
    before = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    return df, before - len(df)


def split_indices(labels, test_size, val_size, seed):
    """Stratified train / validation / test split returned as index arrays."""
    idx = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=rel_val,
        stratify=labels[train_val_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def encode_labels(df):
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    return y, encoder


def scale_features(X_train, X_val, X_test):
    """Fit the scaler on TRAINING data only (prevents data leakage)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    return scaler, X_train_s, scaler.transform(X_val), scaler.transform(X_test)
