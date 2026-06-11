"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, scaling, and class-imbalance correction.

Supports:
  - Kaggle Credit Card Fraud dataset (CSV)
  - Synthetic fallback dataset (when no file is provided)
  - SMOTE oversampling & random undersampling
  - StandardScaler normalisation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Dict, Any


# ── Synthetic dataset generator ──────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 10_000, fraud_ratio: float = 0.02,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic credit-card transaction dataset
    when the user has not uploaded a CSV.

    Features mirror the Kaggle creditcard.csv schema:
      V1–V28  : PCA-transformed features (anonymous)
      Amount  : transaction amount (USD)
      Time    : seconds elapsed since first transaction
      Class   : 0 = legitimate, 1 = fraud
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame(
        rng.normal(0, 1, (n_legit, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    legit["Amount"] = rng.exponential(scale=80, size=n_legit).round(2)
    legit["Time"]   = np.sort(rng.uniform(0, 172_800, n_legit))
    legit["Class"]  = 0

    # Fraudulent transactions (shifted distributions for realism)
    fraud = pd.DataFrame(
        rng.normal(0.8, 1.5, (n_fraud, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    fraud["Amount"] = rng.exponential(scale=300, size=n_fraud).round(2)
    fraud["Time"]   = rng.uniform(0, 172_800, n_fraud)
    fraud["Class"]  = 1

    df = pd.concat([legit, fraud], ignore_index=True).sample(
        frac=1, random_state=random_state
    )
    return df.reset_index(drop=True)


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def load_and_preprocess(
    filepath: str | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    filepath : str | None
        Path to a CSV file (Kaggle creditcard.csv format).
        If None a synthetic dataset is generated.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int

    Returns
    -------
    dict with keys:
        df_raw        – original DataFrame before scaling
        X_train, X_test, y_train, y_test  – train/test arrays
        X_train_smote, y_train_smote       – SMOTE-balanced training data
        X_train_under, y_train_under       – undersampled training data
        scaler        – fitted StandardScaler
        feature_names – list of feature column names
        class_counts  – series with raw class counts
    """
    # 1. Load
    if filepath:
        df = pd.read_csv(filepath)
    else:
        df = generate_synthetic_dataset()

    df_raw = df.copy()

    # 2. Drop fully-null columns / rows
    df.dropna(how="all", inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 3. Drop 'Time' (not predictive in static evaluation)
    if "Time" in df.columns:
        df.drop(columns=["Time"], inplace=True)

    # 4. Feature / label split
    target = "Class"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    y = df[target].values

    # 5. Scale Amount (and any other unscaled columns)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 6. Train / test split (stratified to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 7. SMOTE oversampling
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 8. Random undersampling
    rus = RandomUnderSampler(random_state=random_state)
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

    return {
        "df_raw":          df_raw,
        "X_train":         X_train,
        "X_test":          X_test,
        "y_train":         y_train,
        "y_test":          y_test,
        "X_train_smote":   X_train_smote,
        "y_train_smote":   y_train_smote,
        "X_train_under":   X_train_under,
        "y_train_under":   y_train_under,
        "scaler":          scaler,
        "feature_names":   feature_cols,
        "class_counts":    df_raw["Class"].value_counts(),
    }
