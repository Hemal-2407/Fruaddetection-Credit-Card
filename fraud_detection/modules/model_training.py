"""
model_training.py
-----------------
Trains and persists the following models:
  Supervised   : Logistic Regression, Random Forest, XGBoost
  Unsupervised : Isolation Forest (anomaly detection)
  Deep Learning: Autoencoder (PyTorch, anomaly detection)

Includes hyperparameter tuning via RandomizedSearchCV.
All models are saved to the 'models/' directory using joblib.
"""

from __future__ import annotations

import os
import numpy as np
import joblib
from typing import Dict, Any, Tuple

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline        import Pipeline
from xgboost                 import XGBClassifier

# ── PyTorch Autoencoder (optional – graceful fallback) ───────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Supervised models
# ════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train Logistic Regression with optional randomised hyperparameter search.
    class_weight='balanced' boosts recall on the minority (fraud) class.
    """
    if tune:
        param_dist = {
            "C":       [0.001, 0.01, 0.1, 1, 10, 100],
            "solver":  ["lbfgs", "liblinear"],
            "max_iter":[300, 600, 1000],
        }
        base = LogisticRegression(class_weight="balanced", random_state=random_state)
        cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=10, scoring="f1", cv=cv,
            n_jobs=-1, random_state=random_state, verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = LogisticRegression(
            class_weight="balanced", max_iter=600, random_state=random_state
        )
        model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train Random Forest with optional randomised search.
    """
    if tune:
        param_dist = {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
        }
        base = RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=random_state
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=10, scoring="f1", cv=cv,
            n_jobs=-1, random_state=random_state, verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        )
        model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest.pkl"))
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Train XGBoost with scale_pos_weight to handle class imbalance natively.
    """
    n_neg  = int((y_train == 0).sum())
    n_pos  = int((y_train == 1).sum())
    spw    = n_neg / max(n_pos, 1)   # scale_pos_weight

    if tune:
        param_dist = {
            "n_estimators":   [100, 200, 300],
            "max_depth":      [3, 5, 7],
            "learning_rate":  [0.01, 0.05, 0.1, 0.2],
            "subsample":      [0.6, 0.8, 1.0],
            "colsample_bytree":[0.6, 0.8, 1.0],
        }
        base = XGBClassifier(
            scale_pos_weight=spw, use_label_encoder=False,
            eval_metric="aucpr", random_state=random_state,
            verbosity=0, n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=10, scoring="f1", cv=cv,
            n_jobs=-1, random_state=random_state, verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model = XGBClassifier(
            scale_pos_weight=spw, n_estimators=200, max_depth=5,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric="aucpr", random_state=random_state,
            verbosity=0, n_jobs=-1,
        )
        model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    return model


# ════════════════════════════════════════════════════════════════════════════
# Anomaly detection – Isolation Forest
# ════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.02,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest on raw (imbalanced) training data.
    contamination = expected fraction of outliers (fraud rate).
    Predictions: -1 → anomaly (fraud), +1 → normal.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    joblib.dump(model, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    return model


# ════════════════════════════════════════════════════════════════════════════
# Anomaly detection – Autoencoder (PyTorch)
# ════════════════════════════════════════════════════════════════════════════

class _Autoencoder(nn.Module if _TORCH_AVAILABLE else object):
    """
    Symmetric autoencoder for anomaly detection.
    Trained on legitimate transactions only; frauds produce high reconstruction error.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderWrapper:
    """
    Sklearn-compatible wrapper around the PyTorch autoencoder.
    Exposes .fit(), .reconstruction_error(), and .predict() methods.
    """

    def __init__(self, input_dim: int, epochs: int = 20,
                 lr: float = 1e-3, batch_size: int = 256):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for the Autoencoder model.")
        self.input_dim  = input_dim
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.threshold_ = None
        self.model_     = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "AutoencoderWrapper":
        """Train on legitimate (non-fraud) samples only."""
        X_legit = X_train[y_train == 0].astype(np.float32)
        model   = _Autoencoder(self.input_dim)
        opt     = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.tensor(X_legit))
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                opt.zero_grad()
                recon = model(batch)
                loss  = loss_fn(recon, batch)
                loss.backward()
                opt.step()

        self.model_ = model
        # Set threshold at 95th percentile of reconstruction error on training set
        errors = self._reconstruction_errors(X_legit)
        self.threshold_ = float(np.percentile(errors, 95))
        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            t     = torch.tensor(X.astype(np.float32))
            recon = self.model_(t).numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruction_errors(X.astype(np.float32))

    def predict_proba_fraud(self, X: np.ndarray) -> np.ndarray:
        """Return a [0,1] fraud probability based on reconstruction error."""
        errors = self.reconstruction_error(X)
        # Soft normalisation relative to threshold
        proba = np.clip(errors / (self.threshold_ * 3), 0, 1)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self.reconstruction_error(X)
        return (errors > self.threshold_).astype(int)

    def save(self):
        joblib.dump(self, os.path.join(MODELS_DIR, "autoencoder.pkl"))


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
) -> AutoencoderWrapper:
    """
    Train and save the autoencoder wrapper.
    Falls back gracefully if PyTorch is not available.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed – autoencoder skipped.")
    ae = AutoencoderWrapper(input_dim=X_train.shape[1], epochs=epochs)
    ae.fit(X_train, y_train)
    ae.save()
    return ae


# ════════════════════════════════════════════════════════════════════════════
# Convenience: load saved model
# ════════════════════════════════════════════════════════════════════════════

def load_model(name: str):
    """Load a persisted model by filename (without .pkl)."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
