"""
evaluation.py
-------------
Model evaluation utilities:
  - Classification report (precision / recall / F1 / AUC)
  - Confusion matrix (interactive Plotly)
  - ROC curve (multi-model overlay)
  - Precision-Recall curve
  - Feature importance bar chart
  - Threshold tuning (recall vs precision tradeoff)
  - Real-time fraud simulation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ── Classification summary ────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics for a single model.

    Returns
    -------
    dict with keys: name, threshold, precision, recall, f1, auc_roc,
                    avg_precision, report_str, y_pred, y_proba
    """
    # Probability or decision scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    elif hasattr(model, "predict_proba_fraud"):         # Autoencoder wrapper
        y_proba = model.predict_proba_fraud(X_test)
    else:
        y_proba = model.predict(X_test).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    return {
        "name":           model_name,
        "threshold":      threshold,
        "precision":      precision_score(y_test, y_pred, zero_division=0),
        "recall":         recall_score(y_test, y_pred, zero_division=0),
        "f1":             f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":        roc_auc_score(y_test, y_proba),
        "avg_precision":  average_precision_score(y_test, y_proba),
        "report_str":     classification_report(y_test, y_pred,
                              target_names=["Legitimate", "Fraud"],
                              zero_division=0),
        "y_pred":         y_pred,
        "y_proba":        y_proba,
    }


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of evaluate_model() dicts."""
    rows = []
    for r in results:
        rows.append({
            "Model":     r["name"],
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1-Score":  round(r["f1"],        4),
            "ROC-AUC":   round(r["auc_roc"],   4),
            "Avg Precision": round(r["avg_precision"], 4),
        })
    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False)


# ── Plotly chart helpers ──────────────────────────────────────────────────────

def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Model") -> go.Figure:
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["Legitimate", "Fraud"]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_roc_curves(
    results: List[Dict[str, Any]],
    y_test: np.ndarray,
) -> go.Figure:
    """Overlay ROC curves for multiple models."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{r['name']} (AUC={r['auc_roc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(dash="dash", color="grey"),
    ))
    fig.update_layout(
        title="ROC Curves – All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_pr_curves(
    results: List[Dict[str, Any]],
    y_test: np.ndarray,
) -> go.Figure:
    """Precision-Recall curves (more informative than ROC for imbalanced data)."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, r in enumerate(results):
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{r['name']} (AP={r['avg_precision']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    baseline = y_test.mean()
    fig.add_hline(y=baseline, line_dash="dash", line_color="grey",
                  annotation_text=f"Baseline ({baseline:.3f})")
    fig.update_layout(
        title="Precision-Recall Curves – All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_feature_importance(model, feature_names: List[str],
                             model_name: str = "Model", top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        fig = go.Figure()
        fig.update_layout(title=f"{model_name}: feature importance not available")
        return fig

    idx  = np.argsort(imp)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = imp[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals,
            colorscale="Blues",
            showscale=True,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Feature Importance — {model_name}",
        xaxis_title="Importance",
        height=max(300, top_n * 22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Threshold tuning ──────────────────────────────────────────────────────────

def plot_threshold_tuning(y_test: np.ndarray, y_proba: np.ndarray,
                          model_name: str = "Model") -> go.Figure:
    """
    Plot Precision, Recall, and F1 as a function of decision threshold.
    Helps choose the operating threshold that balances recall vs precision.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(   recall_score(y_test, y_pred, zero_division=0))
        f1s.append(       f1_score(y_test, y_pred, zero_division=0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, name="Precision",
                             line=dict(color="#1E88E5", width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=recalls,    name="Recall",
                             line=dict(color="#E53935",  width=2)))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s,        name="F1-Score",
                             line=dict(color="#43A047",  width=2)))

    best_f1_t = thresholds[np.argmax(f1s)]
    fig.add_vline(x=best_f1_t, line_dash="dash", line_color="orange",
                  annotation_text=f"Best F1 @ {best_f1_t:.2f}")

    fig.update_layout(
        title=f"Threshold Tuning — {model_name}",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def find_optimal_threshold(y_test: np.ndarray, y_proba: np.ndarray,
                            target: str = "f1") -> float:
    """
    Find the threshold that maximises the chosen metric.
    target: 'f1' | 'recall' | 'precision'
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_score = -1
    best_t     = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if target == "recall":
            score = recall_score(y_test, y_pred, zero_division=0)
        elif target == "precision":
            score = precision_score(y_test, y_pred, zero_division=0)
        else:
            score = f1_score(y_test, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_t     = t

    return float(best_t)


# ── Real-time fraud simulation ────────────────────────────────────────────────

def simulate_realtime_stream(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n: int = 200,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Simulate a real-time transaction stream returning a DataFrame
    with columns: transaction_id, true_label, predicted_label,
                  fraud_probability, correct, amount_approx
    """
    indices = np.random.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    Xs = X_test[indices]
    ys = y_test[indices]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(Xs)[:, 1]
    elif hasattr(model, "predict_proba_fraud"):
        probas = model.predict_proba_fraud(Xs)
    else:
        probas = model.predict(Xs).astype(float)

    preds   = (probas >= threshold).astype(int)
    amounts = np.abs(Xs[:, -1]) * 50 + 30   # rough back-approximation

    df = pd.DataFrame({
        "transaction_id":   np.arange(1, len(indices) + 1),
        "true_label":       ys,
        "predicted_label":  preds,
        "fraud_probability":np.round(probas, 4),
        "correct":          (ys == preds),
        "amount_approx":    np.round(amounts, 2),
    })
    return df


def plot_realtime_simulation(stream_df: pd.DataFrame) -> go.Figure:
    """
    Animated scatter plot of the transaction stream.
    X-axis = transaction index, Y-axis = fraud probability.
    Colour = actual class; symbol = prediction.
    """
    colors = stream_df["true_label"].map({0: "#1E88E5", 1: "#E53935"})
    symbols = stream_df["predicted_label"].map({0: "circle", 1: "x"})

    fig = go.Figure()

    for true_cls, grp in stream_df.groupby("true_label"):
        label = "Fraud" if true_cls == 1 else "Legitimate"
        fig.add_trace(go.Scatter(
            x=grp["transaction_id"],
            y=grp["fraud_probability"],
            mode="markers",
            name=label,
            marker=dict(
                color="#E53935" if true_cls == 1 else "#1E88E5",
                size=7,
                symbol=grp["predicted_label"].map({0: "circle", 1: "x"}),
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>Txn #%{x}</b><br>"
                "True: " + label + "<br>"
                "Fraud prob: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Real-Time Fraud Detection Simulation",
        xaxis_title="Transaction Index",
        yaxis_title="Fraud Probability",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_model_comparison(compare_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all model metrics side-by-side."""
    metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, row in compare_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison – Key Metrics",
        yaxis=dict(title="Score", range=[0, 1.05]),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
