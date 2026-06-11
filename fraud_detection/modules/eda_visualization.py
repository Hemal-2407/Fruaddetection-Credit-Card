"""
eda_visualization.py
--------------------
Generates all Exploratory Data Analysis charts using Plotly (interactive).
All functions return plotly Figure objects ready for st.plotly_chart().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(class_counts: pd.Series) -> go.Figure:
    """
    Donut chart showing fraud vs. legitimate transaction counts and percentages.
    """
    labels = ["Legitimate", "Fraud"]
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]
    colors = ["#1E88E5", "#E53935"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Class Distribution: Fraud vs Legitimate",
        showlegend=True,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_amount_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Overlapping histograms of transaction amounts for fraud vs. legitimate.
    """
    legit = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=legit.clip(upper=legit.quantile(0.99)),
        name="Legitimate", opacity=0.65,
        marker_color="#1E88E5", nbinsx=60,
    ))
    fig.add_trace(go.Histogram(
        x=fraud.clip(upper=fraud.quantile(0.99)),
        name="Fraud", opacity=0.75,
        marker_color="#E53935", nbinsx=60,
    ))
    fig.update_layout(
        barmode="overlay",
        title="Transaction Amount Distribution",
        xaxis_title="Amount (USD)",
        yaxis_title="Count",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> go.Figure:
    """
    Interactive correlation heatmap of the top features.
    """
    numeric = df.select_dtypes(include=np.number)
    # Keep the most variance-rich columns
    cols = numeric.var().nlargest(min(max_cols, len(numeric.columns))).index.tolist()
    if "Class" not in cols:
        cols.append("Class")
    corr = numeric[cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_smote_comparison(
    y_before: np.ndarray,
    y_smote: np.ndarray,
    y_under: np.ndarray,
) -> go.Figure:
    """
    Side-by-side grouped bar charts showing class distribution
    before SMOTE, after SMOTE, and after undersampling.
    """
    def counts(y):
        u, c = np.unique(y, return_counts=True)
        d = dict(zip(u.tolist(), c.tolist()))
        return d.get(0, 0), d.get(1, 0)

    scenarios = ["Original", "After SMOTE", "After Undersample"]
    legit_counts = []
    fraud_counts = []

    for y in [y_before, y_smote, y_under]:
        l, f = counts(y)
        legit_counts.append(l)
        fraud_counts.append(f)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Legitimate", x=scenarios, y=legit_counts,
                         marker_color="#1E88E5"))
    fig.add_trace(go.Bar(name="Fraud",      x=scenarios, y=fraud_counts,
                         marker_color="#E53935"))
    fig.update_layout(
        barmode="group",
        title="Class Distribution: Before & After Resampling",
        yaxis_title="Sample Count",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_transaction_time_pattern(df: pd.DataFrame) -> go.Figure:
    """
    Line chart of hourly transaction volume split by class (if Time column exists).
    """
    if "Time" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Time column not available in dataset")
        return fig

    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24

    hourly = df.groupby(["Hour", "Class"]).size().reset_index(name="Count")
    legit = hourly[hourly["Class"] == 0]
    fraud = hourly[hourly["Class"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=legit["Hour"], y=legit["Count"], mode="lines+markers",
                             name="Legitimate", line=dict(color="#1E88E5", width=2)))
    fig.add_trace(go.Scatter(x=fraud["Hour"], y=fraud["Count"], mode="lines+markers",
                             name="Fraud", line=dict(color="#E53935", width=2)))
    fig.update_layout(
        title="Hourly Transaction Pattern",
        xaxis_title="Hour of Day",
        yaxis_title="Transaction Count",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_feature_boxplots(df: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """
    Box plots of the most discriminative features (highest
    mean difference between fraud/legit).
    """
    numeric = df.select_dtypes(include=np.number).drop(columns=["Class", "Time"],
                                                        errors="ignore")
    legit = df[df["Class"] == 0][numeric.columns]
    fraud = df[df["Class"] == 1][numeric.columns]

    diff = (fraud.mean() - legit.mean()).abs().nlargest(top_n).index.tolist()

    fig = make_subplots(rows=2, cols=3, subplot_titles=diff)
    for idx, col in enumerate(diff):
        r, c = divmod(idx, 3)
        fig.add_trace(
            go.Box(y=df[df["Class"] == 0][col], name="Legit",
                   marker_color="#1E88E5", showlegend=(idx == 0)),
            row=r + 1, col=c + 1,
        )
        fig.add_trace(
            go.Box(y=df[df["Class"] == 1][col], name="Fraud",
                   marker_color="#E53935", showlegend=(idx == 0)),
            row=r + 1, col=c + 1,
        )
    fig.update_layout(title="Top Discriminative Features (Box Plot)",
                      height=500, paper_bgcolor="rgba(0,0,0,0)")
    return fig
