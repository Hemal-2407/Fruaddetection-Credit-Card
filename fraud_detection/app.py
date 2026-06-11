"""
app.py  –  Credit Card Fraud Detection System
=============================================
Streamlit UI with 5 tabs:
  1. Data & EDA
  2. Model Training
  3. Evaluation & Comparison
  4. Real-Time Simulation & Threshold Tuning
  5. Download Report

Run:
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from modules.data_preprocessing import load_and_preprocess
from modules.eda_visualization   import (
    plot_class_distribution, plot_amount_distribution,
    plot_correlation_heatmap, plot_smote_comparison,
    plot_transaction_time_pattern, plot_feature_boxplots,
)
from modules.model_training import (
    train_logistic_regression, train_random_forest,
    train_xgboost, train_isolation_forest, train_autoencoder,
)
from modules.evaluation import (
    evaluate_model, compare_models,
    plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
    plot_feature_importance, plot_threshold_tuning,
    find_optimal_threshold, simulate_realtime_stream,
    plot_realtime_simulation, plot_model_comparison,
)
from modules.report_generator import generate_csv_report, generate_pdf_report


# ════════════════════════════════════════════════════════════════════════════
# Page setup
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1565C0 0%, #E53935 100%);
    color: white; padding: 2rem; border-radius: 14px;
    margin-bottom: 1.5rem; text-align: center;
}
.kpi-card {
    background: #F5F5F5; border-radius: 10px;
    padding: 1rem; border-left: 5px solid #1E88E5;
    margin-bottom: 0.8rem;
}
.fraud-alert {
    background: #FFEBEE; border-left: 5px solid #E53935;
    padding: 1rem; border-radius: 8px;
}
.safe-alert {
    background: #E8F5E9; border-left: 5px solid #43A047;
    padding: 1rem; border-radius: 8px;
}
.badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 12px; font-size: 0.82rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 AI-Powered Credit Card Fraud Detection</h1>
    <p>XGBoost · Random Forest · Isolation Forest · Autoencoder · SMOTE · Threshold Tuning</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ════════════════════════════════════════════════════════════════════════════
for key in ["data", "models", "eval_results", "compare_df"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📂 Dataset")
    uploaded_csv = st.file_uploader(
        "Upload creditcard.csv (optional)",
        type=["csv"],
        help="Kaggle Credit Card Fraud Detection dataset. "
             "If not uploaded, a synthetic dataset is generated.",
    )

    st.subheader("🧪 Models to Train")
    train_lr  = st.checkbox("Logistic Regression", value=True)
    train_rf  = st.checkbox("Random Forest",       value=True)
    train_xgb = st.checkbox("XGBoost",             value=True)
    train_iso = st.checkbox("Isolation Forest",    value=True)
    train_ae  = st.checkbox("Autoencoder (PyTorch)", value=False,
                             help="Requires PyTorch. Slower to train.")

    st.subheader("🔧 Options")
    use_smote     = st.radio("Training data",
                             ["SMOTE (oversampling)", "Undersampling", "Original"],
                             index=0)
    hyperparameter_tune = st.checkbox("Hyperparameter tuning (slower)", value=False)
    threshold_mode = st.selectbox("Optimal threshold target",
                                  ["f1", "recall", "precision"], index=0)

    st.divider()
    run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary")


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ════════════════════════════════════════════════════════════════════════════
if run_btn:
    prog = st.progress(0, text="Loading & preprocessing data …")

    # 1. Preprocess
    filepath = None
    if uploaded_csv:
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(uploaded_csv, tmp)
            filepath = tmp.name

    data = load_and_preprocess(filepath=filepath)
    st.session_state["data"] = data
    prog.progress(15, text="Data ready ✓  –  Training models …")

    # Select training arrays based on resampling choice
    if "SMOTE" in use_smote:
        X_tr = data["X_train_smote"]
        y_tr = data["y_train_smote"]
    elif "Under" in use_smote:
        X_tr = data["X_train_under"]
        y_tr = data["y_train_under"]
    else:
        X_tr = data["X_train"]
        y_tr = data["y_train"]

    tune     = hyperparameter_tune
    models   = {}
    results  = []
    n_models = sum([train_lr, train_rf, train_xgb, train_iso, train_ae])
    step     = 70 // max(n_models, 1)
    current  = 15

    def _eval(model, name):
        opt_t = find_optimal_threshold(data["y_test"],
                    model.predict_proba(data["X_test"])[:, 1]
                    if hasattr(model, "predict_proba") else
                    model.predict(data["X_test"]).astype(float),
                    target=threshold_mode)
        return evaluate_model(model, data["X_test"], data["y_test"],
                              model_name=name, threshold=opt_t)

    if train_lr:
        prog.progress(current, text="Training Logistic Regression …")
        m = train_logistic_regression(X_tr, y_tr, tune=tune)
        models["Logistic Regression"] = m
        results.append(_eval(m, "Logistic Regression"))
        current += step

    if train_rf:
        prog.progress(current, text="Training Random Forest …")
        m = train_random_forest(X_tr, y_tr, tune=tune)
        models["Random Forest"] = m
        results.append(_eval(m, "Random Forest"))
        current += step

    if train_xgb:
        prog.progress(current, text="Training XGBoost …")
        m = train_xgboost(X_tr, y_tr, tune=tune)
        models["XGBoost"] = m
        results.append(_eval(m, "XGBoost"))
        current += step

    if train_iso:
        prog.progress(current, text="Training Isolation Forest …")
        fraud_rate = float(data["class_counts"].get(1, 200) /
                           data["class_counts"].sum())
        m = train_isolation_forest(data["X_train"], contamination=fraud_rate)
        models["Isolation Forest"] = m
        # Isolation Forest: -1 = anomaly → 1 (fraud)
        raw_pred = m.predict(data["X_test"])
        y_iso = np.where(raw_pred == -1, 1, 0)
        scores = m.score_samples(data["X_test"])
        scores_norm = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        from sklearn.metrics import (precision_score, recall_score, f1_score,
                                      roc_auc_score, average_precision_score,
                                      classification_report)
        results.append({
            "name":          "Isolation Forest",
            "threshold":     fraud_rate,
            "precision":     precision_score(data["y_test"], y_iso, zero_division=0),
            "recall":        recall_score(data["y_test"], y_iso, zero_division=0),
            "f1":            f1_score(data["y_test"], y_iso, zero_division=0),
            "auc_roc":       roc_auc_score(data["y_test"], scores_norm),
            "avg_precision": average_precision_score(data["y_test"], scores_norm),
            "report_str":    classification_report(data["y_test"], y_iso,
                                 target_names=["Legitimate", "Fraud"],
                                 zero_division=0),
            "y_pred":        y_iso,
            "y_proba":       scores_norm,
        })
        current += step

    if train_ae:
        prog.progress(current, text="Training Autoencoder …")
        try:
            m = train_autoencoder(X_tr, y_tr, epochs=15)
            models["Autoencoder"] = m
            y_ae    = m.predict(data["X_test"])
            probas  = m.predict_proba_fraud(data["X_test"])
            from sklearn.metrics import (precision_score, recall_score, f1_score,
                                          roc_auc_score, average_precision_score,
                                          classification_report)
            results.append({
                "name":          "Autoencoder",
                "threshold":     m.threshold_,
                "precision":     precision_score(data["y_test"], y_ae, zero_division=0),
                "recall":        recall_score(data["y_test"], y_ae, zero_division=0),
                "f1":            f1_score(data["y_test"], y_ae, zero_division=0),
                "auc_roc":       roc_auc_score(data["y_test"], probas),
                "avg_precision": average_precision_score(data["y_test"], probas),
                "report_str":    classification_report(data["y_test"], y_ae,
                                     target_names=["Legitimate", "Fraud"],
                                     zero_division=0),
                "y_pred":        y_ae,
                "y_proba":       probas,
            })
        except Exception as exc:
            st.warning(f"Autoencoder skipped: {exc}")
        current += step

    compare_df = compare_models(results)
    st.session_state["models"]       = models
    st.session_state["eval_results"] = results
    st.session_state["compare_df"]   = compare_df

    prog.progress(100, text="✅ Pipeline complete!")
    time.sleep(0.5)
    prog.empty()
    st.success(f"✅ Trained {len(models)} model(s). Best model by F1: **{compare_df.iloc[0]['Model']}**")


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & EDA",
    "🤖 Model Training",
    "📈 Evaluation",
    "⚡ Simulation",
    "🎯 Predict",
    "⬇️ Download",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 – Data & EDA
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if st.session_state["data"] is None:
        st.info("Run the pipeline from the sidebar to load data.")
    else:
        data = st.session_state["data"]
        df   = data["df_raw"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Transactions", f"{len(df):,}")
        k2.metric("Legitimate",  f"{data['class_counts'].get(0, 0):,}")
        k3.metric("Fraud Cases", f"{data['class_counts'].get(1, 0):,}")
        fraud_pct = data['class_counts'].get(1, 0) / len(df) * 100
        k4.metric("Fraud Rate", f"{fraud_pct:.3f}%")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_class_distribution(data["class_counts"]),
                            use_container_width=True)
        with c2:
            st.plotly_chart(plot_amount_distribution(df), use_container_width=True)

        st.plotly_chart(plot_smote_comparison(
            data["y_train"], data["y_train_smote"], data["y_train_under"]
        ), use_container_width=True)

        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        st.plotly_chart(plot_transaction_time_pattern(df), use_container_width=True)
        st.plotly_chart(plot_feature_boxplots(df), use_container_width=True)

        st.subheader("Raw Data Sample")
        st.dataframe(df.head(50), use_container_width=True, height=300)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 – Model Training Summary
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if st.session_state["models"] is None:
        st.info("Run the pipeline from the sidebar.")
    else:
        models = st.session_state["models"]
        data   = st.session_state["data"]

        st.subheader("Trained Models")
        for name in models:
            st.markdown(f"✅ **{name}** – saved to `models/`")

        st.divider()

        # Feature importance for the best supervised model
        best_name = st.session_state["compare_df"].iloc[0]["Model"]
        if best_name in models:
            st.subheader(f"Feature Importance — {best_name}")
            st.plotly_chart(
                plot_feature_importance(
                    models[best_name],
                    data["feature_names"],
                    model_name=best_name,
                ),
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 – Evaluation
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if st.session_state["eval_results"] is None:
        st.info("Run the pipeline from the sidebar.")
    else:
        results    = st.session_state["eval_results"]
        compare_df = st.session_state["compare_df"]
        y_test     = st.session_state["data"]["y_test"]

        st.subheader("Model Comparison")
        st.dataframe(
            compare_df.style.highlight_max(
                subset=["Recall", "F1-Score", "ROC-AUC"],
                color="#C8E6C9",
            ).highlight_min(
                subset=["Recall"],
                color="#FFCDD2",
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(plot_model_comparison(compare_df), use_container_width=True)
        st.plotly_chart(plot_roc_curves(results, y_test),  use_container_width=True)
        st.plotly_chart(plot_pr_curves(results,  y_test),  use_container_width=True)

        st.divider()
        st.subheader("Confusion Matrices")
        cols = st.columns(min(3, len(results)))
        for i, r in enumerate(results):
            cols[i % len(cols)].plotly_chart(
                plot_confusion_matrix(y_test, r["y_pred"], r["name"]),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Classification Reports")
        for r in results:
            with st.expander(f"{r['name']} — threshold {r['threshold']:.3f}"):
                st.code(r["report_str"], language=None)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 – Real-Time Simulation & Threshold Tuning
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    if st.session_state["eval_results"] is None:
        st.info("Run the pipeline from the sidebar.")
    else:
        results = st.session_state["eval_results"]
        data    = st.session_state["data"]
        models  = st.session_state["models"]

        # Pick best supervised model for simulation
        sup_names  = [r["name"] for r in results
                      if r["name"] not in ("Isolation Forest", "Autoencoder")]
        sim_model_name = st.selectbox("Select model for simulation", sup_names or
                                      [r["name"] for r in results])
        sim_model = models.get(sim_model_name)

        # Threshold slider
        default_t = next((r["threshold"] for r in results
                          if r["name"] == sim_model_name), 0.5)
        threshold = st.slider("Decision Threshold", 0.01, 0.99,
                              float(default_t), 0.01)

        # Threshold tuning plot
        chosen_result = next((r for r in results if r["name"] == sim_model_name), None)
        if chosen_result:
            st.plotly_chart(
                plot_threshold_tuning(data["y_test"], chosen_result["y_proba"],
                                      sim_model_name),
                use_container_width=True,
            )

        # Real-time simulation
        st.subheader("⚡ Real-Time Stream Simulation")
        n_sim = st.slider("Transactions to simulate", 50, 500, 200, 50)
        if st.button("▶ Run Simulation"):
            if sim_model:
                stream = simulate_realtime_stream(
                    sim_model, data["X_test"], data["y_test"],
                    n=n_sim, threshold=threshold,
                )
                st.plotly_chart(plot_realtime_simulation(stream),
                                use_container_width=True)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Transactions", len(stream))
                s2.metric("Fraud Detected", int((stream["predicted_label"] == 1).sum()))
                s3.metric("Correct Predictions",
                           f"{stream['correct'].mean()*100:.1f}%")
                s4.metric("True Frauds",
                           int((stream["true_label"] == 1).sum()))

                st.dataframe(
                    stream.style.apply(
                        lambda row: [
                            "background-color: #FFEBEE" if row["true_label"] == 1
                            else "" for _ in row
                        ],
                        axis=1,
                    ),
                    use_container_width=True,
                    height=300,
                )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 – Single Transaction Prediction
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    if st.session_state["models"] is None:
        st.info("Run the pipeline from the sidebar first.")
    else:
        models   = st.session_state["models"]
        data     = st.session_state["data"]
        results  = st.session_state["eval_results"]
        features = data["feature_names"]

        st.subheader("🎯 Single Transaction Fraud Prediction")
        st.markdown("Enter transaction details or use a sample from the test set.")

        col_l, col_r = st.columns([1, 2])

        with col_l:
            pred_model_name = st.selectbox(
                "Model", [r["name"] for r in results
                          if r["name"] not in ("Isolation Forest", "Autoencoder")]
                         or list(models.keys())
            )
            pred_threshold = st.slider("Threshold", 0.01, 0.99, 0.5, 0.01,
                                       key="pred_thresh")

            use_sample = st.radio("Input method",
                                  ["Use test set sample", "Manual input"])

            if use_sample == "Use test set sample":
                sample_idx = st.slider("Test sample index", 0,
                                       len(data["X_test"]) - 1, 0)
                x_input    = data["X_test"][[sample_idx]]
                true_label = data["y_test"][sample_idx]
                st.info(f"True label: **{'FRAUD 🚨' if true_label == 1 else 'Legitimate ✅'}**")
            else:
                st.markdown("Enter V1–V5 and Amount (simplified):")
                vals = []
                for i in range(1, 6):
                    vals.append(st.number_input(f"V{i}", value=0.0, format="%.4f"))
                amount = st.number_input("Amount ($)", value=50.0, min_value=0.0)
                # Pad remaining features with zeros
                n_feat  = len(features)
                padded  = vals + [0.0] * (n_feat - len(vals) - 1) + [amount]
                x_input = np.array(padded[:n_feat]).reshape(1, -1)
                x_input = data["scaler"].transform(x_input)
                true_label = None

        with col_r:
            if st.button("🔍 Predict", use_container_width=True):
                model = models.get(pred_model_name)
                if model is None:
                    st.error("Model not found.")
                else:
                    if hasattr(model, "predict_proba"):
                        fraud_prob = float(model.predict_proba(x_input)[0, 1])
                    elif hasattr(model, "predict_proba_fraud"):
                        fraud_prob = float(model.predict_proba_fraud(x_input)[0])
                    else:
                        fraud_prob = float(model.predict(x_input)[0])

                    is_fraud = fraud_prob >= pred_threshold

                    if is_fraud:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <h2>🚨 FRAUD DETECTED</h2>
                            <p>Fraud Probability: <strong>{fraud_prob*100:.2f}%</strong></p>
                            <p>Threshold: {pred_threshold:.2f} | Model: {pred_model_name}</p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h2>✅ LEGITIMATE TRANSACTION</h2>
                            <p>Fraud Probability: <strong>{fraud_prob*100:.2f}%</strong></p>
                            <p>Threshold: {pred_threshold:.2f} | Model: {pred_model_name}</p>
                        </div>""", unsafe_allow_html=True)

                    st.progress(fraud_prob,
                                text=f"Fraud probability: {fraud_prob*100:.1f}%")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=fraud_prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Fraud Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#E53935" if is_fraud else "#43A047"},
                            "steps": [
                                {"range": [0, 40],  "color": "#E8F5E9"},
                                {"range": [40, 70], "color": "#FFF9C4"},
                                {"range": [70, 100],"color": "#FFEBEE"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 3},
                                "thickness": 0.75,
                                "value": pred_threshold * 100,
                            },
                        },
                        number={"suffix": "%"},
                    ))
                    fig.update_layout(height=300,
                                      paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 – Download Report
# ──────────────────────────────────────────────────────────────────────────────
with tab6:
    if st.session_state["eval_results"] is None:
        st.info("Run the pipeline from the sidebar first.")
    else:
        results    = st.session_state["eval_results"]
        compare_df = st.session_state["compare_df"]
        data       = st.session_state["data"]

        st.subheader("⬇️ Download Evaluation Report")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📊 CSV Report")
            csv_bytes = generate_csv_report(compare_df, results)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name="fraud_detection_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c2:
            st.markdown("### 📄 PDF Report")
            try:
                cc = data["class_counts"]
                dataset_info = {
                    "total": int(cc.sum()),
                    "legit": int(cc.get(0, 0)),
                    "fraud": int(cc.get(1, 0)),
                }
                pdf_bytes = generate_pdf_report(compare_df, results, dataset_info)
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="fraud_detection_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except RuntimeError as e:
                st.warning(str(e))
