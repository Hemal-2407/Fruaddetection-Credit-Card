# 🔍 AI-Powered Credit Card Fraud Detection System

> End-to-end ML fraud detection system with XGBoost, Isolation Forest, Autoencoder, SMOTE, threshold tuning, real-time simulation, and a Streamlit UI with downloadable reports.

---

## ✨ Features

| Category | Details |
|---|---|
| **Supervised Models** | Logistic Regression · Random Forest · XGBoost |
| **Anomaly Detection** | Isolation Forest · PyTorch Autoencoder |
| **Imbalance Handling** | SMOTE oversampling · Random undersampling · class_weight |
| **Hyperparameter Tuning** | RandomizedSearchCV (StratifiedKFold) |
| **Evaluation Metrics** | Precision · Recall · F1 · ROC-AUC · Avg Precision · Confusion Matrix |
| **Advanced Features** | Threshold tuning · Feature importance · Real-time stream simulation |
| **Visualizations** | Plotly interactive charts (ROC, PR, heatmap, boxplots, gauge) |
| **Reports** | Downloadable CSV + styled PDF (reportlab) |
| **Streamlit UI** | 6-tab interface with live prediction and simulation |

---

## 🗂️ Project Structure

```
fraud_detection/
├── app.py                          # Streamlit entry point (6 tabs)
├── requirements.txt
├── setup.sh                        # One-shot setup script
├── README.md
├── modules/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Load CSV / synthetic data, SMOTE, scaling
│   ├── eda_visualization.py        # All EDA charts (Plotly)
│   ├── model_training.py           # LR, RF, XGB, IsoForest, Autoencoder
│   ├── evaluation.py               # Metrics, charts, simulation, threshold tuning
│   └── report_generator.py        # CSV + PDF report generation
└── models/                         # Saved .pkl model files (auto-created)
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Create venv and install
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 📊 Dataset

The system works with the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (`creditcard.csv`).

If you don't upload a CSV, a **synthetic dataset** (10,000 transactions, 2% fraud) is generated automatically — perfect for demos.

### Dataset Schema
| Column | Description |
|--------|-------------|
| V1–V28 | PCA-transformed anonymised features |
| Amount | Transaction amount (USD) |
| Time | Seconds since first transaction |
| Class | 0 = Legitimate, 1 = Fraud |

---

## 🧠 Models

### Supervised
| Model | Key Strength |
|---|---|
| Logistic Regression | Fast baseline, interpretable coefficients |
| Random Forest | Handles non-linearity, built-in feature importance |
| XGBoost | State-of-the-art gradient boosting, `scale_pos_weight` for imbalance |

### Unsupervised (Anomaly Detection)
| Model | Mechanism |
|---|---|
| Isolation Forest | Isolates anomalies by random partitioning |
| Autoencoder | Trained on legit txns only; fraud → high reconstruction error |

---

## ⚙️ How It Works

```
Raw CSV / Synthetic Data
        │
        ▼
 Data Preprocessing
  • Missing values → median fill
  • Feature scaling (StandardScaler)
  • Train/test split (stratified 80/20)
        │
        ▼
 Imbalance Handling
  ┌─────┴──────────┐
SMOTE          Undersampling
        │
        ▼
 Model Training (with optional RandomizedSearchCV)
  LR · RF · XGB · IsoForest · Autoencoder
        │
        ▼
 Threshold Optimisation
  Find threshold maximising F1 / Recall / Precision
        │
        ▼
 Evaluation
  Confusion Matrix · ROC · PR Curves · Feature Importance
        │
        ▼
 Real-Time Simulation & Single Prediction UI
        │
        ▼
 Download Report (CSV / PDF)
```

---

## 📈 Streamlit UI Tabs

| Tab | Contents |
|---|---|
| 📊 Data & EDA | KPIs, class distribution, amount histogram, SMOTE comparison, correlation heatmap |
| 🤖 Model Training | Training summary, feature importance chart |
| 📈 Evaluation | Model comparison table, ROC/PR curves, confusion matrices, classification reports |
| ⚡ Simulation | Real-time transaction stream, threshold slider, animated scatter plot |
| 🎯 Predict | Single transaction prediction with gauge chart |
| ⬇️ Download | CSV + PDF report download |

---

## 📦 Key Libraries

| Library | Purpose |
|---|---|
| `scikit-learn` | LR, RF, Isolation Forest, metrics, preprocessing |
| `xgboost` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE, RandomUnderSampler |
| `torch` | PyTorch Autoencoder |
| `plotly` | Interactive charts |
| `reportlab` | PDF report generation |
| `streamlit` | Web UI |
| `pandas / numpy` | Data manipulation |
| `joblib` | Model persistence |

---

## 🔑 Key Design Decisions

1. **High Recall Priority** – Fraud detection favours catching frauds (recall) over precision. The system optimises thresholds for this.
2. **Multiple Resampling Strategies** – Users can switch between SMOTE, undersampling, and original data.
3. **Anomaly Detection** – Isolation Forest and Autoencoder detect fraud without labels, enabling unsupervised deployment.
4. **Modular Architecture** – Each concern is isolated in its own module for easy testing and extension.
5. **Graceful Fallbacks** – PyTorch optional; reportlab checked at runtime.

---

## 📄 License

MIT © 2024
