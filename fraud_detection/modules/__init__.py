# modules/__init__.py
from .data_preprocessing import load_and_preprocess, generate_synthetic_dataset
from .eda_visualization  import (
    plot_class_distribution, plot_amount_distribution,
    plot_correlation_heatmap, plot_smote_comparison,
    plot_transaction_time_pattern, plot_feature_boxplots,
)
from .model_training import (
    train_logistic_regression, train_random_forest,
    train_xgboost, train_isolation_forest, train_autoencoder,
    load_model, AutoencoderWrapper,
)
from .evaluation import (
    evaluate_model, compare_models,
    plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
    plot_feature_importance, plot_threshold_tuning,
    find_optimal_threshold, simulate_realtime_stream,
    plot_realtime_simulation, plot_model_comparison,
)
from .report_generator import generate_csv_report, generate_pdf_report
