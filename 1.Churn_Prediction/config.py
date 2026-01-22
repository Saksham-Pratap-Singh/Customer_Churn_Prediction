# config.py - Configuration settings for the project

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
RAW_DATA_PATH = RAW_DATA_DIR / "telco_churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "churn_processed.csv"

# Model paths
BEST_MODEL_PATH = MODELS_DIR / "best_xgboost_model.pkl"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_SEARCH_ITERATIONS = 20
CROSS_VALIDATION_FOLDS = 5

# Target and features
TARGET_COLUMN = "churn"
CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service_type",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "paperless_billing",
    "payment_method"
]

NUMERICAL_FEATURES = [
    "age",
    "tenure",
    "monthly_charges",
    "total_charges"
]

# Class imbalance handling
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 0.7
RANDOM_SEARCH_N_ITER = 20

# Logistic Regression parameters
LOGISTIC_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000]
}

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# XGBoost parameters
XGBOOST_BASE_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'verbosity': 0
}

XGBOOST_TUNING_PARAMS = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.5, 1],
    'min_child_weight': [1, 3, 5]
}

# Evaluation metrics
METRICS = ['accuracy', 'recall', 'precision', 'roc_auc', 'f1', 'specificity']

# Thresholds for churn risk classification
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.7

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Streamlit configuration
STREAMLIT_THEME = "light"
MAX_UPLOAD_SIZE = 200  # MB