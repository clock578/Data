<<<<<<< HEAD
Machine Learning Pipeline (Pandas / Scikit-learn Style)
=======
# Machine Learning Pipeline (Pandas / Scikit-learn Style)
>>>>>>> 6d452376b37f561e0c9915b242efad4b71e26629
Overview
This pipeline reads a CSV dataset, automatically applies multiple preprocessing scenarios, and performs training and evaluation on major machine learning models.
It includes handling imbalanced data with SMOTE, hyperparameter tuning, and cross-validation evaluation.

<<<<<<< HEAD
Key Components
1. Utility Functions
compute_entropy(series: pd.Series, bins: int = 30) -> float
Calculates the entropy (uncertainty) of a given numerical series based on histogram bins.

Input:

series: numerical Pandas Series

bins: number of histogram bins (default 30)

Output: entropy value (float)

remove_outliers_per_class(df: pd.DataFrame, feats: list, target: str = 'class', k: float = 1.5) -> pd.DataFrame
Removes outliers per target class using the IQR method for each feature.

Input:

df: DataFrame

feats: list of numerical feature names

target: target column name (default 'class')

k: IQR multiplier constant (default 1.5)
=======
**1. Utility Functions**
compute_entropy(series: pd.Series, bins: int = 30) -> float
Calculates the entropy (uncertainty) of a given numerical series based on histogram bins.

Input: series: numerical Pandas Series
bins: number of histogram bins (default 30)

Output: entropy value (float)
remove_outliers_per_class(df: pd.DataFrame, feats: list, target: str = 'class', k: float = 1.5) -> pd.DataFrame
Removes outliers per target class using the IQR method for each feature.
Input: df: DataFrame
feats: list of numerical feature names
target: target column name (default 'class')
k: IQR multiplier constant (default 1.5)

Output: DataFrame with outliers removed
log_transform_large_range(df: pd.DataFrame, feats: list, thr: float = 100) -> pd.DataFrame
Applies log transformation to features with large value ranges to stabilize distributions.

Input: df: DataFrame
feats: list of numerical feature names
thr: threshold for value range (default 100)

Output: DataFrame with log-transformed features
scale_high_entropy(df: pd.DataFrame, feats: list, top_n: int = 5) -> (pd.DataFrame, list)
Scales the top N features with the highest entropy using StandardScaler.

Input:df: DataFrame
feats: list of numerical feature names
top_n: number of features to scale (default 5)

Output: scaled DataFrame and list of scaled feature names
scale_all(df: pd.DataFrame, feats: list) -> pd.DataFrame
Scales all specified numerical features using StandardScaler.

Input:df: DataFrame
feats: list of numerical feature names

Output: scaled DataFrame

**2. Preprocessing Function**
run_preprocessing(file_path: str, drop_outliers=False, log_large=False, scale_entropy=False, all_scale=False, entropy_top_n=5, rng_thr=100) -> (pd.DataFrame, pd.Series, list)
Reads a CSV file and processes data based on various preprocessing options, returning the feature matrix X, target vector y, and list of scaled columns.
Parameters: file_path: path to the CSV data file
drop_outliers: whether to remove outliers per class (default False)
log_large: whether to apply log transformation on features with large ranges (default False)
scale_entropy: whether to scale top N entropy features (default False)
all_scale: whether to scale all numerical features (default False)
entropy_top_n: number of top entropy features to scale (default 5)
rng_thr: threshold for log transformation (default 100)
Returns:
X: preprocessed feature DataFrame
y: target Series
scaled_cols: list of scaled feature names
Processing steps:Load data and drop missing values
Convert string columns (excluding target) to categorical type
Extract numerical features
Apply outlier removal, log transformation, and scaling options
Split into features X and target y

**3. Run All Preprocessing Scenarios**
run_all_scenarios(file_path: str) -> dict
Executes the following five preprocessing scenarios and returns results as a dictionary:
baseline (original data)
outlier_removed (outliers removed)
log_transformed (log transformation applied)
entropy_scaled_top5 (top 5 entropy features scaled)
all_scaled (all numerical features scaled)
Returns: dictionary mapping scenario names to (X, y) tuples

**4. Model Evaluation**
evaluate_all_models(X: pd.DataFrame, y: pd.Series, random_state=42) -> dict
Trains and evaluates four classification models on the input data, returning cross-validation scores, test accuracy, macro F1 score, recall for class 1, and feature importances (if available).
Models:
**Logistic Regression** (with balanced class weights)
**Random Forest** (with hyperparameter tuning)
**XGBoost** (with tuning)
**LightGBM** (with tuning)

Process:
Convert categorical features to numeric codes
Split dataset into training and test sets
Apply SMOTE oversampling on training data
Train each model and perform randomized hyperparameter search if applicable
Predict on test set and calculate evaluation metrics
Calculate feature importances when possible
Returns: dictionary mapping model names to dictionaries containing cross-validation mean and std, test accuracy, macro F1, class 1 recall, trained model object, and feature importances

```python
# ==============================
# Machine Learning Pipeline (Pandas / Scikit-learn Style)
# ==============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==============================
# Utility Functions
# ==============================

def compute_entropy(series: pd.Series, bins: int = 30) -> float:
    hist, _ = np.histogram(series.astype(float), bins=bins, density=True)
    return entropy(hist + 1e-9)

def remove_outliers_per_class(df: pd.DataFrame, feats: list, target: str = 'class', k: float = 1.5) -> pd.DataFrame:
    keep_parts = []
    for label in df[target].unique():
        sub = df[df[target] == label].copy()
        mask = np.ones(len(sub), dtype=bool)
        for col in feats:
            q1, q3 = sub[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            mask &= sub[col].between(lower, upper)
        keep_parts.append(sub[mask])
    return pd.concat(keep_parts, ignore_index=True)

def log_transform_large_range(df: pd.DataFrame, feats: list, thr: float = 100) -> pd.DataFrame:
    df_out = df.copy()
    for col in feats:
        col_range = df_out[col].max() - df_out[col].min()
        if col_range > thr:
            shift = -df_out[col].min() + 1 if df_out[col].min() <= 0 else 0
            df_out[col] = np.log1p(df_out[col] + shift)
    return df_out

def scale_high_entropy(df: pd.DataFrame, feats: list, top_n: int = 5):
    entropies = {c: compute_entropy(df[c]) for c in feats}
    top_feats = sorted(entropies, key=entropies.get, reverse=True)[:top_n]
    df_scaled = df.copy()
    df_scaled[top_feats] = StandardScaler().fit_transform(df_scaled[top_feats].astype(float))
    return df_scaled, top_feats

def scale_all(df: pd.DataFrame, feats: list) -> pd.DataFrame:
    df_scaled = df.copy()
    df_scaled[feats] = StandardScaler().fit_transform(df_scaled[feats].astype(float))
    return df_scaled

# ==============================
# Preprocessing Function
# ==============================

def run_preprocessing(file_path: str,
                      drop_outliers: bool = False,
                      log_large: bool = False,
                      scale_entropy: bool = False,
                      all_scale: bool = False,
                      entropy_top_n: int = 5,
                      rng_thr: float = 100):
    df = pd.read_csv(file_path).dropna().reset_index(drop=True)
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'class':
            df[col] = df[col].astype('category')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'class' in num_cols:
        num_cols.remove('class')

    if drop_outliers:
        df = remove_outliers_per_class(df, num_cols, target='class', k=2.0)
    if log_large:
        df = log_transform_large_range(df, num_cols, thr=rng_thr)
>>>>>>> 6d452376b37f561e0c9915b242efad4b71e26629

Output: DataFrame with outliers removed

log_transform_large_range(df: pd.DataFrame, feats: list, thr: float = 100) -> pd.DataFrame
Applies log transformation to features with large value ranges to stabilize distributions.

<<<<<<< HEAD
Input:

df: DataFrame

feats: list of numerical feature names

thr: threshold for value range (default 100)

Output: DataFrame with log-transformed features

scale_high_entropy(df: pd.DataFrame, feats: list, top_n: int = 5) -> (pd.DataFrame, list)
Scales the top N features with the highest entropy using StandardScaler.

Input:

df: DataFrame

feats: list of numerical feature names

top_n: number of features to scale (default 5)

Output: scaled DataFrame and list of scaled feature names

scale_all(df: pd.DataFrame, feats: list) -> pd.DataFrame
Scales all specified numerical features using StandardScaler.

Input:

df: DataFrame

feats: list of numerical feature names

Output: scaled DataFrame

2. Preprocessing Function
run_preprocessing(file_path: str, drop_outliers=False, log_large=False, scale_entropy=False, all_scale=False, entropy_top_n=5, rng_thr=100) -> (pd.DataFrame, pd.Series, list)
Reads a CSV file and processes data based on various preprocessing options, returning the feature matrix X, target vector y, and list of scaled columns.

Parameters:

file_path: path to the CSV data file

drop_outliers: whether to remove outliers per class (default False)

log_large: whether to apply log transformation on features with large ranges (default False)

scale_entropy: whether to scale top N entropy features (default False)

all_scale: whether to scale all numerical features (default False)

entropy_top_n: number of top entropy features to scale (default 5)

rng_thr: threshold for log transformation (default 100)

Returns:

X: preprocessed feature DataFrame

y: target Series

scaled_cols: list of scaled feature names

Processing steps:

Load data and drop missing values

Convert string columns (excluding target) to categorical type

Extract numerical features

Apply outlier removal, log transformation, and scaling options

Split into features X and target y

3. Run All Preprocessing Scenarios
run_all_scenarios(file_path: str) -> dict
Executes the following five preprocessing scenarios and returns results as a dictionary:

baseline (original data)

outlier_removed (outliers removed)

log_transformed (log transformation applied)

entropy_scaled_top5 (top 5 entropy features scaled)

all_scaled (all numerical features scaled)

Returns: dictionary mapping scenario names to (X, y) tuples

4. Model Evaluation
evaluate_all_models(X: pd.DataFrame, y: pd.Series, random_state=42) -> dict
Trains and evaluates four classification models on the input data, returning cross-validation scores, test accuracy, macro F1 score, recall for class 1, and feature importances (if available).

Models:

Logistic Regression (with balanced class weights)

Random Forest (with hyperparameter tuning)

XGBoost (with tuning)

LightGBM (with tuning)

Process:

Convert categorical features to numeric codes

Split dataset into training and test sets

Apply SMOTE oversampling on training data

Train each model and perform randomized hyperparameter search if applicable

Predict on test set and calculate evaluation metrics

Calculate feature importances when possible

Returns: dictionary mapping model names to dictionaries containing cross-validation mean and std, test accuracy, macro F1, class 1 recall, trained model object, and feature importances
=======
# ==============================
# Run All Preprocessing Scenarios
# ==============================

def run_all_scenarios(file_path: str) -> dict:
    scenarios = {
        'baseline': dict(drop_outliers=False, log_large=False, scale_entropy=False, all_scale=False),
        'outlier_removed': dict(drop_outliers=True, log_large=False, scale_entropy=False, all_scale=False),
        'log_transformed': dict(drop_outliers=False, log_large=True, scale_entropy=False, all_scale=False),
        'entropy_scaled_top5': dict(drop_outliers=False, log_large=False, scale_entropy=True, all_scale=False),
        'all_scaled': dict(drop_outliers=False, log_large=False, scale_entropy=False, all_scale=True)
    }
    results = {}
    print(f"\nFile: {file_path}")
    for name, params in scenarios.items():
        print(f"\nRunning: {name}")
        X, y, scaled = run_preprocessing(file_path, **params)
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        if scaled:
            print(f"Scaled features: {scaled}")
        results[name] = (X, y)
    return results

# ==============================
# Model Evaluation
# ==============================

def evaluate_all_models(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    results = {}
    X = X.copy()
    for col in X.select_dtypes(include='category').columns:
        X[col] = X[col].cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train_resampled, y_train_resampled = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
        'LightGBM': lgb.LGBMClassifier(is_unbalance=True, random_state=random_state)
    }

    param_spaces = {
        'Random Forest': {
            'n_estimators': [100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'n_estimators': [100],
            'max_depth': [3, 4],
            'learning_rate': [0.1],
            'subsample': [1.0],
            'colsample_bytree': [1.0],
            'min_child_weight': [1],
            'tree_method': ['gpu_hist']
        },
        'LightGBM': {
            'n_estimators': [100],
            'max_depth': [-1, 10, 20],
            'learning_rate': [0.1],
            'num_leaves': [31, 50],
            'subsample': [1.0],
            'colsample_bytree': [1.0]
        }
    }

    for name, model in base_models.items():
        print(f"\n=== {name} ===")
        if name in param_spaces:
            print(f"Tuning {name} ...")
            model = RandomizedSearchCV(
                model,
                param_spaces[name],
                n_iter=3,
                cv=3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            ).fit(X_train_resampled, y_train_resampled).best_estimator_

        else:
            model.fit(X_train_resampled, y_train_resampled)

        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=3, scoring='f1')
        report = classification_report(y_test, y_pred, output_dict=True)

        importance = None
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=X.columns)
        elif hasattr(model, "coef_"):
            importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)

        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': report['macro avg']['f1-score'],
            'recall_class1': report['1']['recall'],
            'model': model,
            'feature_importance': importance
        }

    return results
```
