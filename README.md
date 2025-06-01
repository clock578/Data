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

    if scale_entropy:
        df, scaled_cols = scale_high_entropy(df, num_cols, top_n=entropy_top_n)
    elif all_scale:
        df = scale_all(df, num_cols)
        scaled_cols = num_cols
    else:
        scaled_cols = []

    X = df.drop('class', axis=1)
    y = df['class']
    return X, y, scaled_cols

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
