# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import warnings
warnings.filterwarnings("ignore")

# --Util functions--

# compute by entropy
def compute_entropy(series, bins=30):
    hist, _ = np.histogram(series.astype(float), bins=bins, density=True)
    return entropy(hist + 1e-9)

# remove the outliers
def remove_outliers_per_class(df, feats, target='class', k=1.5):
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

# transform large distributed features to log
def log_transform_large_range(df, feats, thr=100):
    df_out = df.copy()
    for col in feats:
        col_range = df_out[col].max() - df_out[col].min()
        if col_range > thr:
            min_val = df_out[col].min()
            shift = -min_val + 1 if min_val <= 0 else 0
            df_out[col] = np.log1p(df_out[col] + shift)
    return df_out

# scale top-5 entropy features 
def scale_high_entropy(df, feats, top_n=5):
    ent = {c: compute_entropy(df[c]) for c in feats}
    top_feats = sorted(ent, key=ent.get, reverse=True)[:top_n]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[top_feats] = scaler.fit_transform(df_scaled[top_feats].astype(float))
    return df_scaled, top_feats

# scale all features
def scale_all(df, feats):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feats] = scaler.fit_transform(df_scaled[feats].astype(float))
    return df_scaled

# main preprocessing function
def run_preprocessing(file_path,
                      drop_outliers=False,
                      log_large=False,
                      scale_entropy=False,
                      all_scale=False,
                      entropy_top_n=5,
                      rng_thr=100):
    df = pd.read_csv(file_path).dropna().reset_index(drop=True)

# Change the obejective features to categorical features for meachine learning
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'class':
            df[col] = df[col].astype('category')

# Remove the target feature
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

# Pre-processing execution function by strategy
def run_all_scenarios(file_path):
    scenarios = {
        'baseline': dict(drop_outliers=False, log_large=False, scale_entropy=False, all_scale=False),
        'outlier_removed': dict(drop_outliers=True, log_large=False, scale_entropy=False, all_scale=False),
        'log_transformed': dict(drop_outliers=False, log_large=True, scale_entropy=False, all_scale=False),
        'entropy_scaled_top5': dict(drop_outliers=False, log_large=False, scale_entropy=True, all_scale=False, entropy_top_n=5),
        'all_scaled': dict(drop_outliers=False, log_large=False, scale_entropy=False, all_scale=True)
    }
# Save in dictionary form
    print(f"\nFile: {file_path}")
    results = {}
    for name, params in scenarios.items():
        print(f"\nRunning: {name}")
        X, y, scaled = run_preprocessing(file_path, **params)
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        if scaled:
            print(f"Scaled features: {scaled}")
        results[name] = (X, y)
    return results

noNull_results = run_all_scenarios("noNull_data.csv")
processed_results = run_all_scenarios("processed_data.csv")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def evaluate_all_models(X, y, random_state=42):
    results = {}

# Handling categorical variables
    for col in X.select_dtypes(include='category').columns:
        X[col] = X[col].astype('category').cat.codes

# train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

# Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define hyperparameter space
    xgb_params = {
        'n_estimators': [100],
        'max_depth': [3, 4],
        'learning_rate': [0.1],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'min_child_weight': [1],
        'tree_method': ['gpu_hist']
    }

    rf_params = {
        'n_estimators': [100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    lgb_params = {
        'n_estimators': [100],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.1],
        'num_leaves': [31, 50],
        'subsample': [1.0],
        'colsample_bytree': [1.0]
    }

    base_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        'LightGBM': lgb.LGBMClassifier(random_state=42, is_unbalance=True)
    }

    for name, model in base_models.items():
        print(f"\n=== {name} ===")

        if name == 'Random Forest':
            print("Tuning Random Forest ...")
            search = RandomizedSearchCV(
                model, rf_params, n_iter=3, cv=3, random_state=42, n_jobs=-1,
                scoring='f1', verbose=1
            )
            search.fit(X_train_resampled, y_train_resampled)
            model = search.best_estimator_
            feature_names = X_train.columns
            importance = model.feature_importances_
            print(f"Best RF params: {search.best_params_}")

        elif name == 'XGBoost':
            print("Tuning XGBoost ...")
            search = RandomizedSearchCV(
                model, xgb_params, n_iter=3, cv=3, random_state=42, n_jobs=-1,
                scoring='f1', verbose=1
            )
            search.fit(X_train_resampled, y_train_resampled)
            model = search.best_estimator_
            feature_names = X_train.columns
            importance = model.feature_importances_
            print(f"Best XGB params: {search.best_params_}")

        elif name == 'LightGBM':
            print("Tuning LightGBM ...")
            search = RandomizedSearchCV(
                model, lgb_params, n_iter=3, cv=3, random_state=42, n_jobs=-1,
                scoring='f1', verbose=1
            )
            search.fit(X_train_resampled, y_train_resampled)
            model = search.best_estimator_
            feature_names = X_train.columns
            importance = model.feature_importances_
            print(f"Best LGBM params: {search.best_params_}")

        else:
            model.fit(X_train_resampled, y_train_resampled)
            feature_names = X_train.columns
            if hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0])
            else:
                importance = None

# Cross-validation
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=3, scoring='f1')

# Testset evaluation
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': report['macro avg']['f1-score'],
            'recall_class1': report['1']['recall'],
            'model': model,
            'feature_importance': pd.Series(importance, index=feature_names) if importance is not None else None
        }

    return results
# Perform evaluation by no-Nan dataset
print("\nEvaluating models for noNull dataset")
noNull_eval_results = {}
for strategy, (X, y) in noNull_results.items():
    print(f"\nStrategy: {strategy}")
    noNull_eval_results[strategy] = evaluate_all_models(X, y)

# Perform evaluation by feature-selected dataset
print("\nEvaluating models for processed dataset")
processed_eval_results = {}
for strategy, (X, y) in processed_results.items():
    print(f"\nStrategy: {strategy}")
    processed_eval_results[strategy] = evaluate_all_models(X, y)
# Merge and sort results
df_combined = pd.concat([df_noNull, df_processed], ignore_index=True)
df_combined = df_combined.sort_values(by=['F1_macro', 'Recall_1', 'Accuracy'], ascending=[False, False, False])
# Print
pd.set_option('display.max_rows', None)
print("\nOverall Results Summary:")
display(df_combined)
# Select the best combination
best_strategy = 'outlier_removed'
best_dataset = noNull_results[best_strategy]
X_full, y_full = best_dataset 

for col in X_full.select_dtypes(include='category').columns:
    X_full[col] = X_full[col].astype('category').cat.codes

# Split the selected best dataset into training and testing sets
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_final_resampled, y_train_final_resampled = smote.fit_resample(X_train_final, y_train_final)

# XGBoost hyperparameters (using the current settings)
model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1,
    tree_method='gpu_hist'
)

model.fit(X_train_final_resampled, y_train_final_resampled)

y_pred = model.predict(X_test_final)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix
cm = confusion_matrix(y_test_final, y_pred) 

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Final XGBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

report = classification_report(y_test_final, y_pred, output_dict=True) 
accuracy = accuracy_score(y_test_final, y_pred)

# Extract only the necessary values
summary = {
    'Accuracy': accuracy,
    'Class 0 Precision': report['0']['precision'],
    'Class 0 Recall': report['0']['recall'],
    'Class 0 F1': report['0']['f1-score'],
    'Class 1 Precision': report['1']['precision'],
    'Class 1 Recall': report['1']['recall'],
    'Class 1 F1': report['1']['f1-score'],
    'Macro Avg F1': report['macro avg']['f1-score'],
    'Weighted Avg F1': report['weighted avg']['f1-score']
}

# Display as a DataFrame
df_metrics = pd.DataFrame.from_dict(summary, orient='index', columns=['Score'])
print("\nFinal Model Performance Summary:")
display(df_metrics)
