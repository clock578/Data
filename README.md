"""
comprehensive_ml_pipeline.py

A comprehensive machine learning pipeline that performs automatic preprocessing, model training,
hyperparameter tuning, evaluation, and summary of the best-performing configurations.

Author: [Your Name or GitHub handle]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def comprehensive_ml_pipeline(file_path, target_column, test_size=0.2, random_state=42, n_iter=5):
    """
    Perform a comprehensive ML pipeline: preprocessing, model tuning, training, and evaluation.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file (with or without '.csv' extension).
    target_column : str
        Name of the target column in the dataset.
    test_size : float, optional, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, optional, default=42
        Seed used by the random number generator.
    n_iter : int, optional, default=5
        Number of parameter settings that are sampled in RandomizedSearchCV.

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'top_5_results': List of top 5 result dictionaries.
        - 'all_results': List of all result dictionaries.
        - 'summary_dataframe': DataFrame summarizing top 5 results.
        - 'best_pipeline': Best performing sklearn Pipeline object.
        Returns None if loading fails or no valid combinations found.

    Notes
    -----
    - Supports numeric (MinMax, Standard, Robust) and categorical (OneHot, Ordinal) preprocessing.
    - Models: Logistic Regression, Random Forest, XGBoost.
    - Evaluation: Accuracy, Precision, Recall, F1-score, AUC (binary & multiclass).
    - Hyperparameter tuning with RandomizedSearchCV and 3-fold CV.
    """

    print("Loading data...")
    try:
        data = pd.read_csv(file_path if file_path.endswith('.csv') else file_path + '.csv')
    except FileNotFoundError:
        print(f"Error: File '{file_path if file_path.endswith('.csv') else file_path + '.csv'}' not found.")
        return None

    print(f"Data loaded successfully. Shape: {data.shape}")

    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in the data.")
        return None

    X = data.drop(columns=[target_column])
    y = data[target_column]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler()
    }

    encoders = {
        'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=True),
        'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    }

    models = {
        'logistic': {
            'model': LogisticRegression(random_state=random_state),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__max_iter': [1000]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=random_state),
            'params': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [2, 5]
            }
        },
        'xgboost': {
            'model': XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False),
            'params': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.01, 0.1]
            }
        }
    }

    results = []
    has_numeric_cols = len(numeric_cols) > 0
    has_categorical_cols = len(categorical_cols) > 0
    scaler_items = scalers.items() if has_numeric_cols else [('no_scaler', None)]
    encoder_items = encoders.items() if has_categorical_cols else [('no_encoder', None)]

    for scaler_name, scaler in scaler_items:
        for encoder_name, encoder in encoder_items:
            for model_name, model_info in models.items():
                try:
                    transformers = []
                    if has_numeric_cols and scaler:
                        transformers.append(('num', scaler, numeric_cols))
                    if has_categorical_cols and encoder:
                        transformers.append(('cat', encoder, categorical_cols))
                    if not transformers:
                        continue

                    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model_info['model'])
                    ])

                    random_search = RandomizedSearchCV(
                        pipeline,
                        model_info['params'],
                        n_iter=n_iter,
                        cv=3,
                        scoring='accuracy',
                        n_jobs=-1,
                        random_state=random_state
                    )
                    random_search.fit(X_train, y_train)
                    best_pipeline = random_search.best_estimator_

                    y_pred = best_pipeline.predict(X_test)
                    y_pred_proba = None
                    if hasattr(best_pipeline, 'predict_proba'):
                        try:
                            y_pred_proba = best_pipeline.predict_proba(X_test)
                        except Exception:
                            pass

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    auc = None
                    if y_pred_proba is not None:
                        if y.nunique() == 2:
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            try:
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                            except Exception:
                                auc = None

                    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=3, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()

                    result = {
                        'scaler': scaler_name if scaler else 'N/A',
                        'encoder': encoder_name if encoder else 'N/A',
                        'model': model_name,
                        'best_params': random_search.best_params_,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'pipeline': best_pipeline
                    }
                    results.append(result)

                except Exception as e:
                    print(f"  Error during pipeline for {scaler_name} + {encoder_name} + {model_name}: {str(e)}")
                    continue

    if not results:
        print("No results were generated.")
        return None

    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 5 BEST COMBINATIONS")
    print("=" * 80)

    top_5_results_display = results[:5]
    for i, result in enumerate(top_5_results_display, 1):
        print(f"\nRank {i}:")
        print(f"  Scaler: {result['scaler']}")
        print(f"  Encoder: {result['encoder']}")
        print(f"  Model: {result['model']}")
        print(f"  Best Parameters: {result['best_params']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1 Score: {result['f1_score']:.4f}")
        print(f"  AUC: {result['auc']:.4f}" if result['auc'] is not None else "  AUC: N/A")
        print(f"  CV Score: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print("-" * 60)

    summary_df = pd.DataFrame([
        {
            'Rank': i + 1,
            'Scaler': result['scaler'],
            'Encoder': result['encoder'],
            'Model': result['model'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1_score'],
            'AUC': result['auc'] if result['auc'] is not None else 'N/A',
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        }
        for i, result in enumerate(top_5_results_display)
    ])

    return {
        'top_5_results': top_5_results_display,
        'all_results': results,
        'summary_dataframe': summary_df,
        'best_pipeline': results[0]['pipeline'] if results else None
    }

if __name__ == "__main__":
    print("Starting comprehensive ML pipeline analysis with RandomizedSearchCV...")
    results_output = comprehensive_ml_pipeline('processed_data', 'class', n_iter=5)

    if results_output and not results_output['summary_dataframe'].empty:
        summary_df = results_output['summary_dataframe']
        print("\n" + "=" * 100)
        print("FINAL SUMMARY - TOP 5 COMBINATIONS")
        print("=" * 100)
        print(summary_df.to_string(index=False))

        best_result = results_output['top_5_results'][0]
        print(f"\nBest combination: {best_result['scaler']} + {best_result['encoder']} + {best_result['model']}")
        print(f"Best accuracy: {best_result['accuracy']:.4f}")
        print("\nAnalysis completed successfully!")
    elif results_output is None:
        print("Analysis failed, possibly due to file loading or other critical error.")
    else:
        print("Analysis completed, but no results were generated.")
