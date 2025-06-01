Machine Learning Pipeline (Pandas / Scikit-learn Style)
Overview
This pipeline reads a CSV dataset, automatically applies multiple preprocessing scenarios, and performs training and evaluation on major machine learning models.
It includes handling imbalanced data with SMOTE, hyperparameter tuning, and cross-validation evaluation.

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

Output: DataFrame with outliers removed

log_transform_large_range(df: pd.DataFrame, feats: list, thr: float = 100) -> pd.DataFrame
Applies log transformation to features with large value ranges to stabilize distributions.

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