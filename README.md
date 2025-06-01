# Comprehensive Machine Learning Pipeline

This project provides a fully automated, configurable machine learning pipeline that applies various combinations of data preprocessing techniques and classification models to find the best-performing setup.

# Features

- Load data from CSV
- Apply multiple scalers and encoders:
  - **Scalers**: MinMax, Standard, Robust
  - **Encoders**: OneHot, Ordinal
- Train multiple models with hyperparameter tuning:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluate using accuracy, precision, recall, F1, AUC, and cross-validation
- Return top 5 best-performing combinations
- Scikit-learn compatible design with clear docstrings

# Function: `comprehensive_ml_pipeline`

```python
comprehensive_ml_pipeline(file_path, target_column, test_size=0.2, random_state=42, n_iter=5)
