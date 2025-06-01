# Comprehensive Machine Learning Pipeline

This project provides a fully automated, configurable machine learning pipeline that applies various combinations of preprocessing techniques and classification models to find the best-performing setup.

## Features

-  Load data from CSV
-  Automatically apply preprocessing combinations:
  - **Scalers**: MinMax, Standard, Robust
  - **Encoders**: OneHot, Ordinal
-  **Evaluate multiple models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
-  Perform hyperparameter tuning using RandomizedSearchCV
-  Evaluate performance using:
  - Accuracy, Precision, Recall, F1 Score, AUC, Cross-validation
-  Return top 5 best-performing model + preprocessing combinations
- Designed in Scikit-learn style with clear modular structure

---

## Usage

```python
from pipeline import comprehensive_ml_pipeline

result = comprehensive_ml_pipeline(
    file_path='processed_data',
    target_column='class',
    test_size=0.2,
    random_state=42,
    n_iter=5
)
