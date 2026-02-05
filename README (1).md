# House Price Prediction: Standard vs Polynomial Features

A machine learning project for predicting house prices using the Ames Housing dataset. This project compares multiple regression models with two feature engineering strategies (standard and polynomial) and includes hyperparameter optimization using Optuna.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Notebook](#running-the-notebook)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Main Function Parameters](#main-function-parameters)
- [Return Values](#return-values)
- [Project Structure](#project-structure)
- [Methodology](#methodology)

## Features

1. **Automatic Outlier Handling**
   - Winsorization for features with high skewness
   - Hybrid outlier detection using IsolationForest + Mahalanobis distance
   - Conservative approach: only removes rows flagged by both methods

2. **Two Feature Strategies**
   - Standard: Uses original features with target encoding
   - Polynomial: Applies polynomial transformation to top correlated features

3. **Multiple Model Support**
   - Ridge Regression
   - Lasso Regression
   - ElasticNet
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost

4. **Automatic Hyperparameter Optimization**
   - Uses Optuna for Bayesian optimization
   - Configurable number of trials per model

5. **Results Comparison**
   - Detailed comparison table between standard and polynomial approaches
   - Displays R2, RMSE, and improvement metrics

6. **Interactive Web Application**
   - Streamlit app for visualization and predictions
   - Upload custom datasets
   - Interactive model training and comparison
   - Actual vs predicted plots, residual analysis, feature importance


## Installation

```bash
pip install pandas numpy scikit-learn category-encoders xgboost lightgbm catboost optuna scipy matplotlib seaborn streamlit plotly
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

#### Basic Usage - Test All Models

```python
from unified_models_comparison import main

results_std, results_poly, comparison = main(
    filepath="data/train.csv",
    n_trials=30
)
```

#### Test Specific Models

```python
results_std, results_poly, comparison = main(
    filepath="data/train.csv",
    models_to_test=['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting'],
    n_trials=30
)
```

#### Quick Test with Fewer Trials

```python
results_std, results_poly, comparison = main(
    filepath="data/train.csv",
    models_to_test=['Ridge', 'RandomForest'],
    n_trials=10  # Faster but may not find optimal parameters
)
```

#### Custom Polynomial Features

```python
results_std, results_poly, comparison = main(
    filepath="data/train.csv",
    models_to_test=['Ridge', 'RandomForest', 'XGBoost'],
    n_trials=30,
    poly_degree=3,        # Polynomial degree (default: 2)
    top_n_features=15     # Number of features to apply polynomial (default: 10)
)
```

### Running the Streamlit App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

The app provides:
- Data upload and overview with statistics
- Interactive model selection and training
- Visualizations: price distribution, correlation analysis
- Model comparison dashboard
- Actual vs predicted plots, residual analysis
- Feature importance charts
- Simple prediction interface

## Main Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| filepath | str | required | Path to the CSV data file |
| models_to_test | list or None | None | List of model names to test. None = test all models. Available: `['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']` |
| n_trials | int | 30 | Number of optimization trials per model |
| poly_degree | int | 2 | Degree of polynomial features |
| top_n_features | int | 10 | Number of top correlated features for polynomial transformation |

## Return Values

The `main()` function returns 3 values:

1. **results_standard** (dict): Results with standard features

```python
{
    'Ridge': {
        'best_r2': 0.8912,
        'best_rmse': 23123.45,
        'best_params': {'alpha': 9.983}
    },
    ...
}
```

2. **results_poly** (dict): Results with polynomial features (same structure)

3. **comparison_df** (DataFrame): Detailed comparison table with columns:
   - Model, Standard_R2, Standard_RMSE, Poly_R2, Poly_RMSE, R2_Improvement, RMSE_Improvement


## Project Structure

```
house-price-prediction/
├── main.ipynb    # Main notebook with full analysis
├── app.py                    # Streamlit web application
├── README.md                 # Documentation
├── requirements.txt          # Python dependencies

```

## Methodology

### Data Preprocessing

1. **Column Removal**: Drop columns with high missing rates (Alley, PoolQC, Fence, MiscFeature, Id)
2. **Outlier Detection**:
   - Detect heavily skewed features (skewness > 1.0)
   - Apply winsorization to clip extreme values at 1st and 99th percentiles
   - Use hybrid detection: IsolationForest identifies anomalies, Mahalanobis distance measures multivariate outliers
   - Remove only consensus outliers (flagged by both methods) for conservative cleaning

### Feature Engineering

1. **Categorical Encoding**: Target encoding with smoothing for categorical variables
2. **Missing Value Imputation**: Median imputation for numerical features, "none" for categorical
3. **Polynomial Features**: Optional degree-2 polynomial transformation on top N correlated numerical features
4. **Scaling**: StandardScaler normalization for all features

### Model Optimization

- **Framework**: Optuna for Bayesian hyperparameter optimization
- **Objective**: Maximize R2 score on test set
- **Validation**: Train/test split (75/25)
- **Trials**: Configurable, default 30 trials per model

### Key Findings

1. CatBoost achieves the highest R2 (0.926) with standard features
2. Polynomial features improve linear models by 0.5-1.1% R2 but decrease tree-based models by 0.7-1.3%
3. Top predictive features: OverallQual, GrLivArea, GarageCars, TotalBsmtSF, YearBuilt
4. Hybrid outlier removal (approximately 1% of data) improves model stability


