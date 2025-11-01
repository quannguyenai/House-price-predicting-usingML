# Project 5.1 - Predict house pricing: Standard vs Polynomial Features


## Tính năng chính

1. **Xử lý outliers tự động**:
   - Winsorization cho các features có skewness cao
   - Phát hiện outliers bằng phương pháp hybrid (IsolationForest + Mahalanobis distance)

2. **Hai chiến lược features**:
   - **Standard**: Sử dụng features gốc
   - **Polynomial**: Áp dụng polynomial features cho top features có correlation cao nhất

3. **Hỗ trợ nhiều mô hình**:
   - Ridge Regression
   - Lasso Regression
   - ElasticNet
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost

4. **Tối ưu hóa tự động**:
   - Sử dụng Optuna để tự động tìm hyperparameters tốt nhất
   - Tùy chỉnh số lượng trials

5. **So sánh kết quả**:
   - Bảng so sánh chi tiết giữa standard và polynomial
   - Hiển thị R², RMSE và mức cải thiện

## Cài đặt

```bash
pip install pandas numpy scikit-learn category-encoders xgboost lightgbm catboost optuna scipy
```

## Cách sử dụng

### 1. Sử dụng cơ bản - Test tất cả mô hình

```python
from unified_models_comparison import main

results_std, results_poly, comparison = main(
    filepath="train-house-prices-advanced-regression-techniques.csv",
    n_trials=30
)
```

### 2. Test một số mô hình cụ thể

```python
results_std, results_poly, comparison = main(
    filepath="train-house-prices-advanced-regression-techniques.csv",
    models_to_test=['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting'],
    n_trials=30
)
```

### 3. Test nhanh với ít trials hơn

```python
results_std, results_poly, comparison = main(
    filepath="train-house-prices-advanced-regression-techniques.csv",
    models_to_test=['Ridge', 'RandomForest'],
    n_trials=10  # Nhanh hơn nhưng có thể không tối ưu
)
```

### 4. Tùy chỉnh polynomial features

```python
results_std, results_poly, comparison = main(
    filepath="train-house-prices-advanced-regression-techniques.csv",
    models_to_test=['Ridge', 'RandomForest', 'XGBoost'],
    n_trials=30,
    poly_degree=3,        # Degree của polynomial (mặc định: 2)
    top_n_features=15     # Số features để áp dụng polynomial (mặc định: 10)
)
```


## Tham số của hàm main()

- `filepath` (str): Đường dẫn đến file CSV
- `models_to_test` (list hoặc None): 
  - Danh sách tên mô hình muốn test
  - None = test tất cả mô hình
  - Các mô hình có sẵn: `['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']`
- `n_trials` (int, mặc định=30): Số lần tối ưu hóa cho mỗi mô hình
- `poly_degree` (int, mặc định=2): Bậc của polynomial features
- `top_n_features` (int, mặc định=10): Số features có correlation cao nhất để áp dụng polynomial

## Kết quả trả về

Hàm `main()` trả về 3 giá trị:

1. **results_standard** (dict): Kết quả với standard features
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

2. **results_poly** (dict): Kết quả với polynomial features
3. **comparison_df** (DataFrame): Bảng so sánh chi tiết

