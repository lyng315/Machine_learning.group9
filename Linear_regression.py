import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Hàm để tính giá trị dự đoán
def predict(X, weights):
    return np.dot(X, weights)

# Hàm tính toán hàm mất mát (MSE)
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Hàm hồi quy tuyến tính
def linear_regression(X_train, y_train, learning_rate, n_iterations):
    # Thêm cột 1 cho hệ số bias
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    n_samples, n_features = X_train.shape
    
    # Khởi tạo trọng số
    weights = np.zeros(n_features)
    losses = []  # Danh sách để lưu hàm mất mát

    for _ in range(n_iterations):
        # Dự đoán giá trị
        y_pred = predict(X_train, weights)
        
        # Tính toán hàm mất mát
        loss = compute_loss(y_train, y_pred)
        losses.append(loss)
        
        # Tính gradient và cập nhật trọng số
        gradient = -(2/n_samples) * np.dot(X_train.T, (y_train - y_pred))
        weights -= learning_rate * gradient

    return weights, losses

# Hàm dự đoán
def predict_with_model(X, weights):
    X = np.c_[np.ones(X.shape[0]), X]  # Thêm cột 1 cho bias
    return predict(X, weights)

# Đọc dữ liệu từ file CSV
X_train = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtrain.csv")
y_train = pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytrain.csv")
X_val = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xvalidation.csv")
y_val = pd.read_csv("C:\\Users\\BXT\\Downloads\\Yvalidation.csv")
X_test = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtest.csv")
y_test = pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytest.csv")

# Chuyển đổi dữ liệu thành numpy array
X_train = X_train.values
y_train = y_train.values.flatten()
X_val = X_val.values
y_val = y_val.values.flatten()
X_test = X_test.values
y_test = y_test.values.flatten()

# Huấn luyện mô hình
weights, losses = linear_regression(X_train, y_train, learning_rate=0.015, n_iterations=80)

# Dự đoán trên các tập dữ liệu
y_train_pred = predict_with_model(X_train, weights)
y_val_pred = predict_with_model(X_val, weights)
y_test_pred = predict_with_model(X_test, weights)

# Tính toán các chỉ số đánh giá cho các tập dữ liệu
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Tính toán chỉ số đánh giá cho từng tập
metrics_train = calculate_metrics(y_train, y_train_pred)
metrics_val = calculate_metrics(y_val, y_val_pred)
metrics_test = calculate_metrics(y_test, y_test_pred)

metrics = {
    'Train': {
        'MAE': metrics_train[0],
        'MSE': metrics_train[1],
        'R-squared': metrics_train[3],
        'RMSE': metrics_train[2]
    },
    'Validation': {
        'MAE': metrics_val[0],
        'MSE': metrics_val[1],
        'R-squared': metrics_val[3],
        'RMSE': metrics_val[2]
    },
    'Test': {
        'MAE': metrics_test[0],
        'MSE': metrics_test[1],
        'R-squared': metrics_test[3],
        'RMSE': metrics_test[2]
    }
}

# Tạo DataFrame chứa các chỉ số đánh giá
metrics_df = pd.DataFrame(metrics).T
print("\nBảng các chỉ số đánh giá trên 3 tập:")
print(metrics_df)

# Lưu trọng số mô hình thành file .pkl
joblib.dump(weights, 'linear_regression_weights.pkl')
print("Trọng số mô hình đã được lưu thành công thành linear_regression_weights.pkl")

# Vẽ biểu đồ hàm mất mát
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Hàm mất mát (MSE)', color='blue')
plt.title('Quá trình giảm hàm mất mát trong quá trình huấn luyện')
plt.xlabel('Số vòng lặp (iterations)')
plt.ylabel('Mất mát (MSE)')
plt.legend()
plt.grid()
plt.show()

# Vẽ biểu đồ phân tán cho cả ba tập
plt.figure(figsize=(15, 5))

# Biểu đồ cho tập huấn luyện
plt.subplot(1, 3, 1)
plt.scatter(y_train.flatten(), y_train_pred.flatten(), color='blue', alpha=0.7)
slope, intercept = np.polyfit(y_train.flatten(), y_train_pred.flatten(), 1)
x_vals = np.linspace(y_train.min(), y_train.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Đường hồi quy')
plt.title('Tập Huấn Luyện')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.legend()

# Biểu đồ cho tập validation
plt.subplot(1, 3, 2)
plt.scatter(y_val.flatten(), y_val_pred.flatten(), color='green', alpha=0.7)
slope, intercept = np.polyfit(y_val.flatten(), y_val_pred.flatten(), 1)
x_vals = np.linspace(y_val.min(), y_val.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Đường hồi quy')
plt.title('Tập Validation')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.legend()

# Biểu đồ cho tập kiểm tra
plt.subplot(1, 3, 3)
plt.scatter(y_test.flatten(), y_test_pred.flatten(), color='orange', alpha=0.7)
slope, intercept = np.polyfit(y_test.flatten(), y_test_pred.flatten(), 1)
x_vals = np.linspace(y_test.min(), y_test.max(), 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='red', linestyle='--', label='Đường hồi quy')
plt.title('Tập Kiểm Tra')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.legend()

plt.tight_layout()
plt.show()
