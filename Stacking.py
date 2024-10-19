import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
import tensorflow as tf
from tensorflow.keras.models import load_model

# Đọc dữ liệu từ các tệp CSV đã chia
X_train = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtrain.csv").values
y_train = pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytrain.csv").values.ravel()
X_validation = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xvalidation.csv").values
y_validation = pd.read_csv("C:\\Users\\BXT\\Downloads\\Yvalidation.csv").values.ravel()
X_test = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtest.csv").values
y_test = pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytest.csv").values.ravel()

# Tải các mô hình đã lưu
with open('C:\\Users\\BXT\\Downloads\\ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('C:\\Users\\BXT\\Downloads\\linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

# Tải mô hình Neural Network
neural_model = load_model('C:\\Users\\BXT\\Downloads\\neural_network.h5')

# Định nghĩa mô hình stacking
estimators = [
    ('ridge', ridge_model),
    ('linear', linear_model)
]

stacked_model = StackingRegressor(estimators=estimators, final_estimator=neural_model)

# Huấn luyện mô hình stacking
stacked_model.fit(X_train, y_train)

# Hàm đánh giá mô hình
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Đánh giá mô hình trên tập {dataset_name}:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- MAE: {mae:.4f}")
    print(f"- R²: {r2:.4f}\n")

# Đánh giá mô hình trên các tập dữ liệu
evaluate_model(stacked_model, X_train, y_train, "Train")
evaluate_model(stacked_model, X_validation, y_validation, "Validation")
evaluate_model(stacked_model, X_test, y_test, "Test")
