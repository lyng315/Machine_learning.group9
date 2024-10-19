import pandas as pd
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle  # Thêm thư viện pickle

# Đọc dữ liệu từ các tệp CSV đã chia
X_train = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtrain.csv").values
y_train = pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytrain.csv").values.ravel()
X_validation = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xvalidation.csv").values
y_validation = pd.read_csv("C:\\Users\\BXT\\Downloads\\Yvalidation.csv").values.ravel()
X_test = pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtest.csv").values
y_test = pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytest.csv").values.ravel()

# Huấn luyện mô hình Ridge Regression
ridge_model = Ridge(alpha=20)
ridge_model.fit(X_train, y_train)

# Hàm tính và in các chỉ số đánh giá mô hình
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Evaluation on {dataset_name}:")
    print(f"  MSE: {mse}")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R² Score: {r2}\n")
    
    return y, y_pred

# Đánh giá và vẽ biểu đồ trên các tập dữ liệu
def plot_evaluation(y_actual, y_pred, dataset_name, color):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_actual, y_pred, color=color, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')  # Đường y = x
    plt.xlabel("Giá trị thực tế")
    plt.ylabel("Giá trị dự đoán")
    plt.title(f"Biểu đồ sai số giữa giá trị thực tế và giá trị dự đoán ({dataset_name})")
    plt.grid(True)
    plt.show()

# Đánh giá mô hình và vẽ biểu đồ
y_train_actual, y_train_pred = evaluate_model(ridge_model, X_train, y_train, "Train Data")
plot_evaluation(y_train_actual, y_train_pred, "Train Data", 'blue')

y_validation_actual, y_validation_pred = evaluate_model(ridge_model, X_validation, y_validation, "Validation Data")
plot_evaluation(y_validation_actual, y_validation_pred, "Validation Data", 'green')

y_test_actual, y_test_pred = evaluate_model(ridge_model, X_test, y_test, "Test Data")
plot_evaluation(y_test_actual, y_test_pred, "Test Data", 'orange')

# Chuẩn hóa dữ liệu (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# Sử dụng SGD để huấn luyện Ridge Regression và theo dõi loss
sgd_regressor = SGDRegressor(
    penalty='l2',          # Ridge Regression tương ứng với L2 regularization
    alpha=0.01,            # Hệ số regularization
    learning_rate='constant', 
    eta0=0.01,             # Learning rate
    max_iter=1,            # Số vòng lặp mỗi lần fit (huấn luyện từng bước)
    warm_start=True,       # Không khởi tạo lại trọng số sau mỗi lần gọi fit()
    random_state=42
)

# Theo dõi loss trên cả 3 tập: train, validation, test
n_epochs = 100
train_losses = []

for epoch in range(n_epochs):
    sgd_regressor.fit(X_train_scaled, y_train)  # Huấn luyện trên tập train
    
    # Dự đoán và tính loss cho từng tập
    y_train_pred = sgd_regressor.predict(X_train_scaled)
   
    # Tính Mean Squared Error (MSE) cho mỗi tập
    train_loss = mean_squared_error(y_train, y_train_pred)
    
    # Lưu các giá trị loss
    train_losses.append(train_loss)

# Vẽ biểu đồ loss function cho cả 3 tập
plt.plot(range(1, n_epochs + 1), train_losses, label="Training Loss", color='blue')

# Cấu hình biểu đồ
plt.xlabel("Epoch")
plt.ylabel("Giá trị loss")
plt.title("Loss Function theo thời gian (SGD)")
plt.legend()
plt.grid(True)
plt.show()

# Lưu mô hình Ridge vào tệp .pkl
with open("ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_model, f)

# Lưu mô hình SGD vào tệp .pkl
with open("sgd_model.pkl", "wb") as f:
    pickle.dump(sgd_regressor, f)

print("Mô hình đã được lưu vào tệp .pkl")
