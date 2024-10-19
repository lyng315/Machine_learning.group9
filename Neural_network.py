import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Đặt random state để tái tạo kết quả
np.random.seed(42)
tf.random.set_seed(42)

# Đọc dữ liệu từ các tệp CSV
X_train_full = pd.read_csv('Xtrain.csv')
X_val = pd.read_csv('Xvalidation.csv')
X_test = pd.read_csv('Xtest.csv')
y_train_full = pd.read_csv('Ytrain.csv').values.flatten()  # Chuyển y thành mảng 1 chiều
y_val = pd.read_csv('Yvalidation.csv').values.flatten()
y_test = pd.read_csv('Ytest.csv').values.flatten()

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Định nghĩa mô hình
model = Sequential([
    Dense(128, input_shape=(X_train_full.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Thiết lập EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Huấn luyện mô hình
history = model.fit(
    X_train_full, y_train_full,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'Đánh giá mô hình trên tập {label}:')
    print(f'- MSE: {mse:.4f}')
    print(f'- RMSE: {rmse:.4f}')
    print(f'- MAE: {mae:.4f}')
    print(f'- R^2: {r2:.4f}')
    return mae, mse, rmse

# Thực hiện đánh giá cho từng tập dữ liệu
train_mae, train_mse, train_rmse = evaluate_model(y_train_full, model.predict(X_train_full).flatten(), 'Train')
val_mae, val_mse, val_rmse = evaluate_model(y_val, model.predict(X_val).flatten(), 'Validation')
test_mae, test_mse, test_rmse = evaluate_model(y_test, model.predict(X_test).flatten(), 'Test')

# Vẽ biểu đồ hàm mất mát
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Vẽ biểu đồ phân tán cho các tập dữ liệu
plt.figure(figsize=(15, 5))

# Biểu đồ phân tán cho tập Train
plt.subplot(1, 3, 1)
plt.scatter(y_train_full, model.predict(X_train_full).flatten(), color='blue', alpha=0.5)
plt.plot([min(y_train_full), max(y_train_full)], [min(y_train_full), max(y_train_full)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Train Set')

# Biểu đồ phân tán cho tập Validation
plt.subplot(1, 3, 2)
plt.scatter(y_val, model.predict(X_val).flatten(), color='green', alpha=0.5)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Validation Set')

# Biểu đồ phân tán cho tập Test
plt.subplot(1, 3, 3)
plt.scatter(y_test, model.predict(X_test).flatten(), color='orange', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Test Set')

plt.tight_layout()
plt.show()

# Lưu mô hình vào file pkl
with open('neural_network.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Mô hình đã được lưu thành công với tên 'neural_network.pkl'.")
