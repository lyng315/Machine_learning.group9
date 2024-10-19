import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV hoặc DataFrame
df = pd.read_csv("C:\\Users\\BXT\\Downloads\\Train.csv\\Train.csv")

# Hiển thị 5 dòng đầu tiên
print(df.head())

# Xác định nhãn và đặc trưng
target = 'Power'  # Nhãn là cột 'Power'
features = [i for i in df.columns if i not in [target]]  # Các đặc trưng là các cột còn lại

# Sao chép dữ liệu
original_df = df.copy(deep=True)

# In thông tin về số lượng đặc trưng và mẫu trong tập dữ liệu
print('\n\033[1mInference:\033[0m The Dataset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))



df.info()


# Giữ lại 300 mẫu đầu tiên
df = df.iloc[:300].copy()  

# In thông tin về số lượng mẫu sau khi giữ lại
print('\n\033[1mInference:\033[0m The Dataset now consists of {} samples.'.format(df.shape[0]))


print(df.nunique().sort_values())


# Loại bỏ các cột 'Location' và 'Unnamed: 0'
df_cleaned = df.drop(['Location', 'Unnamed: 0'], axis=1)

# Hiển thị thông tin sau khi loại bỏ các cột
print(df_cleaned.head())
print('\n\033[1mInference:\033[0m After removing the columns, the dataset consists of {} features & {} samples.'.format(df_cleaned.shape[1], df_cleaned.shape[0]))


print(df_cleaned.describe())


# Tạo biểu đồ phân phối cho biến mục tiêu Power
plt.figure(figsize=[8,4])
sns.distplot(df['Power'], color='g', hist_kws=dict(edgecolor="black", linewidth=2), bins=30)

# Đặt tiêu đề cho biểu đồ
plt.title('Phân phối biến mục tiêu - Công suất phát điện (MW)')

# Hiển thị biểu đồ
plt.show()




# Dưới đây là các đặc trưng số từ dự án tua bin của bạn
numeric_features = ['Temp_2m', 'RelHum_2m','WS_100m', 'WD_10m','WD_100m','WS_10m', 'DP_2m', 'WG_10m','Power'] 

# Vẽ biểu đồ 2 hàng: hàng trên là histogram và hàng dưới là boxplot
fig, axes = plt.subplots(2, len(numeric_features), figsize=(15, 6))  # Tạo lưới với 2 hàng

# Hàng đầu tiên: Biểu đồ phân phối (histogram)
for i, feature in enumerate(numeric_features):
    sns.histplot(df_cleaned[feature], bins=10, kde=True, ax=axes[0, i], color=np.random.rand(3,))  # Tạo biểu đồ phân phối
    axes[0, i].set_title(f'Phân phối của {feature}')  # Tiêu đề cho biểu đồ

# Hàng thứ hai: Biểu đồ hộp (boxplot)
for i, feature in enumerate(numeric_features):
    sns.boxplot(y=df_cleaned[feature], ax=axes[1, i], color=np.random.rand(3,))  # Tạo biểu đồ hộp
    axes[1, i].set_title(f'Boxplot của {feature}')  # Tiêu đề cho biểu đồ

# Hiển thị bố cục rõ ràng
plt.tight_layout()
plt.show()


# Lưu kích thước ban đầu của DataFrame
rs, cs = df_cleaned.shape

# Xóa các hàng trùng lặp
df_cleaned.drop_duplicates(inplace=True)

# Kiểm tra xem kích thước của DataFrame có thay đổi hay không
if df_cleaned.shape == (rs, cs):
    print('\n\033[1mInference:\033[0m Bộ dữ liệu không có bất kỳ bản sao nào')
else:
    print(f'\n\033[1mInference:\033[0m Số lượng bản sao đã loại bỏ/sửa chữa ---> {rs - df_cleaned.shape[0]}')


# Kiểm tra các phần tử null
gtn = pd.DataFrame(df_cleaned.isnull().sum().sort_values(), columns=['Tổng số giá trị Null'])
gtn['Tỷ lệ phần trăm'] = round(gtn['Tổng số giá trị Null'] / df_cleaned.shape[0], 3) * 100
print(gtn)



# Lưu dữ liệu ban đầu để so sánh sau khi loại bỏ ngoại lệ
df_before_outliers_removal = df_cleaned.copy()

# Sao chép dữ liệu ban đầu
df_outliers_removed = df_cleaned.copy()

# Hàm loại bỏ ngoại lệ sử dụng IQR
def remove_outliers_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)  # Phân vị thứ 1
    Q3 = df[feature].quantile(0.75)  # Phân vị thứ 3
    IQR = Q3 - Q1  # Khoảng tứ phân vị (IQR)
    
    # Ngưỡng trên và dưới
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Loại bỏ các giá trị ngoài ngưỡng trên và dưới
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    
    return df

# Loại bỏ ngoại lệ cho từng đặc trưng số
for feature in numeric_features:
    df_outliers_removed = remove_outliers_iqr(df_outliers_removed, feature)

# In thông tin về số lượng mẫu trước và sau khi loại bỏ ngoại lệ
print('\n\033[1mInference:\033[0m\nTrước khi loại bỏ các ngoại lệ, bộ dữ liệu có {} mẫu.'.format(df_before_outliers_removal.shape[0]))
print('Sau khi loại bỏ các ngoại lệ, tập dữ liệu hiện có {} mẫu.'.format(df_outliers_removed.shape[0]))


# Tính ma trận tương quan giữa các đặc trưng số
corr_matrix = df_cleaned[numeric_features].corr()

# Vẽ biểu đồ heatmap của ma trận tương quan
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

# Đặt tiêu đề cho biểu đồ
plt.title('Ma trận tương quan giữa các đặc trưng số', fontsize=16)

# Hiển thị biểu đồ
plt.show()


# Giữ lại bản sao của df_outliers_removed sau khi loại bỏ ngoại lai
df_final = df_outliers_removed.drop(['Time','WS_10m', 'WD_10m', 'DP_2m'], axis=1)

# In thông tin sau khi xóa cột từ bộ dữ liệu đã loại bỏ ngoại lai
print('\n\033[1mInference:\033[0m Sau khi loại bỏ các cột, bộ dữ liệu hiện có {} đặc trưng & {} mẫu.'.format(df_final.shape[1], df_final.shape[0]))

# Hiển thị 5 dòng đầu tiên của bộ dữ liệu sau khi loại bỏ cột
print(df_final.head())

scaler = StandardScaler()

# Các đặc trưng số cần chuẩn hóa
numeric_features_final = ['Temp_2m', 'RelHum_2m', 'WS_100m', 'WD_100m', 'WG_10m', 'Power']

# Áp dụng chuẩn hóa
df_final[numeric_features_final] = scaler.fit_transform(df_final[numeric_features_final])

# Hiển thị 5 dòng đầu tiên sau khi chuẩn hóa
print(df_final.head())


# Giả sử df_final là DataFrame đã chuẩn hóa của bạn
X = df_final.drop('Power', axis=1)  # Tập đặc trưng
y = df_final['Power']  # Tập nhãn

# Kiểm tra số lượng giá trị NaN trong y
print("Số lượng NaN trong y:", y.isnull().sum())

# Chia dữ liệu thành tập train (60%) và tập còn lại (40%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Chia tập còn lại thành tập validation (50% của 40%) và tập test (50% của 40%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# In thông tin về kích thước của các tập dữ liệu
print(f'Train set: {X_train.shape[0]} samples')
print(f'Validation set: {X_val.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')



# Lưu các tập dữ liệu vào file CSV
X_train.to_csv("C:\\Users\\BXT\\Downloads\\Xtrain.csv", index=False)
y_train.to_csv("C:\\Users\\BXT\\Downloads\\Ytrain.csv", index=False)

X_val.to_csv("C:\\Users\\BXT\\Downloads\\Xvalidation.csv", index=False)
y_val.to_csv("C:\\Users\\BXT\\Downloads\\Yvalidation.csv", index=False)

X_test.to_csv("C:\\Users\\BXT\\Downloads\\Xtest.csv", index=False)
y_test.to_csv("C:\\Users\\BXT\\Downloads\\Ytest.csv", index=False)

print(pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtrain.csv"))
print(pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytrain.csv"))
print(pd.read_csv("C:\\Users\\BXT\\Downloads\\Xvalidation.csv"))
print(pd.read_csv("C:\\Users\\BXT\\Downloads\\Yvalidation.csv"))
print(pd.read_csv("C:\\Users\\BXT\\Downloads\\Xtest.csv"))
print(pd.read_csv("C:\\Users\\BXT\\Downloads\\Ytest.csv"))


# Tính ma trận tương quan giữa các đặc trưng số
corr_matrix_final = df_final[numeric_features_final].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix_final, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

# Đặt tiêu đề cho biểu đồ
plt.title('Ma trận tương quan giữa các đặc trưng số', fontsize=16)

# Hiển thị biểu đồ
plt.show()