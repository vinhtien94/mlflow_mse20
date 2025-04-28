from sklearn.datasets import make_classification
import pandas as pd
import random

# Tạo dữ liệu mô phỏng
X, y = make_classification(
    n_samples=1000,        # Số lượng mẫu
    n_features=12,         # Số lượng đặc trưng
    n_informative=8,       # Số đặc trưng quan trọng
    n_redundant=2,         # Số đặc trưng dư thừa
    n_classes=2,           # Số lớp phân loại (0: Không trả nợ, 1: Có khả năng trả nợ)
    random_state=42        # Để kết quả tái lập
)

# Danh sách các tỉnh/thành phố ở Việt Nam
vietnam_cities = ['Hà Nội', 'Hồ Chí Minh', 'Huế', 'Đà Nẵng', ' Cần Thơ',
                  'Hải Phòng','Lai Châu','Điện Biên','Sơn La','Lạng Sơn','Quảng Ninh',
                  'Thanh Hoá','Nghệ An','Hà Tĩnh',
                  'Cao Bằng','Tuyên Quang','Lào Cai','Thái Nguyên','Phú Thọ','Bắc Ninh','Hưng Yên', 'Ninh Bình','Quảng Trị','Quảng Ngãi',
                  'Gia Lai','Khánh Hòa','Lâm Đồng','Đắk Lắk','Đồng Nai','Tây Ninh','Vĩnh Long','Đồng Tháp','Cà Mau','An Giang'
                ]


# Tùy chỉnh các cột đặc trưng
columns = [
    "age",                # Độ tuổi
    "gender",             # Giới tính
    "birth_year",         # Năm sinh
    "marital_status",     # Tình trạng hôn nhân
    "hometown",           # Quê quán
    "credit_score",       # Điểm tín dụng
    "income",             # Thu nhập
    "loan_amount",        # Số tiền vay
    "education_level",    # Trình độ học vấn
    "debt_to_income",     # Tỷ lệ nợ trên thu nhập
    "savings",            # Tiền tiết kiệm
    "loan_duration"       # Thời gian vay
]

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(X, columns=columns[:len(X[0])])
df['target'] = y  # Thêm cột mục tiêu (0 hoặc 1)

# Điều chỉnh dữ liệu thành hợp lý hơn:
# Độ tuổi: 20 - 60
df['age'] = (df['age'] * 10 + 40).clip(20, 60).astype(int)

# Giới tính: Nam/Nữ
df['gender'] = df['gender'].apply(lambda x: "Male" if x > 0 else "Female")

# Năm sinh: Dựa trên độ tuổi
df['birth_year'] = 2025 - df['age']

# Tình trạng hôn nhân: Độc thân/Kết hôn
df['marital_status'] = df['marital_status'].apply(lambda x: "Married" if x > 0 else "Single")

# Quê quán: Chọn ngẫu nhiên từ danh sách tỉnh/thành phố Việt Nam
df['hometown'] = [random.choice(vietnam_cities) for _ in range(len(df))]

# Điểm tín dụng: 300 - 850 (giá trị phổ biến trong hệ thống điểm tín dụng)
df['credit_score'] = (df['credit_score'] * 100 + 575).clip(300, 850).astype(int)

# Thu nhập: 5 triệu - 100 triệu VND
df['income'] = (df['income'] * 10_000_000 + 50_000_000).clip(5_000_000, 100_000_000).astype(int)

# Số tiền vay: 100 triệu - 500 triệu VND
df['loan_amount'] = (df['loan_amount'] * 50_000_000 + 100_000_000).clip(100_000_000, 500_000_000).astype(int)

# Trình độ học vấn: 1 - 5 (1: Trung học, 2: Cao đẳng, 3: Đại học, 4: Thạc sĩ, 5: Tiến sĩ)
df['education_level'] = df['education_level'].apply(lambda x: random.randint(1, 5))

# Tỷ lệ nợ trên thu nhập: 0.1 - 0.9
df['debt_to_income'] = (df['debt_to_income'] * 0.2 + 0.5).clip(0.1, 0.9).round(2)

# Tiền tiết kiệm: 0 - 200 triệu VND
df['savings'] = (df['savings'] * 50_000_000 + 100_000_000).clip(0, 200_000_000).astype(int)

# Thời gian vay: 6 - 60 tháng
df['loan_duration'] = (df['loan_duration'] * 10 + 36).clip(6, 60).astype(int)

# Lưu dữ liệu vào file CSV (nếu cần)
df.to_csv("data/loan_prediction_data.csv", index=False)

# Hiển thị một vài dòng dữ liệu để kiểm tra
print(df.head())