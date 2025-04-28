# app.py
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template
import pandas as pd
import datetime


# Kết nối tới MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model từ Model Registry
model_name = "Best_Logistic_Model"
model_version = 1

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

current_year = datetime.datetime.now().year
birth_year_options = list(range(1990, current_year - 18 + 1))

current_year = datetime.datetime.now().year
birth_year_options = list(range(1990, current_year - 18 + 1))


# Tạo Flask app
app = Flask(__name__)

# Các tên feature giả định (12 features)
feature_names = [
    'age',
    'gender',
    'birth_year',
    'marital_status',
    'hometown',
    'credit_score',
    'income',
    'loan_amount',
    'education_level',
    'debt_to_income',
    'savings',
    'loan_duration'
]

# ✨ Thêm đoạn này vào nha:
feature_labels = {
    'age': 'Tuổi',
    'gender': 'Giới tính',
    'birth_year': 'Năm sinh',
    'marital_status': 'Tình trạng hôn nhân',
    'hometown': 'Tỉnh thành',
    'credit_score': 'Điểm tín dụng',
    'income': 'Thu nhập',
    'loan_amount': 'Khoản vay',
    'education_level': 'Trình độ học vấn',
    'debt_to_income': 'Tỉ lệ nợ/thu nhập',
    'savings': 'Tiền tiết kiệm',
    'loan_duration': 'Thời hạn vay'
}


@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            features = []
            for feature in feature_names:
                value = float(request.form[feature])
                features.append(value)
            df = pd.DataFrame([features], columns=feature_names)

            pred = model.predict(df)[0]
            prediction = f"{pred}"
        except Exception as e:
            prediction = f"Lỗi: {e}"
    return render_template("index.html", prediction=prediction, feature_names=feature_names, feature_labels=feature_labels,birth_year_options=birth_year_options)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
