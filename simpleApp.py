# app.py
import mlflow
import mlflow.sklearn
from flask import Flask, request, render_template_string
import pandas as pd

# Kết nối tới MLflow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model từ Model Registry
model_name = "Best_Logistic_Model"
model_version = 1

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Tạo Flask app
app = Flask(__name__)

# Template HTML đơn giản
template = """
<!doctype html>
<title>Dự đoán Kết quả</title>
<h2>Nhập 20 đặc trưng (cách nhau bằng dấu phẩy):</h2>
<form method=post>
  <input type=text name=features size=100>
  <input type=submit value=Dự đoán>
</form>
{% if prediction %}
<h3>Kết quả dự đoán: {{ prediction }}</h3>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            features = request.form['features']
            feature_list = [float(x) for x in features.split(',')]
            df = pd.DataFrame([feature_list])

            pred = model.predict(df)[0]
            prediction = f"Kết quả dự đoán: {pred}"
        except Exception as e:
            prediction = f"Lỗi: {e}"
    return render_template_string(template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
