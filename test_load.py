import mlflow
import mlflow.sklearn

# Kết nối tới MLflow tracking server đang chạy
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Dùng đúng run ID và artifact path
model_uri = "runs:/7052e2a5f5514aed936879e7f84b2445/Decision_Tree_model"

try:
    model = mlflow.sklearn.load_model(model_uri)
    print("✅ MLflow model loaded successfully!")
except Exception as e:
    print("❌ Error loading MLflow model:", e)
