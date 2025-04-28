# register_best_model.py
import mlflow
from mlflow.tracking import MlflowClient

# Kết nối MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Lấy tất cả các run trong experiment Loan_Classification
experiment = client.get_experiment_by_name("Loan_Classification")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy_manual DESC"], max_results=1)

# Chọn run có accuracy cao nhất
best_run = runs[0]
print(f"✅ Best Run ID: {best_run.info.run_id}")
print(f"✅ Accuracy: {best_run.data.metrics['accuracy_manual']}")

# Đăng ký mô hình
model_uri = f"runs:/{best_run.info.run_id}/model"
model_name = "Best_Logistic_Model"

# Đăng ký model
result = mlflow.register_model(model_uri, model_name)

print(f"✅ Model đã được đăng ký với tên: {result.name}, version: {result.version}")
