import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

# Kết nối MLflow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan_Classification")

# Bật autolog
mlflow.sklearn.autolog()

# Đọc dữ liệu
data = pd.read_csv('data/loan_prediction_data.csv')

# Encode cột object
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

X = data.drop('target', axis=1)
y = data['target']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter search
C_values = [0.01, 0.1, 1, 10, 100]
solvers = ['lbfgs', 'liblinear']

for C in C_values:
    for solver in solvers:
        with mlflow.start_run():
            model = LogisticRegression(C=C, solver=solver, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ C={C}, solver={solver}, Accuracy={acc:.4f}")

            # Log thêm thủ công nếu muốn
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_metric("accuracy_manual", acc)
