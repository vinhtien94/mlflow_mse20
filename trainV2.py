import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

# Kết nối MLflow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan_Classification_Extended")

# Bật MLflow autolog
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

# ----- 2. Logistic Regression -----
C_values = [0.01, 0.1, 1, 10, 100]
solvers = ['lbfgs', 'liblinear']

for C in C_values:
    for solver in solvers:
        with mlflow.start_run(run_name="Logistic Regression"):
            model = LogisticRegression(C=C, solver=solver, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ Logistic Regression - C={C}, solver={solver}, Accuracy={acc:.4f}")

            # Log thêm thủ công
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_metric("accuracy_manual", acc)

# ----- 3. GridSearchCV + Logistic Regression -----
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

with mlflow.start_run(run_name="GridSearch Logistic Regression"):
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ GridSearchCV - Best C={grid.best_params_['C']}, solver={grid.best_params_['solver']}, Accuracy={acc:.4f}")

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy_manual", acc)

# ----- 4. Random Forest Classifier -----
with mlflow.start_run(run_name="Random Forest"):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Random Forest - Accuracy={acc:.4f}")

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy_manual", acc)
