# Makefile for MLOps_MSE_HCM_SP25 Project (with best_model.py)

# Directories and files
DATA=./data/loan_prediction_data.csv
APP_DIR=.
MLRUNS_DIR=./mlruns

# Targets
.PHONY: all train register app mlflow-server clean reset

all: run

run: train register app

train:
	@echo "==> Training Logistic Regression models and logging with MLflow..."
	python train.py

register:
	@echo "==> Registering the best model into MLflow Model Registry..."
	python best_model.py

app:
	@echo "==> Starting Flask Web App at http://127.0.0.1:5001/"
	python app.py

mlflow-server:
	@echo "==> Starting MLflow Tracking Server at http://127.0.0.1:5000/"
	mlflow server --backend-store-uri "file:///$(CURDIR)/mlruns" --default-artifact-root "file:///$(CURDIR)/mlruns" --host 127.0.0.1 --port 5000
	@echo "==> MLflow server started."
clean:
	@echo "==> Cleaning temporary files..."
	del /s /q __pycache__ >nul 2>&1 || echo "No __pycache__ found."

reset:
	@echo "==> Resetting project: deleting all MLflow runs and artifacts..."
	rmdir /s /q $(MLRUNS_DIR)
	del /s /q __pycache__ >nul 2>&1 || echo "No __pycache__ found."
