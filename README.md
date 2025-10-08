# 🩺 Medical Condition Classification

## 📘 Project Overview
This project focuses on predicting **medical conditions** based on health-related risk factors using the **Healthcare Risk Factors Dataset** from Kaggle.  
The goal is to develop a **robust and explainable machine learning pipeline** that classifies patients’ potential medical conditions by analyzing various behavioral, demographic, and health indicators.

The workflow includes **data exploration**, **feature engineering**, **model training**, **evaluation**, and **experiment tracking** using **MLflow**, with **DVC** for dataset and model version control.

---

## 🚀 Features
- Data preprocessing and exploratory data analysis (EDA)  
- Handling missing values and class imbalance  
- Model training using:
  - **XGBoost**
  - **LightGBM**
  - **Scikit-Learn** models (Logistic Regression, Random Forest, etc.)
- Performance evaluation using metrics like Accuracy, F1-score, ROC-AUC, etc.  
- Experiment tracking with **MLflow**
- Version control for data and models with **DVC**
- Visualization of key insights using **Matplotlib** and **Seaborn**

---

## 🧰 Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Experiment Tracking | `mlflow` |
| Data & Model Versioning | `dvc` |
| Environment | `Python 3.12+` |

---
## 📊 Experiment Tracking

All experiments (models, hyperparameters, metrics, artifacts) are logged in MLflow.
Use the MLflow UI to visualize model performance comparisons and parameter tuning results.
## 📈 Results

The models are evaluated on multiple metrics.
Best performing model (expected): LightGBM or XGBoost with optimized parameters.
Visualization plots include:

- Feature importance

- Confusion matrix

- ROC curves

## ⚡ Deployment
- Run FastAPI Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
- Run Streamlit interface
```bash
streamlit run main.py
```

## 🐳 Run with Docker

You can also containerize the project to ensure consistent deployment.

Build the Docker Image

```bash
docker build -t medical-condition-classification .
```