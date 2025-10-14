from src.constant import TEST_DATA_PATH,TARGET_COLUMN,CLASSIFIER_PATH,REPORTS_DIR,TARGET_ENCODER_PATH,MLFLOW_TRACKING_URI
from src.utils import read_params
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


import joblib
import json

def evaluate_model(config: dict) -> None:
    """
    Load the test data and the trained model, make predictions, and print the classification report.
    """
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(CLASSIFIER_PATH)
    best_iter = model.best_iteration
    
    
    
    y_pred = model.predict(X_test)

    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    target_names = target_encoder.classes_.tolist()
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)   

    features_importance = None
    try:
        features_importance = model.feature_importances_
        feature_names = X_test.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': features_importance})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        FEATURES_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
        FEATURES_IMPORTANCE_PNG = REPORTS_DIR / "feature_importance.png"
        fi_df.to_csv(FEATURES_IMPORTANCE_CSV, index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(FEATURES_IMPORTANCE_PNG)
        plt.close()
    except AttributeError:
        print("The model does not have feature_importances_ attribute.")
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    METRICS_PATH = REPORTS_DIR / "metrics.json"
    CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
    CM_PATH = REPORTS_DIR / "confusion_matrix.png"
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(report)
    ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PATH)
    

    ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()
    with mlflow.start_run():
       
        mlflow.log_params(config["model"]["params"])
        mlflow.log_params(config["preprocessing"])

        mlflow.log_param("best_iteration", best_iter)
        mlflow.log_metrics(metrics)
        if features_importance is not None:
            mlflow.log_artifact(FEATURES_IMPORTANCE_CSV)
            mlflow.log_artifact(FEATURES_IMPORTANCE_PNG)
        mlflow.log_artifact(CLASSIFICATION_REPORT_PATH)
        mlflow.log_artifact(CM_PATH)
        


if __name__ == "__main__":
    config = read_params()
    params =config["eval"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(params["mlflow_experiment_name"])
    evaluate_model(config=config)
    print("Evaluation complete. Reports saved in the reports directory.")
