from src.constant import TEST_DATA_PATH,TARGET_COLUMN,CLASSIFIER_PATH,REPORTS_DIR,TARGET_ENCODER_PATH
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import json

def evaluate_model() -> None:
    """
    Load the test data and the trained model, make predictions, and print the classification report.
    """
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(CLASSIFIER_PATH)
    y_pred = model.predict(X_test)

    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    target_names = target_encoder.classes_.tolist()
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)   
    
    
    try:
        features_importance = model.feature_importances_
        feature_names = X_test.columns
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': features_importance})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png")
        plt.close()
    except AttributeError:
        print("The model does not have feature_importances_ attribute.")
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }, f, indent=4)
    
    

    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)
    ConfusionMatrixDisplay(cm, display_labels=target_names).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    

    plt.close()
if __name__ == "__main__":
    evaluate_model()
    print("Evaluation complete. Reports saved in the reports directory.")
