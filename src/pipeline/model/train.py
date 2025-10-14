from src.constant import TRAIN_DATA_PATH,VALIDATION_DATA_PATH,TARGET_COLUMN,CLASSIFIER_PATH
from src.utils import read_params
from xgboost import XGBClassifier
import pandas as pd
import joblib


def train_model(X_train: pd.DataFrame, y_train: pd.Series,X_val: pd.DataFrame, y_val: pd.Series, model_params: dict) -> XGBClassifier:
    """
    Train an XGBoost classifier on the training data and validate it on the validation data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        model_params (dict): Parameters for the XGBoost classifier.

    Returns:
        XGBClassifier: Trained XGBoost classifier.
    """

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],  verbose=True)
    joblib.dump(model, CLASSIFIER_PATH)
    return model

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VALIDATION_DATA_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]
    config = read_params()
    params = config["model"]

    model = train_model(X_train, y_train, X_val, y_val, model_params=params['params'])
    print("Model training completed and saved to disk.")
