from src.constant import (CLEANED_DATA_PATH,TRAIN_DATA_PATH,TEST_DATA_PATH,VALIDATION_DATA_PATH,TARGET_COLUMN,
                          NORMALIZER_PATH,TARGET_ENCODER_PATH,GENDER_MAP_PATH)
from src.utils import read_params
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
import joblib


def preprocess_data(df: pd.DataFrame,params:dict) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by encoding categorical variables, scaling numerical features,
    and splitting the data into training, testing, and validation sets.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
    
    """
    X=df.drop(columns=['Medical Condition'])
    y=df['Medical Condition']
    features_names = X.columns.tolist()
    categorical_features_names = [col for col in features_names if X[col].nunique() <= 25]
    numerical_features_names = list(set(features_names) - set(categorical_features_names))

    gender_map = {'Male': 0, 'Female': 1}

    X['Gender'] = X['Gender'].map(gender_map)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y)
    X_test,X_val,y_test,y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42,stratify=y_test)
    if not params["normalization"]:
        scaler = MinMaxScaler() if params['scaling']=='minmax' else StandardScaler()
        X_train[numerical_features_names] = scaler.fit_transform(X_train[numerical_features_names])
        X_test[numerical_features_names] = scaler.transform(X_test[numerical_features_names])
        X_val[numerical_features_names] = scaler.transform(X_val[numerical_features_names])
        joblib.dump(scaler, NORMALIZER_PATH)

    X_train[TARGET_COLUMN] = y_train
    X_test[TARGET_COLUMN] = y_test
    X_val[TARGET_COLUMN] = y_val

    X_train.to_csv(TRAIN_DATA_PATH, index=False)
    X_test.to_csv(TEST_DATA_PATH, index=False)
    X_val.to_csv(VALIDATION_DATA_PATH, index=False)

    joblib.dump(encoder, TARGET_ENCODER_PATH)
    joblib.dump(gender_map, GENDER_MAP_PATH)


if __name__ == "__main__":
    df = pd.read_csv(CLEANED_DATA_PATH)
    config = read_params()
    params = config["preprocessing"]
    preprocess_data(df,params=params)
    print("Preprocessing completed. Processed data saved to disk.")