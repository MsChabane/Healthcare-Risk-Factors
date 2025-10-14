from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
PARAMS_PATH = ROOT_DIR / "params.yaml"


DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH= DATA_DIR / "raw"/"dirty_v3_path.csv"
CLEANED_DATA_PATH = CLEANED_DATA_DIR / "cleaned.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"

TARGET_COLUMN = "Medical Condition"
NORMALIZER_PATH = MODELS_DIR / "normaliser.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"
GENDER_MAP_PATH = MODELS_DIR / "gender_map.pkl"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl" 




