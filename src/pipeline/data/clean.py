from src.constant import CLEANED_DATA_PATH, RAW_DATA_PATH
import pandas as pd

def clean_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(["noise_col","random_notes"],axis=1,inplace=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    

if __name__ == "__main__":
    clean_data()
    