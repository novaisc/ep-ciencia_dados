import pandas as pd
from src.config import PATH_TO_RESULTS

def save_df_to_csv(name: str, df: pd.DataFrame, path: str = PATH_TO_RESULTS):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    - name (str): The name of the CSV file (without extension).
    - df (pd.DataFrame): The DataFrame to save.
    - path (str): The directory where the CSV file will be saved.
    """
    import os

    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, f"{name}.csv")
    df.to_csv(file_path, index=True)
    print(f"DataFrame saved to {file_path}")
