import pandas as pd
from src.config import DATA_DIR

class DatasetHandler:
    def __init__(self, file_name: str):
        self.filepath = file_name
        self.original_data = pd.read_csv(f"{DATA_DIR}/{file_name}.csv")
        self.data = self.original_data.copy()
        print(f"Dataset carregado com {self.data.shape[0]} linhas e {self.data.shape[1]} colunas.")
    
    def remove_random_subset(self, fraction=0.1, random_state=42):
        removed = self.data.sample(frac=fraction, random_state=random_state)
        self.data = self.data.drop(removed.index).reset_index(drop=True)
        print(f"{len(removed)} registros removidos. Restaram {len(self.data)} registros.")
        return self.data

    def remove_columns(self, columns):
        before = self.data.shape[1]
        self.data.drop(columns=columns, inplace=True, errors='ignore')
        after = self.data.shape[1]
        print(f"{before - after} colunas removidas.")
        return self.data
    
    def remove_rows_by_index(self, index_list):
        before = self.data.shape[0]
        self.data.drop(index=index_list, inplace=True, errors='ignore')
        self.data.reset_index(drop=True, inplace=True)
        after = self.data.shape[0]
        print(f"{before - after} linhas removidas.")
        return self.data

    def get_data(self):
        return self.data
    
    def reset_data(self):
        self.data = self.original_data.copy()
        print("Dataset restaurado ao estado original.")
        return self.data
