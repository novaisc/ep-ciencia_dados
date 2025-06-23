import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class PreProcess:
    def __init__(self, categorical_cols, target_col):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.ohe = OneHotEncoder(drop='first', sparse_output=False)
        self.scaler = MinMaxScaler()
        self.numerical_cols = None

    def fit(self, df: pd.DataFrame):
        # Identifica colunas numéricas (exceto a target)
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=[self.target_col], errors='ignore').columns.tolist()
        # Fit dos encoders apenas com dados de treino
        self.ohe.fit(df[self.categorical_cols])
        self.scaler.fit(df[self.numerical_cols])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Codifica categóricas
        encoded = self.ohe.transform(df[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(self.categorical_cols),
            index=df.index
        )
        df = df.drop(columns=self.categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        # Normaliza numéricas
        scaled = self.scaler.transform(df[self.numerical_cols])
        scaled_df = pd.DataFrame(scaled, columns=self.numerical_cols, index=df.index)
        df[self.numerical_cols] = scaled_df

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
