import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

class PreProcess:
    def __init__(self, categorical_cols, target_col):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.ohe = OneHotEncoder(drop='first')
        self.scaler = MinMaxScaler()
        self.fitted = False

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._encode_categoricals(df)
        df = self._normalize_numericals(df)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.ohe.fit(df[self.categorical_cols])
        encoded = self.ohe.transform(df[self.categorical_cols])
        
        if hasattr(encoded, 'toarray'):
            encoded = encoded.toarray()

        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(self.categorical_cols),
            index=df.index
        )
        
        df = df.drop(columns=self.categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def _normalize_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=[self.target_col], errors='ignore').columns.tolist()
        self.scaler.fit(df[self.numerical_cols])
        scaled = self.scaler.transform(df[self.numerical_cols])
        scaled_df = pd.DataFrame(scaled, columns=self.numerical_cols, index=df.index)
        df[self.numerical_cols] = scaled_df
        return df
