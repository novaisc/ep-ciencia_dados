from src.dataset_handler import DatasetHandler
from src.preprocessing import PreProcess
from src.models import ModelRunner
from src.config import DATA_SET_NAME, LINES_TO_REMOVE, COLUMNS_TO_REMOVE, CATEGORICAL_COLUMNS, TARGET_COLUMN
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

def main():
    # load dataset
    dh = DatasetHandler(DATA_SET_NAME)

    # remove part of the dataset
    dh.remove_rows_by_index(list(range(LINES_TO_REMOVE)))

    # remove columns
    dh.remove_columns(COLUMNS_TO_REMOVE)

    data = dh.get_data()
    print("\nDataFrame:")
    print(data)

    # preprocess data
    pre = PreProcess(CATEGORICAL_COLUMNS, TARGET_COLUMN)
    df_preprocessed = pre.run(data)
    print("\nDataFrame Preprocessed:")
    print(df_preprocessed)

    # split into X and y
    X = df_preprocessed.drop(columns=["Cognitive_Score"])
    y = df_preprocessed["Cognitive_Score"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    runner = ModelRunner(X_train, y_train, X_val, y_val)
    runner.add_model("Regressão Linear", LinearRegression())
    runner.add_model("Árvore de Decisão", DecisionTreeRegressor(random_state=42))
    runner.train_all()
    runner.show_results()

    pred_df = runner.get_predictions("Árvore de Decisão")
    print(pred_df.head())



if __name__ == "__main__":
    main()