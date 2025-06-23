from src.dataset_handler import DatasetHandler
from src.preprocessing import PreProcess
from src.models import ModelRunner
from src.config import DATA_SET_NAME, LINES_TO_REMOVE, COLUMNS_TO_REMOVE, CATEGORICAL_COLUMNS, TARGET_COLUMN
from src.plotter import Plotter

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    print("Iniciando o processo de treinamento e avaliação dos modelos...")
    # Carrega o dataset cru
    dh = DatasetHandler(DATA_SET_NAME)
    dh.remove_rows_by_index(list(range(LINES_TO_REMOVE)))
    dh.remove_columns(COLUMNS_TO_REMOVE)
    data = dh.get_data()

    # Divide em X e y antes do pré-processamento
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Pré-processa apenas o treino com fit, os demais com transform
    print("Iniciando o pré-processamento dos dados...")
    pre = PreProcess(CATEGORICAL_COLUMNS, TARGET_COLUMN)
    X_train_proc = pre.fit_transform(X_train)
    X_val_proc = pre.transform(X_val)
    X_test_proc = pre.transform(X_test)

    # Inicializa os modelos
    print("Inicializando os modelos...")
    runner = ModelRunner(X_train_proc, y_train, X_val_proc, y_val)
    runner.add_model("Regressão Linear", LinearRegression())
    runner.add_model("Árvore de Decisão", DecisionTreeRegressor(random_state=42))
    runner.add_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42))
    runner.add_model("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    runner.add_model("KNN", KNeighborsRegressor(n_neighbors=5))
    runner.add_model("SVR", SVR(kernel='rbf'))
    runner.add_model("MLP", MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))

    # Treina e avalia na validação
    print("Treinando e avaliando os modelos...")
    runner.train_all()
    runner.show_results()

    # Avalia no conjunto de teste
    print("Avaliação final no conjunto de teste...")
    runner.evaluate_on_test_set(X_test_proc, y_test)

    # Obtém os resultados de teste para comparação
    all_test_results = runner.get_all_test_results()

    # Plota gráficos com base no conjunto de teste
    print("Gerando gráficos de avaliação...")
    plotter = Plotter()
    for model_name in runner.models.keys():
        pred_df = runner.get_test_predictions(model_name)
        plotter.plot_dispersion(pred_df, model_name)
        plotter.plot_distribution_errors(pred_df, model_name)
        plotter.plot_residuals(pred_df, model_name)
        plotter.plot_boxplot_residuals(pred_df, model_name)
        plotter.plot_distribution_real_vs_predicted(pred_df, model_name)
        plotter.plot_erro_vs_score_real(pred_df, model_name)
        plotter.plot_faixa_de_score(pred_df, model_name)

    plotter.plot_model_comparison(all_test_results, metrics_to_plot=['MAE', 'RMSE', 'R2'])
if __name__ == "__main__":
    main()