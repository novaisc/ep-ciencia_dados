from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score
import pandas as pd

class ModelRunner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.models = {}
        self.predictions_val = {}
        self.predictions_test = {}
        self.results_val = {}
        self.test_results = {}

    def add_model(self, name, model):
        self.models[name] = model

    def train_all(self):
        print("Iniciando treinamento de todos os modelos...")
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred_val = model.predict(self.X_val)
            self.predictions_val[name] = pd.DataFrame({'Real': self.y_val, 'Previsto': y_pred_val})
            self._evaluate_model(name, self.y_val, y_pred_val, self.results_val)
        print("Treinamento concluído.")

    def _evaluate_model(self, name, y_true, y_pred, results_dict):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_true, y_pred)
        results_dict[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    def show_results(self):
        print("\n=== Resultados na Validação ===")
        for name, metrics in self.results_val.items():
            print(f"\nModelo: {name}")
            print(f"MAE: {metrics['MAE']:.4f}")
            print(f"MSE: {metrics['MSE']:.4f}")
            print(f"RMSE: {metrics['RMSE']:.4f}")
            print(f"R2: {metrics['R2']:.4f}")

    def evaluate_on_test_set(self, X_test, y_test):
        self.test_results = {}
        self.test_predictions = {}

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            self.test_results[name] = self._evaluate_test(y_test, y_pred)
            
            df_pred = X_test.copy()
            df_pred['Cognitive_Score'] = y_test.values
            df_pred['Predicted_Score'] = y_pred
            self.test_predictions[name] = df_pred

    def _evaluate_test(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

    def get_test_predictions(self, model_name):
        return self.test_predictions.get(model_name, None)

    def get_all_test_results(self):
        return pd.DataFrame.from_dict(self.test_results, orient='index')
