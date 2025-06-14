from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score

class ModelRunner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.models = {}
        self.results = {}
        self.predictions = {}

    def add_model(self, name, model):
        self.models[name] = model

    def train_all(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_val)
            self.results[name] = self._evaluate(y_pred)
            self.predictions[name] = self._build_prediction_df(y_pred)

    def _evaluate(self, y_pred):
        return {
            "MAE": mean_absolute_error(self.y_val, y_pred),
            "MSE": mean_squared_error(self.y_val, y_pred),
            "RMSE": root_mean_squared_error(self.y_val, y_pred),
            "R2": r2_score(self.y_val, y_pred)
        }

    def _build_prediction_df(self, y_pred):
        df = self.X_val.copy()
        df["Cognitive_Score"] = self.y_val.values
        df["Predicted_Score"] = y_pred
        return df

    def show_results(self):
        for name, metrics in self.results.items():
            print(f"\nModelo: {name}")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    def get_predictions(self, model_name):
        return self.predictions.get(model_name, None)
