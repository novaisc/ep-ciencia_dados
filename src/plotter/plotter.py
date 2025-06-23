from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import PATH_TO_PLOTS
from src.utils import create_directory

class Plotter:
    def __init__(self):
        pass

    def plot_dispersion(self, df: pd.DataFrame, model_name: str):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="Cognitive_Score", y="Predicted_Score", data=df, alpha=0.5)
        plt.plot([0, 100], [0, 100], color='red', linestyle='--', label='Ideal')
        plt.xlabel("Cognitive Score (real)")
        plt.ylabel("Predicted Score")
        plt.title(f"{model_name}: Real vs. Predito")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/dispersion.png")
        plt.close()

    def plot_distribution_errors(self, df: pd.DataFrame, model_name: str):
        df["Erro Absoluto"] = abs(df["Predicted_Score"] - df["Cognitive_Score"])
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Erro Absoluto"], bins=30, kde=True)
        plt.title(f"{model_name}: Distribuição dos Erros Absolutos")
        plt.xlabel("Erro Absoluto")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/error_distribution.png")
        plt.close()

    def plot_residuals(self, df: pd.DataFrame, model_name: str):
        residuals = df["Cognitive_Score"] - df["Predicted_Score"]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["Predicted_Score"], y=residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Score")
        plt.ylabel("Resíduo (Real - Predito)")
        plt.title(f"{model_name}: Gráfico de Resíduos")
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/residuos.png")
        plt.close()

    def plot_boxplot_residuals(self, df: pd.DataFrame, model_name: str):
        residuals = df["Cognitive_Score"] - df["Predicted_Score"]
        plt.figure(figsize=(6, 5))
        sns.boxplot(y=residuals)
        plt.title(f"{model_name}: Boxplot dos Resíduos")
        plt.ylabel("Resíduo")
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/boxplot_residuos.png")
        plt.close()

    def plot_distribution_real_vs_predicted(self, df: pd.DataFrame, model_name: str):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(df["Cognitive_Score"], label="Real", fill=True)
        sns.kdeplot(df["Predicted_Score"], label="Predito", fill=True)
        plt.title(f"{model_name}: Distribuição Real vs. Predita")
        plt.xlabel("Score")
        plt.ylabel("Densidade")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/distribuicao_real_vs_predito.png")
        plt.close()

    def plot_erro_vs_score_real(self, df: pd.DataFrame, model_name: str):
        df["Erro Absoluto"] = abs(df["Predicted_Score"] - df["Cognitive_Score"])
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["Cognitive_Score"], y=df["Erro Absoluto"], alpha=0.5)
        plt.title(f"{model_name}: Erro Absoluto vs. Score Real")
        plt.xlabel("Cognitive Score (Real)")
        plt.ylabel("Erro Absoluto")
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/erro_vs_score_real.png")
        plt.close()

    def plot_faixa_de_score(self, df: pd.DataFrame, model_name: str):
        bins = [0, 50, 70, 100]
        labels = ["Baixa", "Média", "Alta"]
        df["Classe Real"] = pd.cut(df["Cognitive_Score"], bins=bins, labels=labels)
        df["Classe Predita"] = pd.cut(df["Predicted_Score"], bins=bins, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.countplot(x="Classe Real", data=df, alpha=0.5, label="Real")
        sns.countplot(x="Classe Predita", data=df, alpha=0.5, label="Predito", color="orange")
        plt.title(f"{model_name}: Frequência por Faixa de Score")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        create_directory(f"{PATH_TO_PLOTS}/{model_name}")
        plt.savefig(f"{PATH_TO_PLOTS}/{model_name}/frequencia_faixa_score.png")
        plt.close()

    def plot_model_comparison(self, results_df: pd.DataFrame, metrics_to_plot=['MAE', 'RMSE', 'R2']):
        """
        Plota gráfico de barras comparando os modelos para as métricas selecionadas.

        Args:
            results_df (pd.DataFrame): DataFrame com índice dos nomes dos modelos e colunas com métricas.
            metrics_to_plot (list): Lista com as métricas que quer comparar (ex: ['MAE', 'RMSE', 'R2']).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6), squeeze=False)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[0][i]
            sns.barplot(x=results_df.index, y=results_df[metric], ax=ax, palette='viridis', hue=results_df.index, legend=False)
            ax.set_title(f'Comparação de Modelos por {metric}')
            ax.set_xlabel('Modelo')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Adiciona valor numérico em cima das barras
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f')

        plt.tight_layout()
        plt.suptitle('Comparação de Desempenho dos Modelos no Conjunto de Teste', y=1.05, fontsize=16)
        plt.savefig(f"{PATH_TO_PLOTS}/frequencia_faixa_score.png")
        plt.close()
