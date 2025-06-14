# EP de Ciência de dados - Análise de Dados Cognitivos

Este projeto realiza a análise de dados de desempenho cognitivo humano, utilizando técnicas de pré-processamento, manipulação de datasets e modelos de aprendizado de máquina para prever a pontuação cognitiva com base em diferentes variáveis.

## Estrutura do Projeto

A estrutura do projeto é organizada da seguinte forma:
```
src/
├── config/           # Configurações e constantes do projeto
├── data/             # Dados utilizados no projeto
├── dataset_handler/  # Manipulação do dataset
├── models/           # Execução e avaliação de modelos
├── preprocessing/    # Pré-processamento dos dados
```
## Dependências

As dependências do projeto estão listadas no arquivo `pyproject.toml`. Para instalá-las, utilize o [Poetry](https://python-poetry.org/):

```bash
poetry install
```

## Como Executar

Certifique-se de que o dataset `human_cognitive_performance.csv` está localizado no diretório `src/data/`.

Execute o script principal `app.py`:

```bash
python app.py
```

## Funcionalidades

- **Manipulação de Dataset**: Remoção de linhas, colunas e subconjuntos aleatórios.
- **Pré-processamento**: Codificação de variáveis categóricas e normalização de variáveis numéricas.
- **Treinamento de Modelos**: Suporte para Regressão Linear e Árvore de Decisão.
- **Avaliação de Modelos**: Métricas como MAE, MSE, RMSE e R².

## Configurações

As configurações do projeto estão definidas no arquivo `src/config/constants.py`. Você pode ajustar os seguintes parâmetros:

- `DATA_DIR`: Diretório onde o dataset está localizado.
- `DATA_SET_NAME`: Nome do arquivo do dataset (sem extensão).
- `LINES_TO_REMOVE`: Número de linhas a serem removidas do início do dataset.
- `COLUMNS_TO_REMOVE`: Lista de colunas a serem removidas.
- `CATEGORICAL_COLUMNS`: Colunas categóricas para codificação.
- `TARGET_COLUMN`: Coluna alvo para predição.

## Exemplo de Saída

Ao executar o projeto, você verá:

- Informações sobre o dataset carregado.
- Dataset pré-processado.
- Divisão dos dados em treino, validação e teste.
- Resultados das métricas de avaliação para cada modelo.
- Predições realizadas pelo modelo selecionado.
