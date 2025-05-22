# Se necessário, instale as bibliotecas:
# pip install scikit-learn pandas matplotlib seaborn numpy

# Importação das bibliotecas necessárias para manipulação de dados, visualização e machine learning
from sklearn.datasets import fetch_openml  # Para baixar datasets do OpenML
import pandas as pd                        # Para manipulação de DataFrames
import matplotlib.pyplot as plt            # Para visualização de gráficos
import seaborn as sns                      # Para visualização avançada
import numpy as np                         # Para operações numéricas

from sklearn.model_selection import train_test_split  # Para separar treino e teste
from sklearn.neighbors import KNeighborsClassifier    # Algoritmo KNN
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score        # Métrica de avaliação
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Escalonamento dos dados

print("Iniciando pipeline de validação cruzada com KNN...")

# 1. Carregar a base de dados do OpenML
print("Baixando o dataset de problemas ortopédicos do OpenML...")
dados = fetch_openml(data_id=1523)
print("Dataset baixado.")

# 2. Transformando os dados em DataFrame
print("Convertendo dados para DataFrame...")
tabela_dados = pd.DataFrame(data=dados['data'])
print("Primeiras linhas do DataFrame:")
print(tabela_dados.head())

# 3. Mapeando as classes para nomes legíveis
print("Mapeando as classes para nomes legíveis...")
classes = {'1':'Disk Hernia', '2':'Normal', '3':'Spondylolisthesis'}
tabela_dados['diagnostic'] = [classes[target] for target in dados.target]
print("Classes mapeadas. Exemplo:")
print(tabela_dados['diagnostic'].value_counts())

# 4. Análise exploratória dos dados
print("Informações do DataFrame:")
print(tabela_dados.info())
print("Estatísticas descritivas:")
print(tabela_dados.describe())

# 5. Removendo outlier identificado na coluna V6
print("Removendo outlier na coluna V6 (valores > 400)...")
outlier_count = tabela_dados.loc[tabela_dados['V6'] > 400].shape[0]
tabela_dados.drop(tabela_dados.loc[tabela_dados['V6'] > 400].index, inplace=True)
print(f"{outlier_count} outlier(s) removido(s).")

# 6. Separação dos dados em features (X) e target (y)
print("Separando features (X) e target (y)...")
x = tabela_dados.drop(columns=['diagnostic'])
y = tabela_dados['diagnostic']

# 7. Dividindo em treino e teste (80% treino, 20% teste)
print("Dividindo dados em treino e teste...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Formato do treino: {x_train.shape}, Formato do teste: {x_test.shape}")

# 8. Normalizando os dados (MinMaxScaler)
print("Normalizando os dados com MinMaxScaler (0-1)...")
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("Dados normalizados.")

# 9. Criando e treinando o modelo KNN
print("Criando e treinando o modelo KNN (k=3)...")
modelo_classificador = KNeighborsClassifier(n_neighbors=3)
modelo_classificador.fit(x_train_scaled, y_train)
print("Modelo treinado.")

# 10. Validação do modelo (predição no conjunto de teste)
print("Realizando predições no conjunto de teste...")
y_predito = modelo_classificador.predict(x_test_scaled)

# 11. Exibindo matriz de confusão
print("Exibindo matriz de confusão...")
matriz_confusao = confusion_matrix(y_true=y_test, y_pred=y_predito, labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])
fig = plt.figure(figsize=(8, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])
disp.plot(values_format='d')
plt.title("Matriz de Confusão - KNN")
plt.show()

# 12. Exibindo relatório de classificação
print("Relatório de classificação:")
print(classification_report(y_test, y_predito))

# 13. Validação cruzada com K-Fold
print("Executando validação cruzada (K-Fold, 5 folds)...")
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=5, shuffle=True)
result = cross_val_score(modelo_classificador, x, y, cv=kfold)
print("K-Fold (R^2) Scores:", result)
print("Média dos scores de validação cruzada:", result.mean())

# 14. Buscando o melhor valor de K para o KNN
print("Buscando o melhor valor de K para o KNN (de 1 a 14)...")
error = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_scaled, y_train)
    pred_i = knn.predict(x_test_scaled)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio para cada valor de K')
plt.xlabel('Valor de K')
plt.ylabel('Erro Médio')
plt.show()

# 15. GridSearchCV para encontrar melhores hiperparâmetros do KNN
print("Executando GridSearchCV para encontrar melhores hiperparâmetros do KNN...")
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

param_grid = {
    'n_neighbors': [8, 14],
    'weights': ['uniform', 'distance'],
    'metric': ['cosine', 'euclidean', 'manhattan']
}
gs_metric = make_scorer(accuracy_score, greater_is_better=True)
grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    scoring=gs_metric,
    cv=5,
    n_jobs=4,
    verbose=3
)
grid.fit(x_train_scaled, y_train)
knn_params = grid.best_params_
print('Melhores hiperparâmetros encontrados:', knn_params)

# 16. Resultados detalhados do GridSearchCV
print("Resultados detalhados do GridSearchCV:")
print(grid.cv_results_)

print("Pipeline de validação cruzada finalizado.")