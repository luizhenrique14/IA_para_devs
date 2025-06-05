# Se necessário, instale as bibliotecas:
# pip install pandas scikit-learn matplotlib seaborn numpy

# 1. Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados tabulares
from sklearn.model_selection import train_test_split  # Separação dos dados em treino e teste
from sklearn.neighbors import KNeighborsClassifier  # Algoritmo KNN para classificação
from sklearn.metrics import accuracy_score  # Métrica de avaliação de acurácia
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matriz de confusão
from sklearn.metrics import classification_report  # Relatório de métricas de classificação
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Escalonamento dos dados
import matplotlib.pyplot as plt  # Visualização de gráficos
import seaborn as sns  # Visualização avançada de gráficos
import numpy as np  # Operações numéricas
import warnings  # Controle de avisos

print("Iniciando pipeline de detecção de fraude em cartão de crédito...")

# 2. Carregando a base de dados
print("Lendo o arquivo de dados de transações de cartão de crédito...")
dados = pd.read_csv('card_transdata.csv', sep=',')
print("Primeiras linhas dos dados:")
print(dados.head(3))

# 3. Análise inicial dos dados
print("Formato do dataset (linhas, colunas):", dados.shape)
print("Verificando valores nulos por coluna:")
print(dados.isnull().sum())

# 4. Removendo linhas com valores nulos, se houver
if dados.isnull().sum().sum() > 0:
    print("Removendo linhas com valores nulos...")
    dados = dados.dropna()
else:
    print("Nenhum valor nulo encontrado.")

# 5. Análise exploratória dos dados
print("Estatísticas descritivas das variáveis numéricas:")
print(dados.describe())

# 6. Análise da proporção de fraudes
Total = len(dados)
TotalNaoFraudes = dados[dados["fraud"] == 0].fraud.count()
TotalFraudes = dados[dados["fraud"] == 1].fraud.count()
Percentual_Fraudes = TotalFraudes / Total
print("Total de dados:", Total)
print("Total de não fraudes:", TotalNaoFraudes)
print("Total de fraudes:", TotalFraudes)
print("Percentual de fraudes na base: {:.2f}%".format(Percentual_Fraudes * 100))

# 7. Visualização da proporção de fraudes
categororias = ["Non-Fraud", "Fraud"]
plt.pie(dados["fraud"].value_counts(), labels=categororias, autopct="%.0f%%", explode=(0, 0.1), colors=("g", "r"))
plt.title("Distribuição de Fraudes na Base")
plt.show()

# 8. Análise exploratória de variáveis numéricas em transações fraudulentas
dados_fraudes = dados[dados["fraud"] == 1]
Colunas_Numericas = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"]
for column in Colunas_Numericas:
    plt.figure()
    sns.histplot(dados_fraudes[column], bins=10, kde=True)
    plt.title(f'Distribuição de {column} em fraudes')
    plt.show()

# 9. Correlação entre variáveis
print("Matriz de correlação entre variáveis:")
correlation_matrix = dados.corr().round(2)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.title("Matriz de Correlação")
plt.show()

# 10. Separação das features (X) e target (y)
print("Selecionando features e variável alvo...")
x = dados[['distance_from_home', 'ratio_to_median_purchase_price', 'online_order']]
y = dados['fraud']

# 11. Separação em treino e teste
print("Dividindo dados em treino (80%) e teste (20%)...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=7
)
print(f"Formato do treino: {x_train.shape}, Formato do teste: {x_test.shape}")

# 12. Escalonamento das variáveis (normalização MinMaxScaler)
print("Normalizando as variáveis com MinMaxScaler (0 a 1)...")
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test)
print("Exemplo de dados escalonados (primeira linha):", x_train_escalonado[0])

# 13. Buscando o melhor valor de K para o KNN
print("Testando diferentes valores de K para o KNN (de 1 a 9)...")
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test_escalonado)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Taxa de Erro Médio para cada valor de K')
plt.xlabel('Valor de K')
plt.ylabel('Erro Médio')
plt.show()

# 14. Treinando o modelo KNN com o melhor K (exemplo: 5)
print("Treinando o modelo KNN com n_neighbors=5...")
modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train)
print("Modelo treinado.")

# 15. Fazendo previsões no conjunto de teste
print("Realizando predições no conjunto de teste...")
y_predito = modelo_classificador.predict(x_test_escalonado)

# 16. Matriz de confusão
print("Exibindo matriz de confusão...")
matriz_confusao = confusion_matrix(y_test, y_predito)
plt.figure(figsize=(8, 4))
sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predição")
plt.ylabel("Dados Reais")
plt.title("Matriz de Confusão - KNN")
plt.show()

# 17. Relatório de classificação (precision, recall, f1-score, acurácia)
print("Relatório de classificação:")
print(classification_report(y_test, y_predito))

print("Pipeline de detecção de fraude finalizado.")