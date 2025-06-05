# Se necessário, instale as bibliotecas:
# pip install pandas scikit-learn matplotlib seaborn numpy plotly-express openpyxl

# 1. Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados tabulares
import seaborn as sb  # Visualização de dados
import matplotlib.pyplot as plt  # Visualização de gráficos
import numpy as np  # Operações numéricas
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler  # Pré-processamento
from sklearn.model_selection import train_test_split  # Separação treino/teste
from sklearn.neighbors import KNeighborsClassifier  # Algoritmo KNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report  # Avaliação
from sklearn.pipeline import Pipeline  # Pipeline para SVM
from sklearn.svm import LinearSVC  # SVM linear
from sklearn import metrics  # Métricas diversas
from sklearn.metrics import roc_curve, auc  # Curva ROC e AUC
import plotly_express as px  # Visualização interativa

print("Iniciando pipeline de recrutamento preditivo...")

# 2. Carregando a base de dados
print("Lendo o arquivo de dados de recrutamento...")
dados = pd.read_excel('Recrutamento.xlsx')
print("Primeiras linhas dos dados:")
print(dados.head())

# 3. Análise inicial dos dados
print("Formato do dataset (linhas, colunas):", dados.shape)
print("Verificando valores nulos por coluna:")
print(dados.isnull().sum())

# 4. Tratando valores nulos na coluna 'salary'
print("Preenchendo valores nulos em 'salary' com 0...")
dados['salary'].fillna(value=0, inplace=True)
print("Valores nulos restantes por coluna:")
print(dados.isnull().sum())

# 5. Análise exploratória das variáveis numéricas
print("Visualizando distribuição das principais variáveis numéricas...")
for col in ['hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']:
    plt.figure()
    sb.boxplot(x=dados[col])
    plt.title(f'Boxplot de {col}')
    plt.show()
    plt.figure()
    sb.histplot(data=dados, x=col)
    plt.title(f'Histograma de {col}')
    plt.show()

# 6. Análise de viés de gênero na remuneração
print("Visualizando distribuição de salário por gênero e especialização...")
px.violin(dados, y="salary", x="specialisation", color="gender", box=True, points="all").show()

# 7. Análise de correlação entre scores acadêmicos e contratação
print("Visualizando relação entre scores acadêmicos e contratação...")
sb.pairplot(dados, vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'], hue="status")
plt.show()

# 8. Análise de correlação entre variáveis numéricas
print("Matriz de correlação entre variáveis numéricas:")
correlation_matrix = dados.corr().round(2)
fig, ax = plt.subplots(figsize=(8,8))
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.title("Matriz de Correlação")
plt.show()

# 9. Transformação de variáveis categóricas com LabelEncoder
print("Aplicando LabelEncoder em variáveis categóricas com duas categorias...")
colunas = ['gender', 'workex', 'specialisation', 'status']
label_encoder = LabelEncoder()
for col in colunas:
    dados[col] = label_encoder.fit_transform(dados[col])
print("Exemplo após encoding:")
print(dados.head())

# 10. Aplicando One Hot Encoding em variáveis com múltiplas categorias
print("Aplicando One Hot Encoding em 'hsc_s' e 'degree_t'...")
dummy_hsc_s = pd.get_dummies(dados['hsc_s'], prefix='dummy')
dummy_degree_t = pd.get_dummies(dados['degree_t'], prefix='dummy')
dados_coeded = pd.concat([dados, dummy_hsc_s, dummy_degree_t], axis=1)
dados_coeded.drop(['hsc_s', 'degree_t', 'salary'], axis=1, inplace=True)
print("Exemplo após One Hot Encoding:")
print(dados_coeded.head())

# 11. Nova matriz de correlação após encoding
print("Nova matriz de correlação após encoding:")
correlation_matrix = dados_coeded.corr().round(2)
fig, ax = plt.subplots(figsize=(12,12))
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.title("Matriz de Correlação Pós-Encoding")
plt.show()

# 12. Seleção de features e target
print("Selecionando features e variável alvo...")
x = dados_coeded[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'mba_p']]
y = dados_coeded['status']

# 13. Separação em treino e teste
print("Dividindo dados em treino (80%) e teste (20%)...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=7
)
print(f"Formato do treino: {x_train.shape}, Formato do teste: {x_test.shape}")

# 14. Escalonamento das variáveis (StandardScaler)
print("Padronizando as variáveis com StandardScaler...")
scaler = StandardScaler()
scaler.fit(x_train)
x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test)
print("Exemplo de dados escalonados (primeira linha):", x_train_escalonado[0])

# 15. Testando diferentes valores de K para o KNN
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

# 16. Treinando o modelo KNN com o melhor K (exemplo: 5)
print("Treinando o modelo KNN com n_neighbors=5...")
modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train)
print("Modelo treinado.")

# 17. Fazendo previsões no conjunto de teste
print("Realizando predições no conjunto de teste...")
y_predito = modelo_classificador.predict(x_test_escalonado)

# 18. Matriz de confusão e relatório de classificação para KNN
print("Exibindo matriz de confusão para KNN...")
matriz_confusao = confusion_matrix(y_true=y_test, y_pred=y_predito)
figure = plt.figure(figsize=(15, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)
disp.plot(values_format='d')
plt.title("Matriz de Confusão - KNN")
plt.show()
print("Relatório de classificação para KNN:")
print(classification_report(y_test, y_predito))

# 19. Treinando e avaliando modelo SVM
print("Treinando modelo SVM (LinearSVC)...")
svm = Pipeline([
    ("linear_svc", LinearSVC(C=1))
])
svm.fit(x_train_escalonado, y_train)
y_predito_svm = svm.predict(x_test_escalonado)
print("Exibindo matriz de confusão para SVM...")
matriz_confusao_svm = confusion_matrix(y_true=y_test, y_pred=y_predito_svm)
figure = plt.figure(figsize=(15, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao_svm)
disp.plot(values_format='d')
plt.title("Matriz de Confusão - SVM")
plt.show()
print("Relatório de classificação para SVM:")
print(classification_report(y_test, y_predito_svm))

# 20. Curva ROC e cálculo da AUC para o modelo KNN
print("Calculando probabilidades para curva ROC (KNN)...")
y_prob = modelo_classificador.predict_proba(x_test_escalonado)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(f"AUC do modelo KNN: {roc_auc:.4f}")

print("Plotando curva ROC...")
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, true_positive_rate, color='red', label=f'AUC = {roc_auc:.2f}')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print("Pipeline de recrutamento preditivo finalizado.")