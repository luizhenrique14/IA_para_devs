# KMN e SVN

# ============================================================
# Análise e Modelagem de Dados de Recrutamento
# ============================================================
# Passo a passo deste notebook/script:
#
# 1. Leitura e visualização inicial dos dados
# 2. Análise exploratória: valores únicos, estatísticas, valores nulos
# 3. Visualização de dados: gráficos de distribuição, boxplots, swarmplots, violin plots, pairplots, heatmaps
# 4. Pré-processamento: tratamento de nulos, Label Encoding, One Hot Encoding
# 5. Nova análise de correlação após encoding
# 6. Seleção de features e target
# 7. Separação em treino e teste
# 8. Escalonamento dos dados
# 9. Treinamento e avaliação do modelo KNN (incluindo busca do melhor K)
# 10. Treinamento e avaliação do modelo SVM
# 11. Observações e exemplos de pipelines para outros modelos
# ============================================================

import pandas as pd

# 📥 Lendo a base de dados
print("\n📥 Lendo a base de dados...")
dados = pd.read_excel('Michine_learning_Avancado\Aula 02\Recrutamento.xlsx')
print("Formato dos dados:", dados.shape)
print("Primeiras linhas do DataFrame:")
print(dados.head(10))

# 🔎 Checando valores únicos da coluna status
print("\n🔎 Checando valores únicos da coluna 'status'...")
print("Valores únicos em status:", set(dados.status))

# 📊 Estatísticas descritivas
print("\n📊 Estatísticas descritivas das variáveis numéricas:")
print(dados.describe())

# 🕳️ Visualizando valores nulos
import missingno as msno 
print("\n🕳️ Visualizando valores nulos na base...")
msno.matrix(dados)
import matplotlib.pyplot as plt
plt.show()  # <-- Adicionado para exibir o gráfico

# 🧮 Contando valores nulos por coluna
print("\n🧮 Contando valores nulos por coluna:")
print(dados.isnull().sum())

# 💰 Visualizando distribuição de salário por status
import seaborn as sb
print("\n💰 Visualizando distribuição de salário por status (boxplot)...")
sb.boxplot(x='status', y='salary', data=dados, palette='hls')
plt.show()  # <-- Adicionado para exibir o gráfico

# 🩹 Preenchendo valores nulos de salário com 0
print("\n🩹 Preenchendo valores nulos de salário com 0...")
dados['salary'].fillna(value=0, inplace=True)
print("Valores nulos após preenchimento:")
print(dados.isnull().sum())

# 📦 Análise de outliers e distribuição das variáveis numéricas
print("\n📦 Análise de outliers e distribuição das variáveis numéricas:")
for col in ["hsc_p", "degree_p", "etest_p", "mba_p", "salary"]:
    print(f"  📦 Boxplot e histograma para: {col}")
    sb.boxplot(x=dados[col])
    plt.show()  # <-- Adicionado para exibir o gráfico
    sb.histplot(data=dados, x=col)
    plt.show()  # <-- Adicionado para exibir o gráfico

# 🎓 Swarmplot: Relação entre mba_p, status e experiência de trabalho
print("\n🎓 Swarmplot: Relação entre mba_p, status e experiência de trabalho...")
sb.set_theme(style="whitegrid", palette="muted")
ax = sb.swarmplot(data=dados, x="mba_p", y="status", hue="workex")
ax.set(ylabel="")
plt.show()  # <-- Adicionado para exibir o gráfico

# 👩‍💼👨‍💼 Violin plot: Salário por especialização e gênero
print("\n👩‍💼👨‍💼 Violin plot: Salário por especialização e gênero...")
import plotly_express as px
fig = px.violin(dados, y="salary", x="specialisation", color="gender", box=True, points="all")
fig.show()  # <-- Adicionado para exibir o gráfico

# 📈 Pairplot: Scores acadêmicos por status
print("\n📈 Pairplot: Scores acadêmicos por status...")
sb.pairplot(dados, vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'], hue="status")
plt.show()  # <-- Adicionado para exibir o gráfico

# 🧩 Matriz de correlação das variáveis numéricas
print("\n🧩 Matriz de correlação das variáveis numéricas (apenas colunas numéricas)...")
dados_numericos = dados.select_dtypes(include=['number'])
print("Colunas numéricas consideradas para a correlação:", list(dados_numericos.columns))
correlation_matrix = dados_numericos.corr().round(2)
fig, ax = plt.subplots(figsize=(8,8))    
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.show()  # <-- Adicionado para exibir o gráfico

# 🏷️ Aplicando LabelEncoder nas colunas categóricas
print("\n🏷️ Aplicando LabelEncoder nas colunas categóricas ['gender', 'workex', 'specialisation', 'status']...")
from sklearn.preprocessing import LabelEncoder
colunas = ['gender', 'workex', 'specialisation', 'status']
label_encoder = LabelEncoder()
for col in colunas:
    print(f"  ➡️ Transformando coluna '{col}' em valores numéricos...")
    dados[col] = label_encoder.fit_transform(dados[col])
print("Primeiras linhas após LabelEncoder:")
print(dados.head())

# 🏷️➡️🔢 One Hot Encoding para variáveis categóricas com mais de duas categorias
print("\n🏷️➡️🔢 Aplicando One Hot Encoding em 'hsc_s' e 'degree_t' (criando variáveis dummies)...")
dummy_hsc_s = pd.get_dummies(dados['hsc_s'], prefix='dummy')
dummy_degree_t = pd.get_dummies(dados['degree_t'], prefix='dummy')
print("Colunas criadas para 'hsc_s':", list(dummy_hsc_s.columns))
print("Colunas criadas para 'degree_t':", list(dummy_degree_t.columns))

print("➡️ Concatenando variáveis dummies ao DataFrame principal...")
dados_coeded = pd.concat([dados, dummy_hsc_s, dummy_degree_t], axis=1)

print("➡️ Removendo colunas originais categóricas ['hsc_s', 'degree_t', 'salary'] do DataFrame codificado...")
dados_coeded.drop(['hsc_s', 'degree_t', 'salary'], axis=1, inplace=True)
print("Colunas restantes após remoção:", list(dados_coeded.columns))
print("Primeiras linhas após One Hot Encoding e remoção das colunas originais:")
print(dados_coeded.head())

# 🧩 Nova matriz de correlação após encoding
print("\n🧩 Nova matriz de correlação após encoding (apenas variáveis numéricas)...")
dados_coeded_numerico = dados_coeded.select_dtypes(include=['number'])
print("Colunas numéricas consideradas para a nova correlação:", list(dados_coeded_numerico.columns))
correlation_matrix = dados_coeded_numerico.corr().round(2)
fig, ax = plt.subplots(figsize=(12,12))    
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.show()  # <-- Adicionado para exibir o gráfico

# 🔬 Relplot: Relação entre status e ssc_p
print("\n🔬 Relplot: Relação entre status e ssc_p (após encoding)...")
sb.relplot(x="status", y="ssc_p", hue="status", size="ssc_p",
           sizes=(40, 400), alpha=.5, palette="muted",
           height=6, data=dados_coeded)
plt.show()  # <-- Adicionado para exibir o gráfico

# 🏁 Seleção de features e target
print("\n🏁 Selecionando features ['ssc_p', 'hsc_p', 'degree_p', 'workex', 'mba_p'] e target ['status'] para o modelo...")
x = dados_coeded[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'mba_p']]
y = dados_coeded['status']
print("Features selecionadas:", list(x.columns))
print("Target selecionado: status")

# ✂️ Separação em treino e teste
print("\n✂️ Separando dados em treino e teste (80% treino, 20% teste)...")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7)
print("Formato do treino:", x_train.shape)
print("Formato do teste:", x_test.shape)

# 📏 Escalonamento dos dados
print("\n📏 Escalonando os dados (StandardScaler)...")
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
scaler = StandardScaler() 
scaler.fit(x_train)
x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test) 
print("Exemplo de dados escalonados (primeira linha):", x_train_escalonado[0])

# 🔢 Testando diferentes valores de K para o KNN
print("\n🔢 Testando diferentes valores de K para o KNN (de 1 a 9)...")
import numpy as np
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test_escalonado)
    error.append(np.mean(pred_i != y_test))
print("Erros médios para cada valor de K (de 1 a 9):", error)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()  # <-- Adicionado para exibir o gráfico

# 🤖 Treinando o modelo KNN com K=5
print("\n🤖 Treinando o modelo KNN com K=5...")
modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train) 

# 🔮 Fazendo previsões com o modelo KNN
print("\n🔮 Fazendo previsões com o modelo KNN...")
y_predito = modelo_classificador.predict(x_test_escalonado) 
print("Previsões do KNN:", y_predito)

# 🏆 Avaliando a acurácia do modelo KNN
from sklearn.metrics import accuracy_score
print("\n🏆 Avaliando a acurácia do modelo KNN...")
print("Acurácia do KNN:", accuracy_score(y_test, y_predito))

# 🤖 Testando com o modelo SVM
print("\n🤖 Treinando e avaliando o modelo SVM (LinearSVC)...")
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
svm = Pipeline([
    ("linear_svc", LinearSVC(C=1))
])
svm.fit(x_train_escalonado, y_train) 
y_predito_svm = svm.predict(x_test_escalonado) 
print("Acurácia do SVM:", accuracy_score(y_test, y_predito_svm))

# 📊 Importando métricas para possíveis análises futuras
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# ⚠️ Atenção: predict_proba não está disponível para LinearSVC, use outro classificador se necessário
# y_prob = modelo_classificador.predict_proba(x_test)[:,1] 

# 🧪 Exemplo de pipeline para SVM polinomial (não treinado aqui)
from sklearn.svm import SVC
poly_svm = Pipeline([
    ("svm", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
# poly_svm.fit(x_train, y_train) # Descomente para treinar o SVM polinomial