# Se necessário, instale as bibliotecas:
# pip install pandas matplotlib seaborn scikit-learn

# Importação das bibliotecas para manipulação de dados e visualização
import pandas as pd  # Manipulação de dados em DataFrame
import matplotlib.pyplot as plt  # Visualização de gráficos
import seaborn as sns  # Visualização avançada de gráficos
import numpy as np  # Operações numéricas

# Importação das funções do scikit-learn para modelagem e avaliação
from sklearn.model_selection import train_test_split  # Separação de treino e teste
from sklearn.metrics import accuracy_score  # Métrica de acurácia

from sklearn.tree import DecisionTreeClassifier  # Árvore de decisão
from sklearn.tree import plot_tree  # Função para plotar árvore
from sklearn import tree  # Módulo para manipulação de árvores

from sklearn.ensemble import RandomForestClassifier  # Floresta aleatória (Random Forest)
from sklearn.tree import export_graphviz  # Exportação de árvore para visualização externa

print("Iniciando pipeline de modelos baseados em árvores...")

# 1. Carregando a base de dados
print("Lendo o arquivo de dados...")
dados = pd.read_csv("card_transdata.csv", sep=",")
print("Primeiras linhas do dataset:")
print(dados.head())

# 2. Checando valores nulos
print("Verificando valores nulos por coluna:")
print(dados.isnull().sum())

# 3. Removendo linhas com valores nulos (se houver)
if dados.isnull().sum().sum() > 0:
    print("Removendo linhas com valores nulos...")
    dados = dados.dropna()
else:
    print("Nenhum valor nulo encontrado.")

# 4. Análise de correlação entre as variáveis
print("Calculando matriz de correlação...")
correlation_matrix = dados.corr().round(2)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.title("Matriz de Correlação das Variáveis")
plt.show()

# 5. Separando features (X) e target (y)
print("Separando variáveis preditoras (X) e alvo (y)...")
x = dados.drop(columns=['fraud'])  # Todas as colunas menos 'fraud'
y = dados['fraud']  # Coluna alvo

# 6. Dividindo em treino e teste
print("Dividindo os dados em treino (80%) e teste (20%)...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=7
)
print(f"Formato do treino: {x_train.shape}, Formato do teste: {x_test.shape}")

# 7. Criando e treinando o modelo de Árvore de Decisão
print("Criando modelo de Árvore de Decisão...")
dt = DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth=2)
dt.fit(x_train, y_train)
print("Modelo treinado.")

# 8. Fazendo previsões com a Árvore de Decisão
print("Realizando previsões com o modelo de Árvore de Decisão...")
y_predito = dt.predict(x_test)

# 9. Visualizando a árvore de decisão
print("Plotando a árvore de decisão...")
tree.plot_tree(dt)
plt.title("Árvore de Decisão (max_depth=2)")
plt.show()

# 10. Visualização detalhada da árvore com nomes de features e classes
class_names = ['Não Fraude', 'Fraude']
label_names = [
    'distance_from_home', 'distance_from_last_transaction',
    'ratio_to_median_purchase_price', 'repeat_retailer',
    'used_chip', 'used_pin_number', 'online_order'
]
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,15), dpi=300)
tree.plot_tree(
    dt,
    feature_names=label_names,
    class_names=class_names,
    filled=True
)
plt.title("Árvore de Decisão Detalhada")
fig.savefig('decision_tree.png')
plt.show()

# 11. Avaliando o desempenho da Árvore de Decisão
acc_dt = accuracy_score(y_test, y_predito)
print(f"Acurácia da Árvore de Decisão: {acc_dt:.4f}")

# 12. Criando e treinando o modelo Random Forest
print("Criando modelo Random Forest com 5 árvores (estimators)...")
rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=7)
rf.fit(x_train, y_train)
print("Modelo Random Forest treinado.")

# 13. Fazendo previsões com o Random Forest
print("Realizando previsões com o modelo Random Forest...")
y_predito_random_forest = rf.predict(x_test)

# 14. Avaliando o desempenho do Random Forest
acc_rf = accuracy_score(y_test, y_predito_random_forest)
print(f"Acurácia do Random Forest: {acc_rf:.4f}")

# 15. Visualizando uma árvore individual da Random Forest
print("Plotando uma árvore individual da Random Forest...")
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=800)
tree.plot_tree(
    rf.estimators_[0],
    feature_names=label_names,
    class_names=class_names,
    filled=True
)
plt.title("Primeira Árvore da Random Forest")
fig.savefig('rf_individualtree.png')
plt.show()

# 16. Plotando todas as árvores da Random Forest
print("Plotando as 5 árvores da Random Forest...")
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15,3), dpi=900)
for index in range(0, 5):
    tree.plot_tree(
        rf.estimators_[index],
        feature_names=label_names,
        class_names=class_names,
        filled=True,
        ax=axes[index]
    )
    axes[index].set_title(f'Estimator: {index}', fontsize=11)
fig.savefig('rf_5trees.png')
plt.show()

# 17. Score do Random Forest nos dados de treino e teste
print(f"Score (acurácia) do Random Forest no treino: {rf.score(x_train, y_train):.4f}")
print(f"Score (acurácia) do Random Forest no teste: {rf.score(x_test, y_test):.4f}")

print("Pipeline finalizado.")