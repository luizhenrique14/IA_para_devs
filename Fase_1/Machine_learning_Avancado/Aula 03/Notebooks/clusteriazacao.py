# 1. Importação das bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 2. Leitura dos dados
# Lê o arquivo CSV com os dados dos clientes do shopping.
dados = pd.read_csv("mall.csv", sep=',')

# 3. Análise e limpeza dos dados
# Verifica se há valores nulos em cada coluna.
print(dados.isnull().sum())
# Mostra estatísticas descritivas das variáveis numéricas (média, desvio, min, max, etc).
print(dados.describe())

# 4. Visualização dos dados
# Plota histogramas para todas as variáveis, ajudando a entender a distribuição dos dados.
dados.hist(figsize=(12,12))
plt.show()

# 5. Análise de correlação
# Plota um heatmap mostrando a correlação entre idade, renda anual e pontuação de gastos.
plt.figure(figsize=(6,4))
sns.heatmap(dados[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(method = 'pearson'), annot=True, fmt=".1f")
plt.show()

# 6. Feature Scaling (Padronização dos dados)
# Padroniza as variáveis de renda anual e pontuação de gastos para média 0 e desvio padrão 1.
# Isso é importante para algoritmos baseados em distância, como KMeans e DBSCAN.
scaler = StandardScaler()
scaler.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
dados_escalonados = scaler.transform(dados[['Annual Income (k$)','Spending Score (1-100)']])

# 7. K-Means sem escalonamento
# Aplica o algoritmo KMeans diretamente nos dados originais (sem padronizar).
# Define 6 clusters (n_clusters=6) e treina o modelo.
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
centroides = kmeans.cluster_centers_  # Coordenadas dos centros de cada cluster
kmeans_labels = kmeans.predict(dados[['Annual Income (k$)','Spending Score (1-100)']])  # Rótulo do cluster para cada cliente

# 8. K-Means com escalonamento
# Aplica o KMeans nos dados padronizados (escalonados).
kmeans_escalonados = KMeans(n_clusters=6, random_state=0)
kmeans_escalonados.fit(dados_escalonados)
centroides_escalonados = kmeans_escalonados.cluster_centers_
kmeans_labels_escalonado = kmeans_escalonados.predict(dados_escalonados)

# 9. Visualização dos clusters com feature scaling
# Plota os dados escalonados coloridos por cluster e marca os centroides.
plt.scatter(dados_escalonados[:, 0], dados_escalonados[:, 1], c=kmeans_labels_escalonado, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual (escalonado)')
plt.ylabel('Pontuação de gastos (escalonado)')
plt.scatter(centroides_escalonados[:, 0], centroides_escalonados[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.show()

# 10. Visualização dos clusters sem feature scaling
# Plota os dados originais coloridos por cluster e marca os centroides.
plt.scatter(dados[['Annual Income (k$)']], dados[['Spending Score (1-100)']], c=kmeans_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual')
plt.ylabel('Pontuação de gastos')
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.show()

# 11. Escolhendo o número ideal de clusters (método do cotovelo)
# Testa diferentes valores de K (de 1 a 9) e calcula a soma dos erros quadráticos (inércia) para cada K.
# O ponto onde a redução da inércia começa a diminuir (cotovelo) indica o melhor número de clusters.
sse = []
k = list(range(1, 10))
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
    sse.append(kmeans.inertia_)
plt.plot(k, sse, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
plt.show()

# 12. DBSCAN (Clusterização baseada em densidade)
# Aplica o algoritmo DBSCAN, que agrupa pontos próximos com base em densidade.
# eps define o raio de vizinhança e min_samples o mínimo de pontos para formar um cluster.
dbscan = DBSCAN(eps=10, min_samples=8)
dbscan.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
dbscan_labels = dbscan.labels_  # Rótulo do cluster para cada cliente (-1 indica outlier)

# 13. Visualização dos clusters DBSCAN
# Plota os dados coloridos pelos clusters encontrados pelo DBSCAN.
plt.scatter(dados[['Annual Income (k$)']], dados[['Spending Score (1-100)']], c=dbscan_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# 14. Avaliação dos agrupamentos
# Silhouette Score para KMeans: mede o quão bem os pontos estão agrupados (quanto maior, melhor, máximo 1).
print("Silhouette Score KMeans:", silhouette_score(dados[['Annual Income (k$)','Spending Score (1-100)']], kmeans_labels))
# Silhouette Score para DBSCAN: mesma métrica para o DBSCAN.
print("Silhouette Score DBSCAN:", silhouette_score(dados[['Annual Income (k$)','Spending Score (1-100)']], dbscan_labels))
# Adjusted Rand Score: compara a similaridade entre os agrupamentos do KMeans e do DBSCAN.
print("Adjusted Rand Score:", adjusted_rand_score(kmeans_labels, dbscan_labels))