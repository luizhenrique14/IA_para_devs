import pandas as pd
#pip install -U scikit-learn
from sklearn.model_selection import train_test_split


dados = pd.read_excel('gaf_esp.xlsx') # Lê o arquivo Excel com os dados dos gafanhotos
dados.head() # Mostra as 5 primeiras linhas do DataFrame

dados.describe() # Mostra estatísticas descritivas dos dados


dados.groupby('Espécie').describe() # Agrupa os dados por espécie e mostra estatísticas descritivas

dados.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas') # Plota um gráfico de dispersão entre o comprimento do abdômen e o comprimento das antenas

x = dados[['Comprimento do Abdômen', 'Comprimento das Antenas']] # Seleciona as colunas de comprimento do abdômen e comprimento das antenas como variáveis independentes
y = dados['Espécie'] # Seleciona a coluna de espécie como variável dependente


x_train, x_test, y_train, y_test = train_test_split(x,  # Seleciona as colunas de comprimento do abdômen e comprimento das antenas como variáveis independentes
                                                    y,  # Seleciona a coluna de espécie como variável dependente
                                                    test_size=0.2,  # Define o tamanho do conjunto de teste como 20% dos dados
                                                    stratify=y, # Mantém a proporção das classes no conjunto de teste, garante o equilibrio entre as classes, ou seja, a mesma proporção de gafanhotos e esperancas em relação ao total de dados
                                                    random_state=42 # Define uma semente aleatória para garantir a reprodutibilidade dos resultados, desordenando os dados
                                                    )
list(x_train).count('Gafanhoto') # Conta o número de gafanhotos no conjunto de treino
# 40 - por conta do stratify, o número de gafanhotos e gafanhoto no conjunto de treino é o mesmo, ou seja, 40 gafanhotos e 40 espernacas
list(y_test).count('Esperanca') # Conta o número de gafanhotos no conjunto de teste
# 40 - por conta do stratify, o número de gafanhotos e espernacas no conjunto de teste é o mesmo, ou seja, 10 gafanhotos e 10 espernacas

print("Total base de treino: ", len(x_train))
print("Total base de teste: ", len(y_test))
# Total base de treino:  80
# Total base de teste:  20


# KNeighborsClassifier ou KMN basicamente e um algoritimo com a classificacao dos dados com bas ena distancia de dados vizinhos
from sklearn.neighbors import KNeighborsClassifier # Importa o classificador KNN

# Hiperparametro do nosos modelo é o número de vizinhos considerado (n_neighbors)
modelo_classificador = KNeighborsClassifier(n_neighbors=3) # Cria o classificador KNN com 3 vizinhos mais próximos
# O modelo KNN funciona da seguinte forma: ele pega os 3 vizinhos mais próximos do ponto que você quer classificar e vê qual é a classe que mais aparece entre eles. Se a maioria deles for gafanhoto, o ponto é classificado como gafanhoto. Se a maioria for espernaca, o ponto é classificado como espernaca.

# Está fazendo o treinamento do meu modelo de ML
modelo_classificador.fit(x_train, y_train) # Treina o classificador KNN com os dados de treino (x_train e y_train)

# Realizando as previsões
# Comprimento AB: 8
# Comprimento AT: 6
modelo_classificador.predict([[8,6]]) # Faz uma previsão para um esperanca com comprimento do abdômen de 8 e comprimento das antenas de 6

from sklearn.metrics import accuracy_score # Importa a função para calcular a acurácia do modelo
# A acurácia é a proporção de previsões corretas em relação ao total de previsões feitas. É uma métrica comum para avaliar o desempenho de modelos de classificação.
y_predito = modelo_classificador.predict(x_test) # Faz previsões para o conjunto de teste (x_test) e armazena os resultados em y_predito


accuracy_score(y_true = y_test # y_true = y_test é o valor real, ou seja, o que realmente deveria ser a classificação do dado
               , y_pred=y_predito # y_pred = y_predito é o valor previsto pelo modelo, ou seja, o que o modelo acha que deveria ser a classificação do dado
               ) # Calcula a acurácia do modelo comparando as previsões (y_predito) com os valores reais (y_test), mostra o percentual de acerto do modelo
# 0.95 - 95% de acerto do modelo, ou seja, o modelo acertou 95% das previsões feitas no conjunto de teste

