import pandas as pd
#pip install -U scikit-learn
from sklearn.model_selection import train_test_split


dados = pd.read_excel('gaf_esp.xlsx') # Lê o arquivo Excel com os dados dos gafanhotos
dados.head() # Mostra as 5 primeiras linhas do DataFrame

dados.describe() # Mostra estatísticas descritivas dos dados


dados.groupby('Espécie').describe()# Agrupa os dados por espécie e mostra estatísticas descritivas

dados.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas') # Plota um gráfico de dispersão entre o comprimento do abdômen e o comprimento das antenas

x = dados[['Comprimento do Abdômen', 'Comprimento das Antenas']] # Seleciona as colunas de comprimento do abdômen e comprimento das antenas como variáveis independentes
y = dados['Espécie'] # Seleciona a coluna de espécie como variável dependente


x_train, x_test, y_train, y_test = train_test_split(x,  # Seleciona as colunas de comprimento do abdômen e comprimento das antenas como variáveis independentes
                                                    y,  # Seleciona a coluna de espécie como variável dependente
                                                    test_size=0.2,  # Define o tamanho do conjunto de teste como 20% dos dados
                                                    stratify=y, # Mantém a proporção das classes no conjunto de teste, garante o equilibrio entre as classes, ou seja, a mesma proporção de gafanhotos e esperancas em relação ao total de dados
                                                    random_state=42 # Define uma semente aleatória para garantir a reprodutibilidade dos resultados, desordenando os dados
                                                    )
list(y_train).count('Gafanhoto') # Conta o número de gafanhotos no conjunto de treino

print("Total base de treino: ", len(x_train))
print("Total base de teste: ", len(y_test))
