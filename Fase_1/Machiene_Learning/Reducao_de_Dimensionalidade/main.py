from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

# Exibe todas as colunas do dataframe
pd.set_option('display.max_columns', None)

# Vamos carregar o arquivo csv em nosso drive e analisarmos o nosso dataframe
df_fifa = pd.read_csv("players_22.csv")
df_fifa

# mostraremos as 2 primeiras linhas do dataframe para verificar se o arquivo foi carregado corretamente
df_fifa.head(2)

# mostraremos o número de linhas e colunas do dataframe
df_fifa.shape


# Analisando informações básicas sobre nosso dataframe como tipos de variáveis, tamanho, amostra estatísticas básicas e etc...
display(df_fifa.describe())


print("Informações sobre o DataFrame df_fifa:")
print(df_fifa.info())


# Criando uma matriz de correlação para verificar a relação entre as variáveis do dataframe
# Criando dataframe somente com nossas variáveis numericas
df_fifa_numerico = df_fifa.select_dtypes([np.number])
# Calcula a matriz de correlação
# Cruza as colunas do dataframe para verificar a relação entre as variáveis
# entre si, ou seja, se uma variável influencia na outra.
correlation_matrix = df_fifa_numerico.corr()
correlation_matrix

# Quando se fala de coorelacao agt esta comparando uma variável com outra, ou seja, se uma variável influencia na outra.
# Exibindo a matriz de correlação
import seaborn as sns
# Visualização da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação')
plt.show()


# Dados Nulos
# Vamos verificar quantidades de nulos em nosso dataframe, não podemos aplicar o PCA
# se nossos dados tiverem linhas nulas, devemos tratar esses casos caso ocorram
# Existem diversas técnicas para tratar dados nulos, o método de escolha depende muito do seu objetivo
df_fifa_numerico.isnull().sum()

# Tratando as colunas nulas
from sklearn.impute import SimpleImputer
# Preenche os valores NaN com a média das colunas
imputer = SimpleImputer(strategy='mean')
df_fifa_numerico = pd.DataFrame(imputer.fit_transform(df_fifa_numerico), columns=df_fifa_numerico.columns)

# Padroniza as variáveis
scaler = StandardScaler()
df_fifa_padronizado = scaler.fit_transform(df_fifa_numerico)

# Calcula a variância explicada acumulada
pca = PCA() # Inicializa o PCA
pca.fit(df_fifa_padronizado) #Treinando o PCA com os dados padronizados
variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_) # Variância acumulada
# Visualização da variância explicada acumulada
plt.plot(range(1, len(variancia_cumulativa) + 1), variancia_cumulativa, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Acumulada Explicada')
plt.title('Variância Acumulada Explicada pelo PCA')
plt.show()

# Vamos definir um limiar de 80%, ou seja, queremos obter uma porcentagem de explicancia sobre
# nossos dados de igual a 80%
limiar_de_variancia = 0.80

# Encontrar o número de componentes necessários para atingir ou ultrapassar o limiar
num_de_pca = np.argmax(variancia_cumulativa >= limiar_de_variancia) + 1

print(f"Número de Componentes para {limiar_de_variancia * 100}% da Variância: {num_de_pca}")
#Resultado = Número de Componentes para 80.0% da Variância: 10
#Ou seja utilizando 10 componentes principais, conseguimos explicar 80% da variância dos dados.

# O PCA sao sempre aplicados nas variáveis numéricas, ou seja, não podemos aplicar o PCA em variáveis categóricas.

# Por fim vamos então utilizar nosso número de PCA desejado e reduzir nossas 59 columns para 10


# Inicializa o objeto PCA
pca = PCA(n_components=num_de_pca)
# Aplica o PCA aos dados padronizados
principal_components = pca.fit_transform(df_fifa_padronizado)


# Pegando o número de componentes principais gerados
num_components = principal_components.shape[1]
# Gerando uma lista para cada PCA
column_names = [f'PC{i}' for i in range(1, num_components + 1)]
# Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA
pca_df = pd.DataFrame(data=principal_components, columns=column_names)

pca_df

# Verificando a normalidade dos dados
# Criar histogramas para cada coluna
plt.figure(figsize=(15, 8))
for i, col in enumerate(pca_df.columns[:10]):
    plt.subplot(2, 5, i + 1)  # Aqui, ajustei para 2 linhas e 5 colunas
    sns.histplot(pca_df[col], bins=20, kde=True)
    plt.title(f'Histograma {col}')
plt.tight_layout()
plt.show()


# Verificando a normalidade dos dados, se os dados seguem uma distribuição normal ou não.
from scipy.stats import shapiro
# Vamos olhar para cada coluna a normalidade após a redução de dimensionalidade
for column in pca_df.columns:
    stat, p_value = shapiro(pca_df[column])
    print(f'Variável: {column}, Estatística de teste: {stat}, Valor p: {p_value}')

    # Você pode então interpretar o valor p para determinar se a variável segue uma distribuição normal
    if p_value > 0.05:
        print(f'A variável {column} parece seguir uma distribuição normal.\n')
    else:
        print(f'A variável {column} não parece seguir uma distribuição normal.\n')





