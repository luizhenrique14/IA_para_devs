# Reflexo de um comportamento para saber se o cliente vai fechar a conta ou NotADirectoryError
import pandas as pd

df = pd.read_csv("Churn_Modelling.csv", sep=";")
df.head()


# mostrando no grafico de caixas/block spot
import matplotlib.pyplot as plt

# Criar o gráfico de boxplot
plt.boxplot(df['CreditScore'])
plt.title('CreditScore')
plt.ylabel('Valores')
plt.show()

# Transformando os dados que nao sao numero em numero, repassando para essa transformacao numerica
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Ajustar e transformar os rótulos
df['Surname'] = label_encoder.fit_transform(df['Surname'])
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df.head()


# Treino e Teste
# OK, como próximo passo, antes de normalizar ou padronizar, vamos separar os dados em treino e teste:
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Exited']) # Variáveis características
y = df['Exited'] # O que eu quero prever. (Target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Tipos de escalonamento de dados


# Padronização - MinMaxScaler
# Min Max Scaler - Normalizacao
# Normalização é ajustar os dados para que fiquem numa mesma escala (geralmente entre 0 e 1), melhorando o desempenho do modelo.
# Evita que variáveis com valores muito diferentes dominem o treinamento.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Aplicando a normalização nos dados de treino e teste utilizano o MinMaxScaler:
minMaxScaler = MinMaxScaler() #chamando o metodo de normalização dos dados (0-1)

minMaxScaler.fit(X_train)# Ajusta o scaler aos dados de treino, deve sempre aplicar somente na base de treino, nunca na base de teste, pois isso pode vazar informações do teste para o treino, assim criando um viés no modelo.

# Transformando os dados de treino e teste
x_train_min_max_scaled = minMaxScaler.transform(X_train)# Transformando os dados de treino, para a base de treino
x_test_min_max_scaled= minMaxScaler.transform(X_test)# Transformando os dados de teste, para a base de teste
# Verificando os dados normalizados
x_train_min_max_scaled


# Padronização - StandardScaler
# Scorer - Padronização
# Padronização é ajustar os dados para que tenham média 0 e desvio padrão 1, mantendo a distribuição original dos dados.
standardScaler = StandardScaler() #chamando o metodo de padronização dos dados (média e std)

standardScaler.fit(X_train)# qual média e std será utilizado para o escalonamento, ou seja, o scaler vai calcular a média e o desvio padrão dos dados de treino.

# Transformando os dados de treino e teste
x_train_standard_scaled = standardScaler.transform(X_train) # Transformando os dados de treino, para a base de treino
x_test_standard_scaled  = standardScaler.transform(X_test) # Transformando os dados de teste, para a base de teste
# Verificando os dados padronizados
x_train_standard_scaled


# Utilizando um algoritimo de classificacao para verificar a diferenca entre eles
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3) # Criando o modelo de KNN com 3 vizinhos mais próximos

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)



from sklearn.metrics import accuracy_score

# Treinando com Normalização
model_min_max = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_min_max.fit(x_train_min_max_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_min_max = model.predict(x_test_min_max_scaled)

accuracy_min_max = accuracy_score(y_test, y_pred_min_max)
print(f'Acurácia: {accuracy_min_max:.2f}')
# Acurácia: 0.81


# Treinando com Padronização
model_standard = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_standard.fit(x_train_standard_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_standard = model.predict(x_test_standard_scaled)

accuracy_strandard = accuracy_score(y_test, y_pred_standard)
print(f'Acurácia: {accuracy_strandard:.2f}')
# Acurácia: 0.81