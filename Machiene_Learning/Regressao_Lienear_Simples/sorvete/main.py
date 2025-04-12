# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



###########################################################################################################################################################################

# Importando a base de dados
dados = pd.read_excel("Sorvete.xlsx")  # Lê o arquivo Excel contendo os dados e armazena em um DataFrame chamado 'dados'
dados.head()  # Exibe as primeiras 5 linhas do DataFrame para verificar se os dados foram carregados corretamente

# Visualizando os dados
plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])  # Cria um gráfico de dispersão (scatter plot) para visualizar a relação entre Temperatura e Vendas de Sorvetes
plt.xlabel('Temperatura (°C)')  # Define o rótulo do eixo X como "Temperatura (°C)"
plt.ylabel('Vendas de Sorvetes (milhares)')  # Define o rótulo do eixo Y como "Vendas de Sorvetes (milhares)"
plt.title('Relação entre Temperatura e Vendas de Sorvetes')  # Adiciona um título ao gráfico
plt.show()  # Exibe o gráfico na tela


dados.corr()


###########################################################################################################################################################################

# Dividindo os dados em conjuntos de treinamento e teste
X = dados[['Temperatura']]  # Recurso (variável independente) - o que usamos para prever
y = dados['Vendas_Sorvetes']  # Rótulo (variável dependente) - o que queremos prever

# O método train_test_split divide os dados em dois conjuntos:
# - Conjunto de treinamento (usado para treinar o modelo)
# - Conjunto de teste (usado para avaliar o modelo)
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Dados de entrada (variável independente)
    y,  # Dados de saída (variável dependente)
    test_size=0.2,  # Proporção dos dados reservados para teste (20%)
    random_state=42  # Semente para garantir que a divisão seja reproduzível
)


X_train.shape  # Exibe a forma (dimensões) do conjunto de treinamento de dados de entrada

X_test.shape  # Exibe a forma (dimensões) do conjunto de teste de dados de entrada


###########################################################################################################################################################################

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression() # Cria uma instância do modelo de regressão linear
modelo.fit(X_train, y_train) # Treina o modelo de regressão linear usando os dados de treinamento (X_train e y_train)

# Fazendo previsões no conjunto de teste
previsoes = modelo.predict(X_test)  # Faz previsões usando o modelo treinado no conjunto de teste

###########################################################################################################################################################################

# Avaliando os resultados ✅

#ERROS


# Calcula o Erro Médio Quadrático (MSE - Mean Squared Error), que é a média dos quadrados das diferenças 
# entre os valores reais (y_test) e os valores previstos (previsoes).
# Um valor menor indica que o modelo está mais próximo dos valores reais.
erro_medio_quadratico = mean_squared_error(y_test, previsoes)

# Calcula o Erro Médio Absoluto (MAE - Mean Absolute Error), que é a média das diferenças absolutas 
# entre os valores reais e previstos. Um valor pequeno para MAE significa que as previsões estão próximas dos valores reais.
erro_absoluto_medio = mean_absolute_error(y_test, previsoes)

# Calcula o coeficiente de determinação (R² - R-squared), que mede o ajuste geral do modelo.
# O valor de R² varia entre 0 e 1, onde valores mais próximos de 1 indicam que o modelo explica melhor a variabilidade dos dados.
r_quadrado = r2_score(y_test, previsoes)

# Exibindo as métricas de avaliação
print(f'Erro Médio Quadrático: {erro_medio_quadratico}')  # Exibe o erro médio quadrático
print(f'Erro Absoluto Médio: {erro_absoluto_medio}')  # Exibe o erro absoluto médio
print(f'R² (coeficiente de determinação): {r_quadrado}')  # Exibe o coeficiente de determinação (R²)



#PREVISOES

# Visualizando as previsões
plt.scatter(X_test, y_test, label='Real')  # Cria um gráfico de dispersão para os valores reais (dados de teste)
plt.scatter(X_test, previsoes, label='Previsto', color='red')  # Cria um gráfico de dispersão para os valores previstos pelo modelo
plt.xlabel('Temperatura (°C)')  # Define o rótulo do eixo X como "Temperatura (°C)"
plt.ylabel('Vendas de Sorvetes (milhares)')  # Define o rótulo do eixo Y como "Vendas de Sorvetes (milhares)"
plt.title('Previsões do Modelo de Regressão Linear')  # Adiciona um título ao gráfico
plt.legend()  # Adiciona uma legenda para diferenciar os valores reais e previstos
plt.show()  # Exibe o gráfico na tela