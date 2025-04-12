# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Importando a base de dados
dados = pd.read_excel("Sorvete.xlsx")
dados.head()

# Visualizando os dados
plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvetes (milhares)')
plt.title('Relação entre Temperatura e Vendas de Sorvetes')
plt.show()


dados.corr()