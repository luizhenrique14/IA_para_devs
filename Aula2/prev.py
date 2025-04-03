# Importa o modelo LinearSVC da biblioteca scikit-learn, que será usado para criar um classificador baseado em máquinas de vetores de suporte (SVM).
from sklearn.svm import LinearSVC

# Define os dados de entrada (features) para o treinamento. Cada lista representa um "composto" com características binárias (0 ou 1).
composto1 = [1,1,1]
composto2 = [0,0,0]
composto3 = [1,0,1]
composto4 = [0,1,0]
composto5 = [1,1,0]
composto6 = [0,0,1]

# Agrupa os compostos em uma lista chamada `dados_treino` e define os rótulos correspondentes em `rotulos_treino`.
# Os rótulos indicam a classificação de cada composto: 'S' (Sim) ou 'N' (Não).
dados_treino = [composto1, composto2, composto3, composto4, composto5, composto6]
rotulos_treino = ['S','N','S','N','S','S']

# Cria uma instância do modelo LinearSVC, que será usado para treinar e fazer previsões.
# O parâmetro `dual=False` é usado para evitar o aviso de mudança futura no scikit-learn.
modelo = LinearSVC(dual=False)

# Treina o modelo usando os dados de treinamento (`dados_treino`) e os rótulos (`rotulos_treino`).
# O modelo aprende a relação entre os dados de entrada e os rótulos.
modelo.fit(dados_treino, rotulos_treino)

# Define os dados de teste, que são novos compostos para os quais queremos prever os rótulos.
# Cada lista representa um novo composto com características binárias.
teste1 = [1,0,0]
teste2 = [0,1,1]
teste3 = [1,0,1]

# Agrupa os dados de teste em uma lista chamada `dados_teste`.
dados_teste = [teste1, teste2, teste3]

# Usa o modelo treinado para prever os rótulos dos dados de teste.
# O resultado será uma lista de rótulos previstos para cada composto em `dados_teste`.
previsoes = modelo.predict(dados_teste)

# Cria um dicionário para mapear os rótulos 'S' e 'N' para suas representações completas ('Sim' e 'Não').
mapeamento_de_previsao = { 'S': 'Sim', 'N': 'Não' }

# Exibe as previsões feitas pelo modelo para os dados de teste.
print("Previsões dos modelos de dados:", previsoes)

# Itera sobre as previsões e exibe o composto correspondente e sua previsão mapeada.
for i, previsao in enumerate(previsoes):
    print(f"Composto: {i+1} | Previsão: {mapeamento_de_previsao[previsao]}")