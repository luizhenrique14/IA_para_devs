# Importa o modelo LinearSVC da biblioteca scikit-learn para criar um classificador baseado em máquinas de vetores de suporte (SVM).
# Também importa a função accuracy_score para calcular a taxa de acerto do modelo.
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Define os dados de entrada (features) para o treinamento. Cada lista representa um composto com características binárias (0 ou 1).
composto1 = [1,1,1]
composto2 = [0,0,0]
composto3 = [1,0,1]
composto4 = [0,1,0]
composto5 = [1,1,0]
composto6 = [0,0,1]

# Agrupa os compostos em uma lista chamada `dados_treino` e define os rótulos correspondentes em `rotulos_treino`.
# Os rótulos indicam a classificação de cada composto: 'S' (Sim) ou 'N' (Não).
dados_treino = [composto1, composto2, composto3, composto4, composto5, composto6]
rotulos_treino = ['S', 'N', 'S', 'N', 'S', 'S']

# Cria uma instância do modelo LinearSVC, que será usado para treinar e fazer previsões.
# O parâmetro `dual=False` é configurado para evitar avisos de mudanças futuras no scikit-learn.
modelo = LinearSVC(dual=False)

# Treina o modelo usando os dados de treinamento (`dados_treino`) e os rótulos (`rotulos_treino`).
# O modelo aprende a relação entre os dados de entrada e os rótulos.
modelo.fit(dados_treino, rotulos_treino)

# Define os dados de teste, que são compostos novos para os quais queremos prever os rótulos.
# Cada lista representa um novo composto com características binárias.
teste1 = [1,0,0]
teste2 = [0,1,1]
teste3 = [1,0,1]

# Agrupa os dados de teste em uma lista chamada `dados_teste` e define os rótulos reais correspondentes em `rotulos_teste`.
# Os rótulos reais serão usados para comparar com as previsões do modelo.
dados_teste = [teste1, teste2, teste3]
dados_que_irei_testar = ['S', 'N', 'S']

# Usa o modelo treinado para prever os rótulos dos dados de teste.
# O resultado será uma lista de rótulos previstos para cada composto em `dados_teste`.
previsoes = modelo.predict(dados_teste)

# Calcula a taxa de acerto do modelo comparando os rótulos previstos (`previsoes`) com os rótulos reais (`rotulos_teste`).
taxa_acerto = accuracy_score(dados_que_irei_testar, previsoes)

# Exibe a taxa de acerto do modelo em porcentagem.
print("Taxa de acerto:", (taxa_acerto * 100), "%")