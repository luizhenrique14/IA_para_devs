import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Exemplo de DataFrame com dados nulos
dados = pd.DataFrame({
    'nome': ['João', 'Maria', 'Pedro', None, 'Ana'],
    'idade': [25, None, 30, 22, None],
    'cidade': ['SP', 'RJ', None, 'MG', 'BA']
})

# 1. Visualiza um resumo com contagem de dados não nulos por coluna
msno.bar(dados)
plt.show()  # Exibe o gráfico

# 2. Matriz que mostra onde estão os nulos por linha
msno.matrix(dados)
plt.show()

# 3. Heatmap que mostra correlação entre os nulos (se colunas faltam juntas)
msno.heatmap(dados)
plt.show()

# 4. Dendrograma (agrupa colunas com padrão similar de nulos)
msno.dendrogram(dados)
plt.show()

# Agora, você pode tomar decisões com base nisso. Exemplos de tratamento:

# a) Remover linhas com qualquer valor nulo
dados_sem_nulos = dados.dropna()

# b) Preencher valores nulos com um valor fixo (ex: "Desconhecido" ou média)
dados['nome'] = dados['nome'].fillna('Desconhecido')
dados['idade'] = dados['idade'].fillna(dados['idade'].mean())  # Média das idades
dados['cidade'] = dados['cidade'].fillna('Não informado')

# Mostrar como ficou depois do tratamento
print(dados)
