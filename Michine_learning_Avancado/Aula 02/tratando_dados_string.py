# ------------------------------------------------------------------------------
# Este script realiza o pré-processamento de uma base de dados para Machine Learning.
# Passo a passo do que o código faz:
# 1. Carrega os dados de um arquivo Excel para um DataFrame do pandas.
# 2. Exibe informações iniciais sobre os dados (primeiras linhas, formato, valores únicos, estatísticas).
# 3. Codifica colunas categóricas ('gender', 'workex', 'specialisation', 'status', 'hsc_s', 'degree_t') usando LabelEncoder,
#    ou seja, transforma texto/categorias em números inteiros.
# 4. Remove a coluna 'salary' do DataFrame final (caso exista).
# 5. Exibe o resultado final do DataFrame pronto para uso em modelos de Machine Learning e gera o gráfico de correlação.
# ------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import seaborn as sb

# Lê a base de dados do arquivo Excel
dados = pd.read_excel('Michine_learning_Avancado\Aula 02\Recrutamento.xlsx')
print("✅ Dados carregados do Excel:")
print(dados.head())

# Identifica automaticamente todas as colunas do tipo 'object' (texto/categórica)
colunas_categoricas = dados.select_dtypes(include=['object']).columns.tolist()
print(f"\n🔎 Colunas categóricas detectadas: {colunas_categoricas}")

# Para cada coluna categórica, aplica o LabelEncoder para transformar em números inteiros
label_encoder = LabelEncoder()
for col in colunas_categoricas:
    dados[col] = label_encoder.fit_transform(dados[col])
    print(f"✅ Transformação concluída para coluna: {col} (categorias convertidas para números)")

print("\n🔎 Dados após codificação de todas as colunas categóricas:")
print(dados.head())

# Remove a coluna 'salary' se existir, pois não será usada na análise de correlação
if 'salary' in dados.columns:
    dados.drop(['salary'], axis=1, inplace=True)
    print("\n✅ Coluna 'salary' removida.")

# Agora todas as colunas são numéricas e podem ser usadas na matriz de correlação
correlation_matrix = dados.corr().round(2)

# Gera e exibe o gráfico de correlação entre as variáveis
fig, ax = plt.subplots(figsize=(12,12))    
sb.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)

plt.show()