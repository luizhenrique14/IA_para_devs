import pandas as pd

# Exemplo de tratamento de dados categóricos para números
# -------------------------------------------------------

#  Lendo a base de dados
print("\n Lendo a base de dados...")
dados = pd.read_excel('Michine_learning_Avancado\Aula 02\Recrutamento.xlsx')
print("Formato dos dados:", dados.shape)
print("Primeiras linhas do DataFrame:")
print(dados.head(3))

# 🏷️ 1. LabelEncoder para colunas categóricas binárias ou ordinais
from sklearn.preprocessing import LabelEncoder
colunas_label = ['gender', 'workex', 'specialisation', 'status']
label_encoder = LabelEncoder()
for col in colunas_label:
    print(f"🏷️ Aplicando LabelEncoder na coluna '{col}' (transformando string em número)...")
    dados[col] = label_encoder.fit_transform(dados[col])
print("Exemplo após LabelEncoder:")
print(dados[colunas_label].head())

# 🏷️➡️🔢 2. One Hot Encoding para colunas categóricas nominais com mais de duas categorias
print("\n🏷️➡️🔢 Aplicando One Hot Encoding em 'hsc_s' e 'degree_t' (criando variáveis dummies)...")
dados = pd.get_dummies(dados, columns=['hsc_s', 'degree_t'], prefix=['hsc', 'degree'])
print("Colunas após One Hot Encoding:")
print([col for col in dados.columns if col.startswith('hsc_') or col.startswith('degree_')])

# ✅ ✅gAgara tosass colunas categóricas estãoãe  fofmamo numéréco!l!rint("\f✅ ✅✅TToaas asunas categóricasai tammsmmvlavvrriid dppos  úúmrro"."print("PrPPrirarrssllahhssddataFrame final:"::
print(dados.head(3))