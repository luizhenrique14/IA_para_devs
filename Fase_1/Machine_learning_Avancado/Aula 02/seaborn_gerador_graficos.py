# visualizacao_seaborn.py

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Carregar dataset de exemplo: dados de gorjetas em restaurante
df = sns.load_dataset("tips")

# Exibe as primeiras linhas do dataframe
print("Preview do dataset:")
print(df.head())

# --- 1. Histograma com curva de densidade (KDE) ---
plt.figure(figsize=(8, 4))
sns.histplot(df['total_bill'], kde=True)
plt.title("Distribuição de Total da Conta")
plt.xlabel("Total da Conta")
plt.ylabel("Frequência")
plt.tight_layout()
plt.show()

# --- 2. Gráfico de dispersão (scatterplot) ---
plt.figure(figsize=(8, 4))
sns.scatterplot(data=df, x='total_bill', y='tip')
plt.title("Relação entre Total da Conta e Gorjeta")
plt.xlabel("Total da Conta")
plt.ylabel("Gorjeta")
plt.tight_layout()
plt.show()

# --- 3. Boxplot (distribuição por categoria) ---
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='day', y='tip')
plt.title("Distribuição de Gorjetas por Dia da Semana")
plt.xlabel("Dia")
plt.ylabel("Gorjeta")
plt.tight_layout()
plt.show()

# --- 4. Barplot (média por categoria) ---
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x='day', y='total_bill', estimator='mean')
plt.title("Média de Total da Conta por Dia")
plt.xlabel("Dia")
plt.ylabel("Total da Conta (Média)")
plt.tight_layout()
plt.show()

# --- 5. Heatmap de correlação ---
plt.figure(figsize=(6, 5))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Mapa de Correlação entre Variáveis")
plt.tight_layout()
plt.show()

# --- 6. Pairplot (relacionamentos entre variáveis) ---
sns.pairplot(df, hue='sex')  # Cor por sexo
plt.suptitle("Relações entre Variáveis Numéricas", y=1.02)
plt.show()
