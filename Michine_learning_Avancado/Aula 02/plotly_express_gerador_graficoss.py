# visualizacao_plotly_express.py

import plotly.express as px
import pandas as pd

# Carrega o dataset de exemplo (não precisa baixar nada)
df = px.data.tips()

# Exibe as primeiras linhas do dataset
print("Preview do dataset:")
print(df.head())

# --- 1. Gráfico de Dispersão (Scatter) ---
fig1 = px.scatter(
    df,
    x='total_bill',
    y='tip',
    color='sex',          # Cor por sexo
    size='size',          # Tamanho do ponto pela quantidade de pessoas
    hover_data=['day'],   # Mostra o dia no hover
    title='Total da Conta vs Gorjeta'
)
fig1.show()

# --- 2. Histograma ---
fig2 = px.histogram(
    df,
    x='total_bill',
    nbins=20,
    color='sex',
    title='Distribuição do Total da Conta'
)
fig2.show()

# --- 3. Boxplot ---
fig3 = px.box(
    df,
    x='day',
    y='tip',
    color='sex',
    title='Distribuição das Gorjetas por Dia da Semana'
)
fig3.show()

# --- 4. Barplot (média por categoria) ---
media_df = df.groupby('day')['total_bill'].mean().reset_index()

fig4 = px.bar(
    media_df,
    x='day',
    y='total_bill',
    title='Média do Total da Conta por Dia'
)
fig4.show()

# --- 5. Pizza (Pie chart) ---
fumantes_df = df['smoker'].value_counts().reset_index()
fumantes_df.columns = ['smoker', 'count']

fig5 = px.pie(
    fumantes_df,
    names='smoker',
    values='count',
    title='Distribuição de Fumantes no Restaurante'
)
fig5.show()

# --- 6. Gráfico de Linha (Line chart) ---
# Simula um crescimento de vendas fictício
vendas = pd.DataFrame({
    'mes': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai'],
    'valor': [100, 150, 200, 180, 220]
})

fig6 = px.line(
    vendas,
    x='mes',
    y='valor',
    title='Crescimento Mensal das Vendas (Simulado)'
)
fig6.show()

# --- 7. Gráfico de Violino (Violin plot) ---
fig7 = px.violin(
    df,
    x='day',
    y='tip',
    color='sex',
    box=True,       # Exibe boxplot dentro do violino
    points='all',   # Mostra todos os pontos individuais
    title='Gorjetas por Dia e Sexo - Violin Plot'
)
fig7.show()
    