import matplotlib.pyplot as plt       # Importa o módulo pyplot da biblioteca matplotlib
import numpy as np                    # Importa numpy para trabalhar com arrays e funções matemáticas

# Dados de exemplo para gráficos de linha
x = np.linspace(0, 10, 50)            # Cria 50 valores entre 0 e 10 igualmente espaçados
y = np.sin(x)                         # Calcula o seno dos valores de x
y2 = np.cos(x)                        # Calcula o cosseno dos valores de x

# --- 1. Gráfico de Linha ---
plt.figure(figsize=(8, 4))            # Cria uma nova figura com tamanho definido
plt.plot(x, y, label='Seno', color='blue')             # Plota a curva do seno com legenda
plt.plot(x, y2, label='Cosseno', color='orange', linestyle='--')  # Plota a curva do cosseno
plt.title('Seno e Cosseno')          # Título do gráfico
plt.xlabel('X')                      # Rótulo do eixo X
plt.ylabel('Valor')                  # Rótulo do eixo Y
plt.legend()                         # Exibe a legenda
plt.grid(True)                       # Adiciona a grade no gráfico
plt.show()                           # Mostra o gráfico na tela

# --- 2. Gráfico de Barras ---
categorias = ['A', 'B', 'C', 'D']     # Lista de categorias
valores = [10, 15, 7, 12]             # Valores correspondentes

plt.figure(figsize=(6, 4))            # Cria uma nova figura
plt.bar(categorias, valores, color='skyblue')  # Plota gráfico de barras com cor azul clara
plt.title('Gráfico de Barras')        # Título do gráfico
plt.xlabel('Categorias')              # Rótulo do eixo X
plt.ylabel('Valores')                 # Rótulo do eixo Y
plt.show()                            # Mostra o gráfico

# --- 3. Histograma ---
dados = np.random.randn(1000)         # Gera 1000 números aleatórios com distribuição normal

plt.figure(figsize=(6, 4))            # Cria nova figura
plt.hist(dados, bins=30, color='purple', edgecolor='black')  # Histograma com 30 faixas (bins)
plt.title('Histograma')               # Título do gráfico
plt.xlabel('Valor')                   # Rótulo do eixo X
plt.ylabel('Frequência')              # Rótulo do eixo Y
plt.show()                            # Mostra o gráfico

# --- 4. Gráfico de Dispersão (Scatter) ---
x = np.random.rand(50)                # Gera 50 valores aleatórios entre 0 e 1 para x
y = np.random.rand(50)                # Gera 50 valores aleatórios para y

plt.figure(figsize=(6, 4))            # Cria nova figura
plt.scatter(x, y, color='green')      # Plota os pontos (x, y) na cor verde
plt.title('Gráfico de Dispersão')     # Título do gráfico
plt.xlabel('X')                       # Rótulo do eixo X
plt.ylabel('Y')                       # Rótulo do eixo Y
plt.show()                            # Mostra o gráfico

# --- 5. Gráfico de Pizza (Pie Chart) ---
labels = ['Maçã', 'Banana', 'Laranja', 'Uva']           # Rótulos das fatias
valores = [30, 25, 20, 25]                              # Porcentagem de cada fruta

plt.figure(figsize=(6, 6))              # Cria uma figura quadrada para pizza
plt.pie(valores, labels=labels, autopct='%1.1f%%', startangle=140)  # Gera gráfico de pizza
plt.title('Distribuição de Frutas')     # Título do gráfico
plt.show()                              # Mostra o gráfico
