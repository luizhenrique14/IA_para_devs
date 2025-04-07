import torch
import torch.nn as nn  # Biblioteca para criar redes neurais
import torch.optim as optim  # Biblioteca para ajustar os pesos da rede

# Dados de entrada que o modelo vai usar para aprender
dadosQueModeloUsaPraAprender = torch.tensor(
    [[5.0], [10.0], [10.0], [5.0], [10.0],
     [5.0], [10.0], [10.0], [5.0], [10.0],
     [5.0], [10.0], [10.0], [5.0], [10.0],
     [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

# Resultados esperados para os dados de entrada
resultadosEsperados = torch.tensor(
    [[30.5], [63.0], [67.0], [29.0], [62.0],
     [30.5], [63.0], [67.0], [29.0], [62.0],
     [30.5], [63.0], [67.0], [29.0], [62.0],
     [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

# Definição da rede neural
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Camada de entrada com 1 neurônio e camada oculta com 5 neurônios
        self.fc1 = nn.Linear(1, 5)
        # Camada de saída com 1 neurônio
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        # Passa os dados pela camada oculta com função de ativação ReLU
        x = torch.relu(self.fc1(x))
        # Passa pela camada de saída
        x = self.fc2(x)
        return x

# Criação do modelo
model = Net()

# Função de perda para calcular o erro entre previsão e resultado esperado
critetion = nn.MSELoss()
# Otimizador para ajustar os pesos da rede (usando SGD com taxa de aprendizado 0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento da rede neural
for epoch in range(1000):  # Treina por 1000 épocas
    optimizer.zero_grad()  # Zera os gradientes acumulados
    outputs = model(dadosQueModeloUsaPraAprender)  # Faz previsões com os dados de entrada
    loss = critetion(outputs, resultadosEsperados)  # Calcula o erro (perda)
    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os pesos da rede

    # A cada 100 épocas, imprime o valor da perda
    if epoch % 100 == 99:
        print(f'Epoch {epoch}, Perda: {loss.item()}')

# Teste da rede neural com novos dados
with torch.no_grad():  # Desativa o cálculo de gradientes para economizar memória
    predicoes = model(torch.tensor([[10.0]], dtype=torch.float32))  # Faz previsão com novos dados
    print(f'Previsão do tempo de conclusão: {predicoes.item()} minutos')  # Mostra o resultado