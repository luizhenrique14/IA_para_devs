
# 📦 IMPORTAÇÕES E PARÂMETROS INICIAIS
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from haversine import haversine, Unit
import tkinter as tk
from tkinter import ttk, messagebox

from bairros_sp import BAIRROS_COORDENADAS  # Importa 300 bairros reais de São Paulo

# 🚚 CLASSE ROTA - Representa um indivíduo (possível solução)
class Rota:
    def __init__(self, pontos, autonomia, mutacao):
        self.pontos = pontos[:]
        self.autonomia = autonomia
        self.mutacao = mutacao
        random.shuffle(self.pontos)
        self.distancia_total = self.calcular_distancia()

    def calcular_distancia(self):
        dist = 0
        for i in range(len(self.pontos) - 1):
            dist += haversine(self.pontos[i], self.pontos[i+1], unit=Unit.KILOMETERS)
        return dist

    def fitness(self):
        if self.distancia_total > self.autonomia:
            return 1 / (self.distancia_total * 10)
        return 1 / self.distancia_total

    def crossover(self, outra):
        meio = len(self.pontos) // 2
        filho_pontos = self.pontos[:meio] + [p for p in outra.pontos if p not in self.pontos[:meio]]
        return Rota(filho_pontos, self.autonomia, self.mutacao)

    def mutar(self):
        if random.random() < self.mutacao:
            i, j = random.sample(range(len(self.pontos)), 2)
            self.pontos[i], self.pontos[j] = self.pontos[j], self.pontos[i]
            self.distancia_total = self.calcular_distancia()

# 🧠 CLASSE ALGORITMO GENÉTICO
class AlgoritmoGenetico:
    def __init__(self, coordenadas, autonomia, populacao_size, mutacao):
        self.coordenadas = coordenadas
        self.autonomia = autonomia
        self.populacao = [Rota(coordenadas, autonomia, mutacao) for _ in range(populacao_size)]
        self.melhor_rota = min(self.populacao, key=lambda r: r.distancia_total)
        self.populacao_size = populacao_size
        self.mutacao = mutacao

    def evoluir(self):
        nova_populacao = []
        for i in range(self.populacao_size):
            pais = random.sample(self.populacao, 2)
            filho = pais[0].crossover(pais[1])
            filho.mutar()
            nova_populacao.append(filho)
        self.populacao = nova_populacao
        melhor_geracao = min(self.populacao, key=lambda r: r.distancia_total)
        if melhor_geracao.distancia_total < self.melhor_rota.distancia_total:
            self.melhor_rota = melhor_geracao
        return self.melhor_rota

# 🖼️ CLASSE DE VISUALIZAÇÃO E CONTROLE (Tkinter + Matplotlib)
class Visualizador:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulação de Entrega com Drones - AG")

        self.status = tk.Label(self.root, text="Configure os parâmetros e clique em iniciar", fg="blue")
        self.status.pack(pady=5)

        self.frame_controls = tk.Frame(self.root)
        self.frame_controls.pack(pady=5)

        self.start_button = ttk.Button(self.frame_controls, text="▶ Iniciar", command=self.iniciar)
        self.start_button.grid(row=0, column=0, padx=5)

        self.pause_button = ttk.Button(self.frame_controls, text="⏸ Pausar", command=self.pausar)
        self.pause_button.grid(row=0, column=1, padx=5)

        self.reset_button = ttk.Button(self.frame_controls, text="🔄 Reiniciar", command=self.reiniciar)
        self.reset_button.grid(row=0, column=2, padx=5)

        # Campos para parâmetros
        labels = ["Drones", "Autonomia (km)", "População", "Gerações", "Mutação"]
        defaults = [2, 30, 50, 100, 0.1]
        self.entries = {}
        for i, (label, default) in enumerate(zip(labels, defaults), start=1):
            tk.Label(self.frame_controls, text=label + ":").grid(row=i, column=0)
            entry = ttk.Spinbox(self.frame_controls, from_=1, to=1000, width=5, increment=1 if isinstance(default, int) else 0.01)
            entry.set(default)
            entry.grid(row=i, column=1)
            self.entries[label] = entry

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(10, 6)
        self.fig.subplots_adjust(right=0.75)
        self.canvas_fig = plt.get_current_fig_manager().canvas

        self.rodando = False
        self.ani = None

    def iniciar(self):
        if not self.rodando:
            qtd_drones = int(self.entries["Drones"].get())
            self.autonomia = float(self.entries["Autonomia (km)"].get())
            populacao = int(self.entries["População"].get())
            self.geracoes = int(self.entries["Gerações"].get())
            self.mutacao = float(self.entries["Mutação"].get())

            coordenadas = BAIRROS_COORDENADAS
            partes = [coordenadas[i::qtd_drones] for i in range(qtd_drones)]
            self.algoritmos = [
                AlgoritmoGenetico(parte, self.autonomia, populacao, self.mutacao)
                for parte in partes
            ]

            self.rodando = True
            self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=self.geracoes, interval=500, repeat=False, 
                                               init_func=self.init_anim, blit=False)
            self.canvas_fig.draw()
            plt.show(block=False)
            self.status.config(text="Simulação em andamento...", fg="green")

    def init_anim(self):
        self.ax.clear()

    def pausar(self):
        if self.ani and self.ani.event_source:
            self.ani.event_source.stop()
            self.status.config(text="Simulação pausada", fg="orange")

    def reiniciar(self):
        self.ax.clear()
        self.rodando = False
        self.status.config(text="Simulação reiniciada", fg="blue")

    def update_plot(self, frame):
        self.ax.clear()
        self.ax.set_title(f"Geração {frame + 1}")
        cores = ["blue", "red", "green", "purple", "orange", "black", "cyan", "brown", "pink", "olive"]
        for idx, ag in enumerate(self.algoritmos):
            rota = ag.evoluir()
            x, y = zip(*rota.pontos)
            self.ax.plot(y, x, marker='o', color=cores[idx % len(cores)], label=f'Drone {idx+1}')
        self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
        self.canvas_fig.draw()

        # Geração final: exibir relatório
        if frame + 1 == self.geracoes:
            print("\n📋 RELATÓRIO FINAL")
            for i, ag in enumerate(self.algoritmos):
                rota = ag.melhor_rota
                entregas = len(rota.pontos)
                autonomia = self.autonomia
                distancia = rota.distancia_total
                viagens = max(1, round(distancia / autonomia, 1))
                print(f"Drone {i+1}:")
                print(f"  - Total de entregas: {entregas}")
                print(f"  - Distância total: {distancia:.2f} km")
                print(f"  - Entregas por voo (autonomia {autonomia} km): ~{entregas // viagens}")
                print(f"  - Recarregamentos necessários: {int(viagens) - 1}\n")

# 🚀 PONTO DE ENTRADA
if __name__ == "__main__":
    root = tk.Tk()
    app = Visualizador(root)
    root.mainloop()
