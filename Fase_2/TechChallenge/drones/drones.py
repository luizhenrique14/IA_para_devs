
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from haversine import haversine, Unit
import tkinter as tk
from tkinter import ttk
from bairros_sp import BAIRROS_COORDENADAS

BASE_COORDENADA = (-23.5505, -46.6333)  # Marco Zero de SP (Praça da Sé)

class Rota:
    def __init__(self, pontos, autonomia, mutacao):
        self.entregas = pontos[:]
        self.autonomia = autonomia
        self.mutacao = mutacao
        random.shuffle(self.entregas)
        self.pontos = [BASE_COORDENADA] + self.entregas + [BASE_COORDENADA]
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
        meio = len(self.entregas) // 2
        filho_entregas = self.entregas[:meio] + [p for p in outra.entregas if p not in self.entregas[:meio]]
        return Rota(filho_entregas, self.autonomia, self.mutacao)

    def mutar(self):
        if random.random() < self.mutacao:
            i, j = random.sample(range(len(self.entregas)), 2)
            self.entregas[i], self.entregas[j] = self.entregas[j], self.entregas[i]
            self.pontos = [BASE_COORDENADA] + self.entregas + [BASE_COORDENADA]
            self.distancia_total = self.calcular_distancia()

class AlgoritmoGenetico:
    def __init__(self, coordenadas, autonomia, populacao_size, mutacao):
        self.entregas = coordenadas
        self.autonomia = autonomia
        self.populacao = [Rota(coordenadas, autonomia, mutacao) for _ in range(populacao_size)]
        self.melhor_rota = min(self.populacao, key=lambda r: r.distancia_total)
        self.populacao_size = populacao_size
        self.mutacao = mutacao

    def evoluir(self):
        nova_populacao = []
        for _ in range(self.populacao_size):
            pais = random.sample(self.populacao, 2)
            filho = pais[0].crossover(pais[1])
            filho.mutar()
            nova_populacao.append(filho)
        self.populacao = nova_populacao
        melhor = min(self.populacao, key=lambda r: r.distancia_total)
        if melhor.distancia_total < self.melhor_rota.distancia_total:
            self.melhor_rota = melhor
        return self.melhor_rota

class Visualizador:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulação de Entrega com Drones")

        self.status = tk.Label(root, text="Configure os parâmetros e clique em iniciar", fg="blue")
        self.status.pack(pady=5)
        self.frame_controls = tk.Frame(root)
        self.frame_controls.pack(pady=5)

        labels = ["Drones", "Autonomia (km)", "População", "Gerações", "Mutação"]
        defaults = [3, 30, 80, 150, 0.1]
        self.entries = {}
        for i, (label, default) in enumerate(zip(labels, defaults)):
            tk.Label(self.frame_controls, text=label + ":").grid(row=i, column=0)
            entry = ttk.Spinbox(self.frame_controls, from_=1, to=1000, width=6, increment=1 if isinstance(default, int) else 0.01)
            entry.set(default)
            entry.grid(row=i, column=1)
            self.entries[label] = entry

        self.fig = plt.figure(figsize=(12, 6))
        self.ax_rota = self.fig.add_subplot(121)
        self.ax_evol = self.fig.add_subplot(122)
        self.fig.subplots_adjust(wspace=0.3)
        self.canvas_fig = plt.get_current_fig_manager().canvas

        try:
            self.bg_img = mpimg.imread("mapa_sp.png")
        except:
            self.bg_img = None

    def dividir_entregas(self, coordenadas, qtd_drones):
        random.shuffle(coordenadas)
        chunk = len(coordenadas) // qtd_drones
        partes = [coordenadas[i*chunk:(i+1)*chunk] for i in range(qtd_drones)]
        resto = coordenadas[qtd_drones*chunk:]
        for i, ponto in enumerate(resto):
            partes[i % qtd_drones].append(ponto)
        return partes

    def iniciar(self):
        qtd_drones = int(self.entries["Drones"].get())
        self.autonomia = float(self.entries["Autonomia (km)"].get())
        populacao = int(self.entries["População"].get())
        self.geracoes = int(self.entries["Gerações"].get())
        self.mutacao = float(self.entries["Mutação"].get())

        self.partes = self.dividir_entregas(BAIRROS_COORDENADAS.copy(), qtd_drones)
        self.algoritmos = [AlgoritmoGenetico(parte, self.autonomia, populacao, self.mutacao) for parte in self.partes]
        self.historico = [[] for _ in range(qtd_drones)]

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=self.geracoes, interval=500, repeat=False)
        self.canvas_fig.draw()
        plt.show(block=False)
        self.status.config(text="Simulação em andamento...", fg="green")

    def update_plot(self, frame):
        self.ax_rota.clear()
        self.ax_evol.clear()
        self.ax_rota.set_title(f"Rotas - Geração {frame + 1}")
        self.ax_evol.set_title("Evolução da Rota (km)")
        self.ax_evol.set_xlabel("Geração")
        self.ax_evol.set_ylabel("Distância")

        if self.bg_img is not None:
            self.ax_rota.imshow(self.bg_img, extent=[-46.83, -46.35, -23.75, -23.40], alpha=0.4)

        cores = ["blue", "red", "green", "purple", "orange", "black"]
        for idx, ag in enumerate(self.algoritmos):
            rota = ag.evoluir()
            x, y = zip(*rota.pontos)
            self.ax_rota.plot(y, x, marker='o', linestyle='-', color=cores[idx % len(cores)], label=f'Drone {idx+1}')
            self.ax_rota.scatter(y, x, color=cores[idx % len(cores)], s=20)
            self.historico[idx].append(rota.distancia_total)
            self.ax_evol.plot(self.historico[idx], label=f'Drone {idx+1}', color=cores[idx % len(cores)])

        self.ax_rota.legend(loc='lower left', fontsize=8)
        self.ax_evol.legend(fontsize=8)
        self.canvas_fig.draw()

        if frame + 1 == self.geracoes:
            print("\n📋 RELATÓRIO FINAL")
            for i, ag in enumerate(self.algoritmos):
                rota = ag.melhor_rota
                entregas = rota.entregas
                print(f"\n🚁 Drone {i+1}:")
                print(f"  Total de entregas: {len(entregas)}")
                print(f"  Distância total percorrida: {rota.distancia_total:.2f} km")
                voos = max(1, round(rota.distancia_total / self.autonomia, 1))
                print(f"  Recarregamentos necessários: {int(voos) - 1}")
                print(f"  Sequência de entregas:")
                for idx_bairro, ponto in enumerate(entregas, start=1):
                    print(f"    {idx_bairro}. {ponto}")

# Execução
if __name__ == "__main__":
    root = tk.Tk()
    app = Visualizador(root)
    root.mainloop()
