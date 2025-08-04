import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from haversine import haversine, Unit
import tkinter as tk
from tkinter import ttk
from bairros_sp import BAIRROS_COORDENADAS

# 🏠 PONTO DE PARTIDA DE TODOS OS DRONES (ex: Centro de SP)
BASE_COORDENADA = (-23.5489, -46.6388)  # Sé

# 🚚 CLASSE ROTA
class Rota:
    def __init__(self, pontos, autonomia, mutacao):
        self.base = BASE_COORDENADA
        self.pontos = pontos[:]
        self.autonomia = autonomia
        self.mutacao = mutacao
        random.shuffle(self.pontos)
        self.distancia_total = self.calcular_distancia()

    def calcular_distancia(self):
        dist = 0
        pontos_rota = [self.base] + self.pontos + [self.base]
        for i in range(len(pontos_rota) - 1):
            dist += haversine(pontos_rota[i], pontos_rota[i+1], unit=Unit.KILOMETERS)
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

# 🧠 ALGORITMO GENÉTICO
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

# 🖼️ VISUALIZADOR
class Visualizador:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulação de Entrega com Drones")

        self.status = tk.Label(root, text="Configure os parâmetros e clique em iniciar", fg="blue")
        self.status.pack(pady=5)
        self.frame_controls = tk.Frame(root)
        self.frame_controls.pack(pady=5)

        self.start_button = ttk.Button(self.frame_controls, text="▶ Iniciar", command=self.iniciar)
        self.start_button.grid(row=0, column=0, padx=5)
        self.pause_button = ttk.Button(self.frame_controls, text="⏸ Pausar", command=self.pausar)
        self.pause_button.grid(row=0, column=1, padx=5)
        self.reset_button = ttk.Button(self.frame_controls, text="🔄 Reiniciar", command=self.reiniciar)
        self.reset_button.grid(row=0, column=2, padx=5)

        labels = ["Drones", "Autonomia (km)", "População", "Gerações", "Mutação"]
        defaults = [2, 30, 50, 100, 0.1]
        self.entries = {}
        for i, (label, default) in enumerate(zip(labels, defaults), start=1):
            tk.Label(self.frame_controls, text=label + ":").grid(row=i, column=0)
            entry = ttk.Spinbox(self.frame_controls, from_=1, to=1000, width=5, increment=1 if isinstance(default, int) else 0.01)
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

        self.rodando = False
        self.ani = None

    def dividir_bairros_exclusivos(self, coordenadas, qtd_drones):
        random.shuffle(coordenadas)
        chunk = len(coordenadas) // qtd_drones
        partes = [coordenadas[i*chunk:(i+1)*chunk] for i in range(qtd_drones)]
        resto = coordenadas[qtd_drones*chunk:]
        for i, ponto in enumerate(resto):
            partes[i % qtd_drones].append(ponto)
        return partes

    def iniciar(self):
        if not self.rodando:
            qtd_drones = int(self.entries["Drones"].get())
            self.autonomia = float(self.entries["Autonomia (km)"].get())
            populacao = int(self.entries["População"].get())
            self.geracoes = int(self.entries["Gerações"].get())
            self.mutacao = float(self.entries["Mutação"].get())

            self.partes = self.dividir_bairros_exclusivos(BAIRROS_COORDENADAS.copy(), qtd_drones)
            self.algoritmos = [AlgoritmoGenetico(parte, self.autonomia, populacao, self.mutacao) for parte in self.partes]
            self.historico = [[] for _ in range(qtd_drones)]

            self.rodando = True
            self.ani = animation.FuncAnimation(self.fig, self.update_plot, frames=self.geracoes, interval=500, repeat=False)
            self.canvas_fig.draw()
            plt.show(block=False)
            self.status.config(text="Simulação em andamento...", fg="green")

    def pausar(self):
        if self.ani and self.ani.event_source:
            self.ani.event_source.stop()
            self.status.config(text="Simulação pausada", fg="orange")

    def reiniciar(self):
        self.ax_rota.clear()
        self.ax_evol.clear()
        self.rodando = False
        self.status.config(text="Simulação reiniciada", fg="blue")

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
            # Marcar ida e volta para BASE
            x, y = zip(*([BASE_COORDENADA] + rota.pontos + [BASE_COORDENADA]))
            self.ax_rota.plot(y, x, marker='o', linestyle='-', color=cores[idx % len(cores)], label=f'Drone {idx+1}')
            self.ax_rota.scatter(y, x, color=cores[idx % len(cores)], s=20)
            self.historico[idx].append(rota.distancia_total)
            self.ax_evol.plot(self.historico[idx], label=f'Drone {idx+1}', color=cores[idx % len(cores)])

        self.ax_rota.legend(loc='lower left', fontsize=8)
        self.ax_evol.legend(fontsize=8)
        self.canvas_fig.draw()

        if frame + 1 == self.geracoes:
            print("\\n📋 RELATÓRIO FINAL")
            print(f"Base (ponto de partida/chegada): {BASE_COORDENADA}")
            VELOCIDADE = 40  # km/h
            for i, ag in enumerate(self.algoritmos):
                rota = ag.melhor_rota
                dist = rota.distancia_total
                entregas = len(rota.pontos)
                voos = max(1, round(dist / self.autonomia, 1))
                tempo_horas = dist / VELOCIDADE
                print(f"Drone {i+1}:")
                print(f"  - Sequência de bairros (ordem):")
                for idx, ponto in enumerate(rota.pontos, 1):
                    print(f"     {idx:03}: {ponto}")
                print(f"  - Total de entregas: {entregas}")
                print(f"  - Distância total: {dist:.2f} km")
                print(f"  - Tempo estimado: {tempo_horas:.2f} horas (velocidade {VELOCIDADE} km/h)")
                print(f"  - Recarregamentos necessários: {int(voos) - 1}\\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = Visualizador(root)
    root.mainloop()
