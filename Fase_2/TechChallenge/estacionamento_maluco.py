import tkinter as tk
from tkinter import messagebox
import random
import time
from copy import deepcopy

TAMANHO = 6

CARROS_INICIAIS = [
    ('A', [(0, 0), (0, 1)]),
    ('B', [(0, 3), (1, 3), (2, 3)]),
    ('X', [(2, 1), (2, 2)]),  # carro vermelho
    ('C', [(3, 0), (4, 0)]),
    ('D', [(3, 2), (3, 3), (3, 4)]),
    ('E', [(5, 3), (5, 4), (5, 5)]),
    ('F', [(1, 5), (2, 5)]),
]

CORES = {
    'X': 'red','A': 'lightblue','B': 'lightgreen','C': 'orange','D': 'violet','E': 'gold','.': 'SystemButtonFace'
}

DIRECOES = ['cima', 'baixo', 'esquerda', 'direita']

class EstacionamentoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Estacionamento Maluco")
        self.grade = [['.' for _ in range(TAMANHO)] for _ in range(TAMANHO)]
        self.botoes = []
        self.selecionado = None
        self.movimentos = 0
        self.tentativas = 0
        self.label_movimentos = tk.Label(self.root, text="Movimentos: 0", font=("Arial", 12))
        self.label_movimentos.grid(row=TAMANHO + 1, column=0, columnspan=6)
        self.label_tentativas = tk.Label(self.root, text="Tentativas: 0", font=("Arial", 12))
        self.label_tentativas.grid(row=TAMANHO + 2, column=0, columnspan=6)
        self.inicializar_grade()
        self.desenhar_interface()
        self.atualizar_interface()
        self.root.bind("<Up>", lambda e: self.mover_selecionado('cima'))
        self.root.bind("<Down>", lambda e: self.mover_selecionado('baixo'))
        self.root.bind("<Left>", lambda e: self.mover_selecionado('esquerda'))
        self.root.bind("<Right>", lambda e: self.mover_selecionado('direita'))

    def inicializar_grade(self):
        for carro, posicoes in CARROS_INICIAIS:
            for i, j in posicoes:
                self.grade[i][j] = carro

    def desenhar_interface(self):
        for i in range(TAMANHO):
            linha = []
            for j in range(TAMANHO):
                btn = tk.Button(self.root, text=self.grade[i][j], width=4, height=2,
                                font=('Arial', 14), command=lambda x=i, y=j: self.selecionar(x, y))
                btn.grid(row=i, column=j)
                linha.append(btn)
            self.botoes.append(linha)

        controles = tk.Frame(self.root)
        controles.grid(row=TAMANHO, column=0, columnspan=6, pady=10)

        for direcao in DIRECOES:
            btn = tk.Button(controles, text=direcao.upper(), command=lambda d=direcao: self.mover_selecionado(d))
            btn.pack(side=tk.LEFT, padx=5)

        ag_btn = tk.Button(controles, text="🧠 Resolver com AG", command=self.resolver_ag)
        ag_btn.pack(side=tk.RIGHT, padx=10)

    def atualizar_interface(self):
        for i in range(TAMANHO):
            for j in range(TAMANHO):
                letra = self.grade[i][j]
                self.botoes[i][j]['text'] = letra
                self.botoes[i][j]['bg'] = CORES.get(letra, 'gray')
        if self.selecionado:
            i, j = self.selecionado
            self.botoes[i][j]['bg'] = 'yellow'
        self.label_movimentos['text'] = f"Movimentos: {self.movimentos}"
        self.label_tentativas['text'] = f"Tentativas: {self.tentativas}"
        self.root.update_idletasks()

    def selecionar(self, i, j):
        if self.grade[i][j] != '.':
            self.selecionado = (i, j)
            self.atualizar_interface()

    def mover_selecionado(self, direcao):
        if not self.selecionado:
            return
        i, j = self.selecionado
        letra = self.grade[i][j]
        novo = self.mover_carro(self.grade, letra, direcao)
        if novo:
            self.grade = novo
            self.movimentos += 1
            self.atualizar_interface()
            if any(j == TAMANHO - 1 for i, j in self.encontrar_carro('X')):
                messagebox.showinfo("🏁 Vitória!", f"Você venceu em {self.movimentos} movimentos!")

    def encontrar_carro(self, letra):
        return [(i, j) for i in range(TAMANHO) for j in range(TAMANHO) if self.grade[i][j] == letra]

    def mover_carro(self, grade, letra, direcao):
        posicoes = [(i, j) for i in range(TAMANHO) for j in range(TAMANHO) if grade[i][j] == letra]
        if not posicoes:
            return False
        novo = deepcopy(grade)
        if all(p[0] == posicoes[0][0] for p in posicoes):
            linha = posicoes[0][0]
            js = sorted(p[1] for p in posicoes)
            if direcao == 'esquerda' and js[0] > 0 and novo[linha][js[0] - 1] == '.':
                novo[linha][js[0] - 1] = letra
                novo[linha][js[-1]] = '.'
            elif direcao == 'direita' and js[-1] < TAMANHO - 1 and novo[linha][js[-1] + 1] == '.':
                novo[linha][js[-1] + 1] = letra
                novo[linha][js[0]] = '.'
            else:
                return False
        else:
            coluna = posicoes[0][1]
            is_ = sorted(p[0] for p in posicoes)
            if direcao == 'cima' and is_[0] > 0 and novo[is_[0] - 1][coluna] == '.':
                novo[is_[0] - 1][coluna] = letra
                novo[is_[-1]][coluna] = '.'
            elif direcao == 'baixo' and is_[-1] < TAMANHO - 1 and novo[is_[-1] + 1][coluna] == '.':
                novo[is_[-1] + 1][coluna] = letra
                novo[is_[0]][coluna] = '.'
            else:
                return False
        return novo

    def resolver_ag(self):
        self.tentativas = 0
        self.inicio_tempo = time.time()
        self.executar_geracoes()

    def executar_geracoes(self):
        self.tentativas += 1
        temp = deepcopy(self.grade)
        passos = []

        for _ in range(200):
            carros = sorted(set(c for row in temp for c in row if c != '.'))
            carro = random.choice(carros)
            direcoes_validas = list(DIRECOES)
            random.shuffle(direcoes_validas)
            for direcao in direcoes_validas:
                novo_estado = self.mover_carro(temp, carro, direcao)
                if novo_estado:
                    temp = novo_estado
                    passos.append((carro, direcao))
                    break

            self.grade = deepcopy(temp)
            self.atualizar_interface()

            pos = [(i, j) for i in range(TAMANHO) for j in range(TAMANHO) if temp[i][j] == 'X']
            if any(j == TAMANHO - 1 for i, j in pos):
                tempo_final = time.time() - self.inicio_tempo
                self.grade = temp
                self.atualizar_interface()
                resumo = f"Resolvido com {len(passos)} movimentos após {self.tentativas} tentativas em {tempo_final:.2f} segundos.\n\nSequência:\n"
                for p in passos:
                    resumo += f"{p[0]} → {p[1]}\n"
                messagebox.showinfo("🏆 AG", resumo)
                return

        self.root.after(50, self.executar_geracoes)

if __name__ == "__main__":
    root = tk.Tk()
    app = EstacionamentoApp(root)
    root.mainloop()
