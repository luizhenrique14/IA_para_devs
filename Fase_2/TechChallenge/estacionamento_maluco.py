import tkinter as tk
from tkinter import messagebox
import random
import threading
from copy import deepcopy

TAMANHO = 6
# CARROS_INICIAIS = [
#     ('A', [(0, 0), (0, 1)]),
#     ('B', [(1, 3), (1, 4), (1, 5)]),
#     ('X', [(2, 0), (2, 1)]),  # carro vermelho
#     ('C', [(3, 0), (4, 0)]),
#     ('D', [(3, 2), (3, 3), (3, 4)]),
#     ('E', [(5, 3), (5, 4), (5, 5)]),
# ]

CARROS_INICIAIS = [
    ('A', [(0, 0), (1, 0)]),
    ('B', [(0, 2), (1, 2), (2, 2)]),
    ('C', [(0, 4), (0, 5)]),
    ('D', [(1, 4), (2, 4), (3, 4)]),
    ('X', [(2, 1), (2, 2)]),  # carro vermelho, mais bloqueado
    ('E', [(3, 0), (3, 1), (3, 2)]),
    ('F', [(4, 3), (5, 3), (5, 4)]),
]

CORES = {
    'X': 'red',
    'A': 'lightblue',
    'B': 'lightgreen',
    'C': 'orange',
    'D': 'violet',
    'E': 'gold',
    '.': 'SystemButtonFace'
}


class EstacionamentoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Estacionamento Maluco")
        self.grade = [['.' for _ in range(TAMANHO)] for _ in range(TAMANHO)]
        self.botoes = []
        self.selecionado = None
        self.movimentos = 0
        self.label_movimentos = tk.Label(self.root, text="Movimentos: 0", font=("Arial", 12))
        self.label_movimentos.grid(row=TAMANHO + 1, column=0, columnspan=6)

        self.inicializar_grade()
        self.desenhar_interface()
        self.atualizar_interface()

        # ⌨️ Liga as teclas do teclado
        self.root.bind("<Up>", lambda e: self.mover('cima'))
        self.root.bind("<Down>", lambda e: self.mover('baixo'))
        self.root.bind("<Left>", lambda e: self.mover('esquerda'))
        self.root.bind("<Right>", lambda e: self.mover('direita'))

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

        # 🟢 Botões de controle
        controles = tk.Frame(self.root)
        controles.grid(row=TAMANHO, column=0, columnspan=6, pady=10)

        for direcao in ['cima', 'baixo', 'esquerda', 'direita']:
            btn = tk.Button(controles, text=direcao.upper(), command=lambda d=direcao: self.mover(d))
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

    def selecionar(self, i, j):
        if self.grade[i][j] != '.':
            self.selecionado = (i, j)
            self.atualizar_interface()

    def encontrar_carro(self, letra):
        return [(i, j) for i in range(TAMANHO) for j in range(TAMANHO) if self.grade[i][j] == letra]

    def mover(self, direcao):
        if not self.selecionado:
            return
        i, j = self.selecionado
        letra = self.grade[i][j]
        posicoes = self.encontrar_carro(letra)

        novo = deepcopy(self.grade)

        if all(p[0] == posicoes[0][0] for p in posicoes):  # horizontal
            linha = posicoes[0][0]
            js = sorted(p[1] for p in posicoes)
            if direcao == 'esquerda' and js[0] > 0 and novo[linha][js[0] - 1] == '.':
                novo[linha][js[0] - 1] = letra
                novo[linha][js[-1]] = '.'
            elif direcao == 'direita' and js[-1] < 5 and novo[linha][js[-1] + 1] == '.':
                novo[linha][js[-1] + 1] = letra
                novo[linha][js[0]] = '.'
            else:
                return
        else:  # vertical
            coluna = posicoes[0][1]
            is_ = sorted(p[0] for p in posicoes)
            if direcao == 'cima' and is_[0] > 0 and novo[is_[0] - 1][coluna] == '.':
                novo[is_[0] - 1][coluna] = letra
                novo[is_[-1]][coluna] = '.'
            elif direcao == 'baixo' and is_[-1] < 5 and novo[is_[-1] + 1][coluna] == '.':
                novo[is_[-1] + 1][coluna] = letra
                novo[is_[0]][coluna] = '.'
            else:
                return

        self.grade = novo
        self.movimentos += 1
        self.atualizar_interface()
        if self.grade[2][5] == 'X':
            messagebox.showinfo("🏁 Vitória!", f"Você venceu em {self.movimentos} movimentos!")

    # 🤖 Algoritmo Genético simples (modo demo)
    def resolver_ag(self):
        def ag_thread():
            pop = [['direita'] * i for i in range(1, 7)]
            for seq in pop:
                temp = deepcopy(self.grade)
                count = 0
                for mov in seq:
                    pos = [(i, j) for i in range(TAMANHO) for j in range(TAMANHO) if temp[i][j] == 'X']
                    js = sorted(p[1] for p in pos)
                    linha = pos[0][0]
                    if js[-1] >= 5:
                        self.grade = temp
                        self.movimentos += count
                        self.atualizar_interface()
                        messagebox.showinfo("🏆 AG", f"Resolvido com {count} movimentos!")
                        return
                    # tenta mover direita
                    if js[-1] < 5 and temp[linha][js[-1] + 1] == '.':
                        temp[linha][js[-1] + 1] = 'X'
                        temp[linha][js[0]] = '.'
                        count += 1
                self.grade = temp
                self.movimentos += count
                self.atualizar_interface()
            messagebox.showinfo("❌ AG", "Não conseguiu resolver!")

        threading.Thread(target=ag_thread).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = EstacionamentoApp(root)
    root.mainloop()
