import random
import logging

# 🎯 Configuração de logs
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 📚 Dados fictícios de exemplo
professores = ['Ana', 'Bruno', 'Carlos', 'Daniela']
materias = ['Matemática', 'Física', 'História', 'Programação']
turmas = ['Turma1', 'Turma2']
salas = ['101', '102']
horarios = ['Seg 08h', 'Seg 10h', 'Ter 08h', 'Ter 10h']

# 📌 Parâmetros do algoritmo genético
TAMANHO_POPULACAO = 10
GERACOES = 50
TAXA_CROSSOVER = 0.8
TAXA_MUTACAO = 0.1

# 🧬 Representação de um gene: (professor, materia, turma, sala, horario)
def gerar_gene():
    return (
        random.choice(professores),
        random.choice(materias),
        random.choice(turmas),
        random.choice(salas),
        random.choice(horarios)
    )

# 🎲 Gera um indivíduo completo (uma grade de 4 aulas por exemplo)
def gerar_individuo():
    return [gerar_gene() for _ in range(len(materias))]

# 🧪 Avaliação da aptidão (fitness)
def avaliar_fitness(individuo):
    conflitos = 0
    usados = set()
    for gene in individuo:
        chave = (gene[2], gene[4])  # (turma, horario)
        if chave in usados:
            conflitos += 1
        usados.add(chave)
    return 1 / (1 + conflitos)  # Quanto menos conflito, maior o fitness

# 🎯 Seleção por torneio
def selecao_torneio(populacao):
    k = 3
    selecionados = random.sample(populacao, k)
    selecionados.sort(key=lambda x: avaliar_fitness(x), reverse=True)
    return selecionados[0]

# 🔄 Cruzamento (crossover de um ponto)
def crossover(pai1, pai2):
    ponto = random.randint(1, len(pai1)-1)
    filho1 = pai1[:ponto] + pai2[ponto:]
    filho2 = pai2[:ponto] + pai1[ponto:]
    return filho1, filho2

# 💥 Mutação aleatória
def mutacao(individuo):
    if random.random() < TAXA_MUTACAO:
        idx = random.randint(0, len(individuo) - 1)
        individuo[idx] = gerar_gene()
    return individuo

# 🔁 Substitui população
def substituir(populacao, nova_geracao):
    return nova_geracao

# 🟢 Início do algoritmo
def algoritmo_genetico():
    logging.info("🟢 Iniciando o algoritmo genético...")
    populacao = [gerar_individuo() for _ in range(TAMANHO_POPULACAO)]

    for geracao in range(GERACOES):
        logging.info(f"\n📅 Geração {geracao + 1}")
        nova_populacao = []

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            if random.random() < TAXA_CROSSOVER:
                filho1, filho2 = crossover(pai1, pai2)
            else:
                filho1, filho2 = pai1, pai2
            filho1 = mutacao(filho1)
            filho2 = mutacao(filho2)
            nova_populacao.extend([filho1, filho2])

        populacao = substituir(populacao, nova_populacao[:TAMANHO_POPULACAO])

        # 🏁 Verifica condição de término
        melhores = sorted(populacao, key=lambda x: avaliar_fitness(x), reverse=True)
        melhor_fitness = avaliar_fitness(melhores[0])
        logging.info(f"💪 Melhor fitness da geração: {melhor_fitness:.4f}")
        if melhor_fitness == 1.0:
            logging.info("🏆 Solução ideal encontrada!")
            break

    # 🧾 Resultado final
    melhor = melhores[0]
    logging.info("\n✅ Grade de horários final:")
    for aula in melhor:
        logging.info(f"📚 {aula[1]} com {aula[0]} - {aula[2]} - Sala {aula[3]} - {aula[4]}")

# 🚀 Execução
if __name__ == "__main__":
    algoritmo_genetico()
