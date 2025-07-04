# Importação das bibliotecas necessárias
import pandas as pd

# Passo 1: Carregando os datasets de treino e teste
print("Lendo arquivos de treino e teste...")
artigo_treino = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Word2Vec/treino.csv")
artigo_teste = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Word2Vec/teste.csv")
print("Exemplo de dados de treino:")
print(artigo_treino.head())

# Passo 2: Carregando o modelo spaCy para português
print("Carregando modelo spaCy para português...")
import spacy
nlp = spacy.load("pt_core_news_sm")

# Passo 3: Exemplo de processamento de texto com spaCy
print("Processando texto de exemplo com spaCy...")
texto = "Adoro a cidade de caldas novas!"
doc = nlp(texto)
print("Tipo do objeto doc:", type(doc))
print("Entidades nomeadas encontradas:", doc.ents)
print("O segundo token é stopword?", doc[1].is_stop)
print("O segundo token é alfabético?", doc[1].is_alpha)

# Passo 4: Gerando textos para tratamento (títulos em minúsculo)
print("Gerando textos para tratamento (lowercase)...")
textos_para_tratamento = (titulos.lower() for titulos in artigo_treino.title)

# Passo 5: Função para tratar textos (remoção de stopwords e tokens não alfabéticos)
def trata_textos(doc):
    tokens_validos = []
    for token in doc:
        e_valido = not token.is_stop and token.is_alpha
        if e_valido:
            tokens_validos.append(token.text)
    if len(tokens_validos) > 2:
        return " ".join(tokens_validos)

# Testando a função de tratamento em um texto de exemplo
print("Testando função de tratamento de texto...")
texto = "Adoro a 342342 #$@#@#$ cidade de caldas novas!"
doc = nlp(texto)
print("Texto tratado:", trata_textos(doc))

# Passo 6: Aplicando o tratamento em todos os títulos usando processamento em lote do spaCy
print("Processando todos os títulos do dataset de treino com spaCy...")
textos_tratados = [trata_textos(doc) for doc in nlp.pipe(textos_para_tratamento,
                                                        batch_size=1000,
                                                        n_process=-1)]
print("Quantidade de títulos tratados:", len(textos_tratados))

# Passo 7: Criando DataFrame com os títulos tratados
titulos_tratados = pd.DataFrame({"titulo": textos_tratados})
print("Exemplo de títulos tratados:")
print(titulos_tratados.head())

# Passo 8: Treinando modelo Word2Vec CBOW
from gensim.models import Word2Vec
print("Configurando modelo Word2Vec (CBOW)...")
w2v_modelo = Word2Vec(
    sg=0,  # CBOW
    window=2,
    vector_size=300,
    min_count=5,
    alpha=0.03,
    min_alpha=0.007
)

print("Removendo nulos e duplicados dos títulos tratados...")
print("Antes:", len(titulos_tratados))
titulos_tratados = titulos_tratados.dropna().drop_duplicates()
print("Depois:", len(titulos_tratados))

# Passo 9: Preparando lista de listas de tokens para o Word2Vec
print("Gerando lista de listas de tokens para Word2Vec...")
lista_lista_tokens = [titulo.split(" ") for titulo in titulos_tratados.titulo]

# Passo 10: Construindo vocabulário do modelo Word2Vec
print("Construindo vocabulário do Word2Vec...")
w2v_modelo.build_vocab(lista_lista_tokens)
print("Tamanho do vocabulário:", len(w2v_modelo.wv))

# Passo 11: Callback para monitorar o loss durante o treinamento
from gensim.models.callbacks import CallbackAny2Vec
class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss após a época {}: {}'.format(self.epoch, loss))
        else:
            print('Loss após a época {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

# Passo 12: Treinando o modelo Word2Vec (CBOW)
print("Treinando modelo Word2Vec (CBOW)...")
w2v_modelo.train(
    lista_lista_tokens,
    total_examples=w2v_modelo.corpus_count,
    epochs=30,
    compute_loss=True,
    callbacks=[callback()]
)
print("Treinamento CBOW finalizado.")

# Passo 13: Exemplo de palavras mais similares no modelo CBOW
print("Palavras mais similares a 'google' (CBOW):")
print(w2v_modelo.wv.most_similar("google"))

# Passo 14: Treinando modelo Word2Vec Skip-gram
print("Configurando modelo Word2Vec (Skip-gram)...")
w2v_modelo_sg = Word2Vec(
    sg=1,  # Skip-gram
    window=5,
    vector_size=300,
    min_count=5,
    alpha=0.03,
    min_alpha=0.007
)
print("Construindo vocabulário do Skip-gram...")
w2v_modelo_sg.build_vocab(lista_lista_tokens)
print("Treinando modelo Word2Vec (Skip-gram)...")
w2v_modelo_sg.train(
    lista_lista_tokens,
    total_examples=w2v_modelo_sg.corpus_count,
    epochs=30,
    compute_loss=True,
    callbacks=[callback()]
)
print("Treinamento Skip-gram finalizado.")

# Passo 15: Exemplo de palavras mais similares no modelo Skip-gram
print("Palavras mais similares a 'google' (Skip-gram):")
print(w2v_modelo_sg.wv.most_similar("google"))

# Passo 16: Salvando os modelos treinados
print("Salvando os modelos treinados em arquivos txt...")
w2v_modelo.wv.save_word2vec_format("/content/drive/MyDrive/Colab Notebooks/Word2Vec/modelo_cbow.txt", binary=False)
w2v_modelo_sg.wv.save_word2vec_format("/content/drive/MyDrive/Colab Notebooks/Word2Vec/modelo_sg.txt", binary=False)
print("Modelos salvos.")

# Passo 17: Função de tokenização para classificação
print("Definindo função de tokenização para classificação...")
import spacy
nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner", "tagger", "textcat"])
def tokenizador(texto):
    tokens_validos = []
    doc = nlp(texto)
    for token in doc:
        e_valido = not token.is_stop and token.is_alpha
        if e_valido:
            tokens_validos.append(token.text.lower())
    return tokens_validos

# Testando tokenizador
print("Testando tokenizador em texto de exemplo:")
texto = "Adoro a 342342 #$@#@#$ cidade de caldas novas!"
print(tokenizador(texto))

# Passo 18: Função para combinar vetores das palavras (soma)
import numpy as np
def combinacao_de_vetores_por_soma(palavras, modelo):
    vetor_resultante = np.zeros(300)
    for pn in palavras:
        try:
            vetor_resultante =+ modelo.wv.get_vector(pn)
        except KeyError:
            pass
    return vetor_resultante

# Passo 19: Função para gerar matriz de vetores para todos os textos
def matriz_vetores(textos, modelo):
    x = len(textos)
    y = 300
    matriz = np.zeros((x, y))
    for i in range(x):
        palavras = tokenizador(textos.iloc[i])
        matriz[i] = combinacao_de_vetores_por_soma(palavras, modelo)
    return matriz

# Passo 20: Gerando matrizes de vetores para treino e teste (CBOW)
print("Gerando matrizes de vetores para treino e teste (CBOW)...")
matriz_vetores_treino_cbow = matriz_vetores(artigo_treino.title, w2v_modelo)
matriz_vetores_teste_cbow = matriz_vetores(artigo_teste.title, w2v_modelo)
print("Formato matriz treino CBOW:", matriz_vetores_treino_cbow.shape)
print("Formato matriz teste CBOW:", matriz_vetores_teste_cbow.shape)

# Passo 21: Gerando matrizes de vetores para treino e teste (Skip-gram)
print("Gerando matrizes de vetores para treino e teste (Skip-gram)...")
matriz_vetores_treino_sg = matriz_vetores(artigo_treino.title, w2v_modelo_sg)
matriz_vetores_teste_sg = matriz_vetores(artigo_teste.title, w2v_modelo_sg)
print("Formato matriz treino Skip-gram:", matriz_vetores_treino_sg.shape)
print("Formato matriz teste Skip-gram:", matriz_vetores_teste_sg.shape)

# Passo 22: Função de classificação com Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
def classificador(modelo, x_treino, y_treino, x_teste, y_teste):
    print("Treinando classificador de Regressão Logística...")
    RL = LogisticRegression(max_iter=800)
    RL.fit(x_treino, y_treino)
    categorias = RL.predict(x_teste)
    resultados = classification_report(y_teste, categorias)
    print("Relatório de classificação:")
    print(resultados)
    return RL

# Passo 23: Classificação usando vetores CBOW
print("Classificando com vetores CBOW...")
RL_cbow = classificador(
    w2v_modelo,
    matriz_vetores_treino_cbow,
    artigo_treino.category,
    matriz_vetores_teste_cbow,
    artigo_teste.category
)

# Passo 24: Classificação usando vetores Skip-gram
print("Classificando com vetores Skip-gram...")
RL_sg = classificador(
    w2v_modelo_sg,
    matriz_vetores_treino_sg,
    artigo_treino.category,
    matriz_vetores_teste_sg,
    artigo_teste.category
)