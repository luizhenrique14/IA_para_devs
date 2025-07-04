# Importação das bibliotecas necessárias
import pandas as pd  # Para manipulação de dados tabulares
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística
from sklearn.feature_extraction.text import CountVectorizer  # Para criar Bag of Words
from wordcloud import WordCloud  # Para gerar nuvem de palavras
import matplotlib.pyplot as plt  # Para visualização de gráficos
import nltk  # Biblioteca para processamento de linguagem natural
from nltk import tokenize  # Para tokenização de texto
from string import punctuation  # Para manipular pontuações
import unidecode  # Para remover acentos
from sklearn.feature_extraction.text import TfidfVectorizer  # Para criar TF-IDF
from nltk import ngrams  # Para gerar NGrams

# Carregando o dataset
print("Carregando o dataset...")
avaliacoes = pd.read_csv("b2w.csv")
print("Primeiras linhas do dataset:")
print(avaliacoes.head())

# Pré-processamento inicial
print("\nRemovendo colunas irrelevantes e valores nulos...")
avaliacoes = avaliacoes.drop(["original_index", "review_text_processed", "review_text_tokenized",
                              "rating", "kfold_polarity", "kfold_rating"], axis=1)
avaliacoes.dropna(inplace=True, axis=0)
print("Dataset após pré-processamento:")
print(avaliacoes.head())

# Divisão em treino e teste
print("\nDividindo os dados em treino e teste...")
treino, teste, classe_treino, classe_teste = train_test_split(
    avaliacoes.review_text,
    avaliacoes.polarity,
    stratify=avaliacoes.polarity,
    random_state=71
)
print(f"Tamanho do conjunto de treino: {len(treino)}")
print(f"Tamanho do conjunto de teste: {len(teste)}")

# Bag of Words
print("\nCriando Bag of Words com as 100 palavras mais frequentes...")
vetorizar = CountVectorizer(max_features=100)
bag_of_words = vetorizar.fit_transform(avaliacoes.review_text)
print("Formato da matriz Bag of Words:", bag_of_words.shape)

# Treinamento com regressão logística
print("\nTreinando o modelo de regressão logística com Bag of Words...")
treino, teste, classe_treino, classe_teste = train_test_split(
    bag_of_words,
    avaliacoes.polarity,
    stratify=avaliacoes.polarity,
    random_state=71
)
regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia = regressao_logistica.score(teste, classe_teste)
print(f"Acurácia do modelo com Bag of Words: {acuracia:.2f}")

# Função para treinar modelo
print("\nDefinindo função para treinar o modelo com diferentes colunas...")
def treinar_modelo(dados, coluna_texto, coluna_sentimento):
    vetorizar = CountVectorizer(max_features=100)
    bag_of_words = vetorizar.fit_transform(dados[coluna_texto])

    treino, teste, classe_treino, classe_teste = train_test_split(
        bag_of_words,
        dados[coluna_sentimento],
        stratify=dados[coluna_sentimento],
        random_state=71
    )

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)

print(f"Acurácia do modelo com função: {treinar_modelo(avaliacoes, 'review_text', 'polarity'):.2f}")

# Word Cloud
print("\nGerando nuvem de palavras com todas as avaliações...")
todas_avaliacoes = [texto for texto in avaliacoes.review_text]
todas_palavras = ' '.join(todas_avaliacoes)
nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110).generate(todas_palavras)

plt.figure(figsize=(10, 7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis("off")
plt.title("Nuvem de Palavras")
plt.show()

# Remoção de Stop Words
print("\nRemovendo stop words das avaliações...")
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
frase_processada = list()
for avaliacao in avaliacoes.review_text:
    nova_frase = list()
    palavras_texto = tokenize.WhitespaceTokenizer().tokenize(avaliacao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

avaliacoes["texto_sem_stopwords"] = frase_processada
print("Exemplo de texto sem stop words:")
print(avaliacoes["texto_sem_stopwords"].head())

# TF-IDF
print("\nCriando matriz TF-IDF com as 100 palavras mais frequentes...")
tfidf = TfidfVectorizer(lowercase=False, max_features=100)
tfidf_tratados = tfidf.fit_transform(avaliacoes.texto_sem_stopwords)

treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados,
                                                              avaliacoes.polarity,
                                                              stratify=avaliacoes.polarity,
                                                              random_state=71)

regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
print(f"Acurácia do modelo com TF-IDF: {acuracia_tfidf:.2f}")

# NGrams
print("\nGerando NGrams (pares de palavras)...")
frase = "Comprei um ótimo produto"
frase_separada = tokenize.WhitespaceTokenizer().tokenize(frase)
pares = ngrams(frase_separada, 2)
print("Exemplo de NGrams:", list(pares))

tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
vetor_tfidf = tfidf.fit_transform(avaliacoes.texto_sem_stopwords)

treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                              avaliacoes.polarity,
                                                              random_state=71)

regressao_logistica = LogisticRegression(max_iter=200)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print(f"Acurácia do modelo com TF-IDF e NGrams: {acuracia_tfidf_ngrams:.2f}")