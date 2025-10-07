# Importações necessárias
from langchain_community.llms import Ollama  # Importa o modelo Ollama
from langchain.callbacks.manager import CallbackManager  # Gerenciador de callbacks para streaming
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # Handler para mostrar saída em tempo real
from langchain.prompts import PromptTemplate  # Para criar templates de prompts
from langchain.chains import LLMChain  # Para criar cadeias de processamento
from langchain_community.document_loaders import TextLoader  # Para carregar documentos de texto
import os  # Para operações com arquivos

# Função para carregar documentos de texto
def carregar_documentos(caminho_arquivo):
    """
    Carrega um arquivo de texto usando o TextLoader do LangChain
    """
    loader = TextLoader(caminho_arquivo)
    documentos = loader.load()
    return documentos

# Função para limpar o texto (remove espaços em branco extras)
def limpar_texto(texto):
    """
    Remove espaços em branco no início e fim do texto
    """
    return texto.strip()

# Configuração do modelo Ollama (modelo local)
llm = Ollama(
    model="llama2",  # Usa o modelo Llama 2
    num_gpu=0,  # Não usa GPU
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])  # Configura streaming de saída
)

# Define os templates para análise de sentimento e resumo
prompt_sentimento = "Analise o sentimento do seguinte texto em português: {text}"
prompt_resumo = "Gere um resumo em português para o seguinte texto: {text}"

# Cria os templates de prompt usando PromptTemplate
template_sentimento = PromptTemplate(input_variables=["text"], template=prompt_sentimento)
template_resumo = PromptTemplate(input_variables=["text"], template=prompt_resumo)

# Cria as chains (cadeias) para sentimento e resumo
chain_sentimento = LLMChain(llm=llm, prompt=template_sentimento)
chain_resumo = LLMChain(llm=llm, prompt=template_resumo)

# Define o arquivo de entrada
caminho_arquivo = "noticias.txt"

# Verifica se o arquivo existe
if not os.path.exists(caminho_arquivo):
    raise FileNotFoundError(f"O arquivo {caminho_arquivo} não foi encontrado.")

# Carrega os documentos
documentos = carregar_documentos(caminho_arquivo)

# Processa cada documento
for doc in documentos:
    # Limpa o texto do documento
    texto_limpo = limpar_texto(doc.page_content)

    # Realiza análise de sentimento e geração de resumo
    resultado_sentimento = chain_sentimento.invoke({"text": texto_limpo})
    resultado_resumo = chain_resumo.invoke({"text": texto_limpo})

    # Imprime os resultados
    print(f"Notícia: {texto_limpo}")
    print(f"Sentimento: {resultado_sentimento}")
    print(f"Resumo: {resultado_resumo}")
    print("-" * 120)  # Linha separadora