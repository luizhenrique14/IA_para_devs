"""
Exemplo de Sequential Chain - Gerador e Verificador de Letras de Rap

Este exemplo demonstra o uso de Sequential Chain do LangChain para criar um fluxo de
trabalho que:
1. Gera uma letra de rap baseada em um tema
2. Verifica se a letra contém conteúdo inadequado
3. Faz uma verificação final com metadados

Conceitos demonstrados:
- Criação de múltiplos prompts em sequência
- Uso de SequentialChain para encadear operações
- Uso de SimpleMemory para passar dados adicionais
- Formatação de saída estruturada
"""

from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import SimpleMemory
from datetime import datetime
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def criar_chain_geracao_rap():
    """
    Primeira etapa: Geração da letra de rap
    Cria um prompt template e uma chain para gerar a letra inicial
    """
    template_rapper = """Você é um compositor de rap renomado. Sua missão é criar uma letra de rap
    inspirada no tema fornecido.

    Tema da música:
    {input}"""

    prompt_template_rapper = PromptTemplate(
        input_variables=["input"],
        template=template_rapper
    )

    return LLMChain(
        llm=OpenAI(api_key=api_key),
        output_key="letra",
        prompt=prompt_template_rapper
    )

def criar_chain_verificacao():
    """
    Segunda etapa: Verificação de conteúdo inadequado
    Analisa a letra gerada em busca de conteúdo impróprio
    """
    template_verificador = """Você é responsável por revisar letras de rap. Seu trabalho é verificar se as letras contêm
    algum conteúdo violento ou linguagem inadequada.

    Por favor, responda no formato de um dicionário Python:
    letra: a letra recebida
    Palavras_violentas: Verdadeiro ou Falso

    Aqui está a letra a ser verificada:
    {letra}"""

    prompt_template_verificador = PromptTemplate(
        input_variables=["letra"],
        template=template_verificador
    )

    return LLMChain(
        llm=OpenAI(api_key=api_key),
        output_key="letra_verificada",
        prompt=prompt_template_verificador
    )

def criar_chain_final():
    """
    Terceira etapa: Verificação final e adição de metadados
    Adiciona informações de data/hora e status de verificação
    """
    template_final = """Você é responsável pela verificação final das letras de rap. Seu trabalho é garantir que
    a letra revisada esteja dentro dos padrões aceitáveis.

    Sua resposta final deve ser no formato de dicionário Python:
    letra: a letra recebida
    Data e hora da verificação: {data_hora_verificacao}
    Verificada por um humano: {verificada_por_humano}

    Aqui está a letra revisada:
    {letra_verificada}"""

    prompt_template_final = PromptTemplate(
        input_variables=["letra_verificada", "data_hora_verificacao", "verificada_por_humano"],
        template=template_final
    )

    return LLMChain(
        llm=OpenAI(api_key=api_key),
        output_key="saida_final",
        prompt=prompt_template_final
    )

def criar_sequential_chain():
    """
    Monta a cadeia sequencial completa, conectando todas as etapas
    """
    chain_rapper = criar_chain_geracao_rap()
    chain_verificador = criar_chain_verificacao()
    chain_final = criar_chain_final()

    # Criando a sequential chain que conecta todas as etapas
    return SequentialChain(
        memory=SimpleMemory(memories={
            "data_hora_verificacao": str(datetime.utcnow()),
            "verificada_por_humano": "Falso"
        }),
        chains=[chain_rapper, chain_verificador, chain_final],
        input_variables=["input"],
        output_variables=["saida_final"],
        verbose=True
    )

if __name__ == "__main__":
    # Verifica se a chave da API está configurada
    if not api_key:
        print("Erro: Chave da API da OpenAI não encontrada. Configure a variável OPENAI_API_KEY no arquivo .env")
        exit(1)

    # Cria e executa a chain
    chain = criar_sequential_chain()
    
    # Exemplo de uso com um tema para a letra de rap
    tema = "paz e amor"
    print(f"\nGerando letra de rap com o tema: {tema}\n")
    
    resultado = chain.run(input=tema)
    print("\nResultado final:")
    print(resultado)