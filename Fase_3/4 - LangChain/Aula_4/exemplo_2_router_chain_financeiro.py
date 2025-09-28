"""
Exemplo de Router Chain - Consultor Financeiro

Este exemplo demonstra o uso de Router Chain do LangChain para criar um sistema
que roteia perguntas financeiras para diferentes especialistas:
1. Especialista em ações
2. Especialista em renda fixa

Conceitos demonstrados:
- Criação de múltiplos prompts especializados
- Uso de Router Chain para direcionamento baseado no conteúdo
- Tipagem com TypedDict para estruturar a saída
- Uso de operadores | para composição de chains
"""

from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def criar_prompt_especialistas():
    """
    Cria os prompts para cada especialista (ações e renda fixa)
    """
    prompt_acoes = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em investimentos em ações."),
        ("human", "{query}"),
    ])

    prompt_renda_fixa = ChatPromptTemplate.from_messages([
        ("system", "Você é um especialista em investimentos de renda fixa."),
        ("human", "{query}"),
    ])

    return prompt_acoes, prompt_renda_fixa

def criar_chains_especialistas(llm):
    """
    Cria as chains para processar consultas de cada especialista
    """
    prompt_acoes, prompt_renda_fixa = criar_prompt_especialistas()
    
    chain_acoes = prompt_acoes | llm | StrOutputParser()
    chain_renda_fixa = prompt_renda_fixa | llm | StrOutputParser()
    
    return chain_acoes, chain_renda_fixa

def criar_router_chain(llm):
    """
    Cria a chain de roteamento que decide qual especialista deve responder
    """
    route_system = "Roteie a pergunta do usuário para o especialista em ações ou renda fixa."
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", route_system),
        ("human", "{query}"),
    ])

    class RouteQuery(TypedDict):
        destination: Literal["acoes", "renda_fixa"]

    return (
        route_prompt
        | llm.with_structured_output(RouteQuery)
        | itemgetter("destination")
    )

def criar_chain_consultor_financeiro():
    """
    Monta a chain completa que inclui o roteador e os especialistas
    """
    if not api_key:
        raise ValueError("Chave da API da OpenAI não encontrada. Configure a variável OPENAI_API_KEY no arquivo .env")

    # Inicializa o modelo
    llm = ChatOpenAI(api_key=api_key)
    
    # Cria as chains dos especialistas
    chain_acoes, chain_renda_fixa = criar_chains_especialistas(llm)
    
    # Cria a chain de roteamento
    route_chain = criar_router_chain(llm)
    
    # Monta a chain final que combina o roteamento com os especialistas
    chain = {
        "destination": route_chain,
        "query": lambda x: x["query"],
    } | RunnableLambda(
        lambda x: chain_acoes if x["destination"] == "acoes" else chain_renda_fixa
    )
    
    return chain

def processar_consulta(consulta):
    """
    Processa uma consulta financeira usando a chain do consultor
    """
    chain = criar_chain_consultor_financeiro()
    return chain.invoke({"query": consulta})

if __name__ == "__main__":
    # Lista de perguntas para testar o sistema
    perguntas = [
        "Quais são os riscos de investir em ações de tecnologia?",
        "Como funciona o rendimento da poupança?",
        "Qual a melhor estratégia para investir em fundos imobiliários?",
        "Vale a pena investir em CDB?",
    ]
    
    print("Sistema de Consultoria Financeira\n")
    for pergunta in perguntas:
        print(f"\nPergunta: {pergunta}")
        try:
            resposta = processar_consulta(pergunta)
            print("\nResposta do especialista:")
            print(resposta)
        except Exception as e:
            print(f"Erro ao processar a pergunta: {str(e)}")