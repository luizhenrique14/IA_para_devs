"""
Stock Agent Module
This module implements a LangChain-based agent for analyzing stock data and providing insights.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from stock_operations import obter_dados_acoes, formatar_dados_acoes

# Load environment variables
load_dotenv()

# Configure OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0.6, api_key=os.getenv("OPENAI_API_KEY"))

def analisar_dados_acoes(dados_acoes):
    """
    Analyze stock data using OpenAI and provide insights.
    
    Args:
        dados_acoes (List[Tuple]): List of stock data records
    
    Returns:
        str: Analysis and insights from the AI model
    """
    prompt = "Analise os seguintes dados de ações e forneça insights detalhados:\n"
    prompt += formatar_dados_acoes(dados_acoes)
    prompt += "\nPor favor, forneça um resumo e quaisquer tendências ou insights notáveis."
    
    return llm(prompt)

def responder_pergunta(pergunta: str) -> str:
    """
    Process a user question about stocks and return both data and analysis.
    
    Args:
        pergunta (str): User's question about stocks
    
    Returns:
        str: Formatted response with data and AI insights
    """
    dados_acoes = obter_dados_acoes(pergunta)
    if not dados_acoes:
        return f"Nenhum dado encontrado para '{pergunta}'."
    
    resposta = f"Dados encontrados para '{pergunta}':\n"
    resposta += formatar_dados_acoes(dados_acoes)
    resposta += "\nInsights da OpenAI:\n"
    resposta += analisar_dados_acoes(dados_acoes)
    
    return resposta

def main():
    """
    Main function to run the stock agent interactively.
    """
    print("Bem-vindo ao Assistente de Análise de Ações!")
    print("Digite 'sair' para encerrar o programa.")
    
    while True:
        pergunta = input("\nSobre qual ação você deseja saber informações? ")
        if pergunta.lower() == 'sair':
            print("Encerrando o programa. Até logo!")
            break
            
        try:
            resposta = responder_pergunta(pergunta)
            print("\n" + "="*80)
            print(resposta)
            print("="*80)
        except Exception as e:
            print(f"Erro ao processar sua pergunta: {str(e)}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Erro: Chave da API OpenAI não encontrada. Configure a variável OPENAI_API_KEY no arquivo .env")
        exit(1)
    main()