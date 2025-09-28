
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def exemplo_introducao_ia():
    """
    Gera uma introdução sobre Inteligência Artificial.
    """
    template = (
        "Escreva uma introdução para um artigo que explica o conceito de inteligência artificial, "
        "abordando suas aplicações principais e como ela impacta o futuro da tecnologia."
    )
    
    if not api_key:
        return "Chave da API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY."

    llm = OpenAI(api_key=api_key, temperature=0.7)
    
    resposta = llm.invoke(template)
    return resposta

if __name__ == "__main__":
    # Exemplo de como chamar a função e imprimir a resposta
    introducao = exemplo_introducao_ia()
    print("--- Introdução sobre IA ---")
    print(introducao)
