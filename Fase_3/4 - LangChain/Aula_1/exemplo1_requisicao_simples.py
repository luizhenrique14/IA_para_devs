# Exemplo 1 - Requisição simples à OpenAI
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Inicializa o modelo
    llm = OpenAI(api_key=api_key)

    # Faz a requisição
    response = llm.invoke("Explique o conceito de aprendizado por reforço.")
    print("Resposta:", response)

if __name__ == "__main__":
    main()