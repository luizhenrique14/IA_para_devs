# Exemplo 5 - Acesso ao GPT-4
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializa o modelo com configurações específicas para GPT-4
    llm = OpenAI(
        api_key=api_key,
        model="gpt-4",  # Especifica o modelo GPT-4
        temperature=0.7,  # Ajusta a criatividade das respostas
        max_tokens=500   # Limite de tokens na resposta
    )

    # Faz a requisição
    prompt = """
    Explique o conceito de aprendizado por reforço, incluindo:
    1. Definição básica
    2. Principais componentes
    3. Um exemplo prático
    """
    response = llm.invoke(prompt)
    
    print("Resposta do GPT-4:", response)

if __name__ == "__main__":
    main()