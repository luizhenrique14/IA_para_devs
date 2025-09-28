
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def exemplo_extracao_avaliacao(avaliacao_text=None):
    """
    Gera e executa um prompt para extrair informações específicas de uma avaliação de produto.
    """
    if avaliacao_text is None:
        avaliacao_text = "Comprei um notebook Dell. Excelente qualidade, mas demorou 10 dias para chegar."
    
    template = (
        "Extraia as informações abaixo da avaliação de produto fornecida:\\n\\n"
        "- Nome do produto\\n"
        "- Avaliação do cliente\\n"
        "- Tempo de entrega\\n\\n"
        "Avaliação: {avaliacao}"
    )
    
    if not api_key:
        return "Chave da API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY."

    llm = OpenAI(api_key=api_key, temperature=0.5)
    prompt = PromptTemplate(template=template, input_variables=["avaliacao"])
    prompt_completo = prompt.format(avaliacao=avaliacao_text)
    
    resposta = llm.invoke(prompt_completo)
    return resposta

if __name__ == "__main__":
    # Exemplo de como chamar a função e imprimir a resposta
    extracao = exemplo_extracao_avaliacao()
    print("--- Extração da Avaliação ---")
    print(extracao)
