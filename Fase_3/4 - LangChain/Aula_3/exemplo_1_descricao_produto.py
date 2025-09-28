
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
# Certifique-se de ter um arquivo .env com a sua chave da API da OpenAI
# Exemplo: OPENAI_API_KEY="sua-chave-aqui"
api_key = os.getenv("OPENAI_API_KEY")

def exemplo_descricao_produto(produto="Smartphone X"):
    """
    Cria e executa um prompt para descrever as principais características de um produto.
    """
    template = "Descreva as principais características de um produto chamado {produto}."
    
    # Verifica se a chave da API está configurada
    if not api_key:
        return "Chave da API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY."

    llm = OpenAI(api_key=api_key, temperature=0.7)
    prompt = PromptTemplate(template=template, input_variables=["produto"])
    prompt_completo = prompt.format(produto=produto)
    
    resposta = llm.invoke(prompt_completo)
    return resposta

if __name__ == "__main__":
    # Exemplo de como chamar a função e imprimir a resposta
    descricao = exemplo_descricao_produto("Câmera de Segurança Inteligente")
    print("--- Descrição do Produto ---")
    print(descricao)
