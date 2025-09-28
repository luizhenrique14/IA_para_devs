
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def exemplo_gerar_perguntas(texto):
    """
    Gera três perguntas a partir de um texto sobre um determinado assunto.
    """
    template = (
        "Leia o texto abaixo e gere três perguntas que possam ser usadas para avaliar o entendimento do conteúdo.\\n\\n"
        "Texto: {texto}"
    )
    
    if not api_key:
        return "Chave da API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY."

    llm = OpenAI(api_key=api_key, temperature=0.5)
    prompt = PromptTemplate(template=template, input_variables=["texto"])
    prompt_completo = prompt.format(texto=texto)
    
    resposta = llm.invoke(prompt_completo)
    return resposta

if __name__ == "__main__":
    texto_exemplo = "As mudanças climáticas são um dos maiores desafios globais da atualidade. Elas afetam ecossistemas, sociedades e economias, causando eventos climáticos extremos e mudanças nos padrões naturais."
    perguntas = exemplo_gerar_perguntas(texto_exemplo)
    print("--- Perguntas sobre Mudanças Climáticas ---")
    print(perguntas)
