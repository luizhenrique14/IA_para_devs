
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def exemplo_extrair_datas_locais(texto):
    """
    Extrai datas e locais de um texto.
    """
    template = (
        "Extraia as datas e os locais do evento descrito no texto abaixo:\\n\\n{texto}"
    )
    
    if not api_key:
        return "Chave da API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY."

    llm = OpenAI(api_key=api_key, temperature=0.5)
    prompt = PromptTemplate(template=template, input_variables=["texto"])
    prompt_completo = prompt.format(texto=texto)
    
    resposta = llm.invoke(prompt_completo)
    return resposta

if __name__ == "__main__":
    texto_congresso = "O Congresso de Tecnologia será realizado nos dias 15 e 16 de setembro em São Paulo, e no dia 18 no Rio de Janeiro."
    extracao = exemplo_extrair_datas_locais(texto_congresso)
    print("--- Extração de Datas e Locais ---")
    print(extracao)
