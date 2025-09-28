
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave da API da OpenAI do ambiente
api_key = os.getenv("OPENAI_API_KEY")

def exemplo_sumarizacao(texto_longo=None):
    """
    Gera e executa um prompt para resumir um texto em 3 frases.
    """
    if texto_longo is None:
        texto_longo = (
            "A inteligência artificial está transformando diversas indústrias. "
            "Na saúde, ela está melhorando diagnósticos, enquanto no setor de transportes "
            "está otimizando rotas e aumentando a eficiência. No entanto, desafios éticos "
            "e regulatórios continuam a surgir, principalmente em relação à privacidade "
            "e ao uso de dados sensíveis."
        )
    
    template = "Resuma o texto a seguir em 3 frases:\\n\\nTexto: {texto}"
    
    if not api_key:
        return "Chave da API da OpenAI não encontrada. Por favor, configure a variável de ambiente OPENAI_API_KEY."

    llm = OpenAI(api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["texto"])
    prompt_completo = prompt.format(texto=texto_longo)
    
    resposta = llm.invoke(prompt_completo)
    return resposta

if __name__ == "__main__":
    # Exemplo de como chamar a função e imprimir a resposta
    resumo = exemplo_sumarizacao()
    print("--- Sumarização do Texto ---")
    print(resumo)
