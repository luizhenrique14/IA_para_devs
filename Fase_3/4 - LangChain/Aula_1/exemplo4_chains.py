# Exemplo 4 - Encadeamento de etapas (Chains)
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

def primeira_etapa(llm, input_text):
    """Primeira etapa: análise do texto"""
    prompt = f"Analise o seguinte texto: {input_text}"
    return llm.invoke(prompt)

def segunda_etapa(llm, analisado):
    """Segunda etapa: resumo da análise"""
    prompt = f"Agora resuma a análise: {analisado}"
    return llm.invoke(prompt)

def executar_cadeia(llm, input_text):
    """Executa a cadeia completa de processamento"""
    print("Iniciando análise...")
    analise = primeira_etapa(llm, input_text)
    print("\nAnálise completa:", analise)
    
    print("\nGerando resumo...")
    resumo = segunda_etapa(llm, analise)
    return resumo

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializa o modelo
    llm = OpenAI(api_key=api_key)

    # Executa a cadeia
    input_text = "Os impactos da IA na sociedade moderna."
    resultado = executar_cadeia(llm, input_text)
    print("\nResumo final:", resultado)

if __name__ == "__main__":
    main()