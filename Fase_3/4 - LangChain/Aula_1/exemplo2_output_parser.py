# Exemplo 2 - Uso de Output Parsers (JSON estruturado)
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

class Pessoa(BaseModel):
    nome: str = Field(..., description="O nome da pessoa")
    idade: int = Field(..., description="A idade da pessoa")
    profissao: str = Field(..., description="A profissão da pessoa")

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializa o modelo e o parser
    llm = OpenAI(api_key=api_key)
    parser = PydanticOutputParser(pydantic_object=Pessoa)

    # Define o prompt
    prompt = """
    Formate a seguinte resposta em JSON com os campos em lowercase e sem
    caracteres especiais:
    "Nome: Maria, Idade: 30, Profissão: Engenheira"
    Responda no seguinte formato: {"nome": "valor", "idade": valor, "profissao": "valor"}
    """

    # Faz a requisição e tenta fazer o parse
    response = llm.invoke(prompt)
    print("Resposta original do LLM:", response)

    try:
        parsed_response = parser.parse(response)
        print("Resposta formatada:", parsed_response)
    except Exception as e:
        print("Erro ao usar o parser:", e)

if __name__ == "__main__":
    main()