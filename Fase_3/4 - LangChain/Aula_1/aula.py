```python
# Exemplo 1 - Requisição simples à OpenAI
from langchain_openai import OpenAI 
from dotenv import load_dotenv 
import os 

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY") 

llm = OpenAI(api_key=api_key) 
response = llm.invoke("Explique o conceito de aprendizado por reforço.") 
print(response)
```

```python
# Exemplo 2 - Uso de Output Parsers (JSON estruturado)
from langchain_openai import OpenAI 
from langchain.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field 
from dotenv import load_dotenv 
import os 

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY") 
llm = OpenAI(api_key=api_key) 

class Pessoa(BaseModel): 
   nome: str = Field(..., description="O nome da pessoa") 
   idade: int = Field(..., description="A idade da pessoa") 
   profissao: str = Field(..., description="A profissão da pessoa") 

parser = PydanticOutputParser(pydantic_object=Pessoa) 

prompt = """ 
Formate a seguinte resposta em JSON com os campos em lowercase e sem 
caracteres especiais: 
"Nome: Maria, Idade: 30, Profissão: Engenheira" 
Responda no seguinte formato: {"nome": "valor", "idade": valor, "profissao": "valor"} 
""" 

response = llm.invoke(prompt) 
print("Resposta original do LLM:", response) 

try: 
   parsed_response = parser.parse(response) 
   print("Resposta formatada:", parsed_response) 
except Exception as e: 
   print("Erro ao usar o parser:", e) 
```

```python
# Exemplo 3 - Memória de Conversação
from langchain_openai import OpenAI 
from langchain.memory import ConversationBufferMemory 
from dotenv import load_dotenv 
import os 

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY") 
llm = OpenAI(api_key=api_key) 

memory = ConversationBufferMemory() 

conversation = [ 
   {"role": "user", "content": "Qual é o meu nome?"}, 
   {"role": "assistant", "content": "Desculpe, não sei seu nome. Como você se chama?"} 
] 

memory.add_messages(conversation) 

prompt = """ 
Você agora tem o contexto da conversa que já ocorreu. Continue a conversa como 
se estivesse ciente da interação anterior. 
""" 
response = llm.invoke(prompt + str(memory.buffer)) 

print("Resposta do modelo:", response) 
print("Memória da conversa:", memory.buffer)
```

```python
# Exemplo 4 - Encadeamento de etapas (Chains)
from langchain_openai import OpenAI 
from dotenv import load_dotenv 
import os 

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY") 
llm = OpenAI(api_key=api_key) 

def primeira_etapa(input_text): 
   prompt = f"Analise o seguinte texto: {input_text}" 
   return llm.invoke(prompt) 

def segunda_etapa(analisado): 
   prompt = f"Agora resuma a análise: {analisado}" 
   return llm.invoke(prompt) 

def executar_cadeia(input_text): 
   analise = primeira_etapa(input_text) 
   print("Análise:", analise) 
   resumo = segunda_etapa(analise) 
   return resumo 

resultado = executar_cadeia("Os impactos da IA na sociedade moderna.") 
print("Resultado final:", resultado)
```

```python
# Exemplo 5 - Acesso básico ao GPT-4
from langchain_openai import OpenAI 
from dotenv import load_dotenv 
import os 

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY") 
llm = OpenAI(api_key=api_key) 

response = llm.invoke("Explique o conceito de aprendizado por reforço.") 
print(response)
```
