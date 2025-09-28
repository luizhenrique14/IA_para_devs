# Exemplo 3 - Memória de Conversação
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Inicializa o modelo e a memória
    llm = OpenAI(api_key=api_key)
    memory = ConversationBufferMemory()

    # Define a conversa inicial
    conversation = [
        {"role": "user", "content": "Qual é o meu nome?"},
        {"role": "assistant", "content": "Desculpe, não sei seu nome. Como você se chama?"}
    ]

    # Adiciona a conversa à memória
    memory.add_messages(conversation)

    # Faz uma nova requisição com contexto
    prompt = """
    Você agora tem o contexto da conversa que já ocorreu. Continue a conversa como
    se estivesse ciente da interação anterior.
    """
    response = llm.invoke(prompt + str(memory.buffer))

    print("Resposta do modelo:", response)
    print("Memória da conversa:", memory.buffer)

if __name__ == "__main__":
    main()