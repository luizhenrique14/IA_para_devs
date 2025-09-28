# Exemplo 6: SQL Database Agent
import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv

# Nome do arquivo do banco de dados
DB_PATH = "artistas.db"

def setup_database():
    """
    Cria e popula um banco de dados SQLite se ele não existir.
    """
    if os.path.exists(DB_PATH):
        print(f"O banco de dados '{DB_PATH}' já existe. Pulando a criação.")
        return

    print(f"Criando banco de dados de exemplo em '{DB_PATH}'...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Criar tabelas
    cursor.execute("""
    CREATE TABLE artistas (
        id INTEGER PRIMARY KEY,
        nome TEXT NOT NULL,
        genero TEXT NOT NULL
    );
    """)
    cursor.execute("""
    CREATE TABLE albuns (
        id INTEGER PRIMARY KEY,
        titulo TEXT NOT NULL,
        ano_lancamento INTEGER,
        artista_id INTEGER,
        FOREIGN KEY (artista_id) REFERENCES artistas (id)
    );
    """)

    # Inserir dados
    cursor.execute("INSERT INTO artistas (nome, genero) VALUES ('Queen', 'Rock');")
    cursor.execute("INSERT INTO artistas (nome, genero) VALUES ('Michael Jackson', 'Pop');")
    cursor.execute("INSERT INTO artistas (nome, genero) VALUES ('Legião Urbana', 'Rock');")

    cursor.execute("INSERT INTO albuns (titulo, ano_lancamento, artista_id) VALUES ('A Night at the Opera', 1975, 1);")
    cursor.execute("INSERT INTO albuns (titulo, ano_lancamento, artista_id) VALUES ('News of the World', 1977, 1);")
    cursor.execute("INSERT INTO albuns (titulo, ano_lancamento, artista_id) VALUES ('Thriller', 1982, 2);")
    cursor.execute("INSERT INTO albuns (titulo, ano_lancamento, artista_id) VALUES ('Bad', 1987, 2);")
    cursor.execute("INSERT INTO albuns (titulo, ano_lancamento, artista_id) VALUES ('Dois', 1986, 3);")

    conn.commit()
    conn.close()
    print("Banco de dados criado e populado com sucesso!")

def main():
    """
    Função principal que executa o agente SQL.
    """
    load_dotenv()
    
    # Garante que o banco de dados exista
    setup_database()

    # Conecta ao banco de dados usando LangChain
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    
    # Inicializa o modelo de linguagem
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Cria o agente SQL
    # Este agente pode inspecionar o schema do banco, escrever e executar queries SQL.
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True  # Mostra os "pensamentos" do agente
    )

    # Perguntas em linguagem natural para fazer ao banco de dados
    perguntas = [
        "Quantos artistas estão cadastrados?",
        "Liste todos os álbuns da banda Queen.",
        "Qual o ano de lançamento do álbum 'Thriller'?",
        "Quais álbuns foram lançados antes de 1980?",
        "Liste os artistas e a quantidade de álbuns de cada um."
    ]

    for pergunta in perguntas:
        try:
            print(f"\n{'='*50}")
            print(f"Pergunta: {pergunta}")
            print(f"{'-'*50}")
            
            # Invoca o agente com a pergunta
            resultado = agent_executor.invoke({"input": pergunta})
            
            print(f"\nResposta: {resultado['output']}")
            
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()