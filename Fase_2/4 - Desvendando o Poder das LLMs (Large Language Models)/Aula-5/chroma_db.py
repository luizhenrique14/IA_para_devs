import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Configuração dos logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "receitas")

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']


def main():
    generate_data_store()



def generate_data_store():
    logger.info("Iniciando geração do banco de dados vetorial")
    documents = load_documentos()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    logger.info("Banco de dados vetorial gerado com sucesso")


def load_documentos():
    logger.info(f"Carregando documentos do diretório: {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, loader_cls=UnstructuredWordDocumentLoader)
    docs = loader.load()
    logger.info(f"Carregados {len(docs)} documentos")
    return docs



def split_text(documents: list[Document]):
    logger.info("Iniciando divisão dos documentos em chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Divididos {len(documents)} documentos em {len(chunks)} chunks")
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()