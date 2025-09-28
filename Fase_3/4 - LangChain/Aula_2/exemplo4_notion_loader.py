# Exemplo 4: Notion Directory Loader
from langchain_community.document_loaders import NotionDirectoryLoader
from dotenv import load_dotenv
import os

def carregar_notion(notion_diretorio):
    """
    Carrega e processa documentos de um diretório do Notion.
    
    Args:
        notion_diretorio (str): ID da página ou diretório do Notion
    
    Returns:
        list: Lista de documentos do Notion
    """
    # Carrega as variáveis de ambiente
    load_dotenv()
    
    if not os.getenv("NOTION_TOKEN"):
        raise ValueError("Token do Notion não encontrado nas variáveis de ambiente")
    
    print(f"\nCarregando diretório do Notion: {notion_diretorio}")
    loader = NotionDirectoryLoader(notion_diretorio)
    documentos = loader.load()
    
    return documentos

def main():
    # ID do diretório do Notion
    notion_diretorio = "seu_id_do_notion"
    
    try:
        # Carrega os documentos do Notion
        documentos = carregar_notion(notion_diretorio)
        
        # Exibe informações e conteúdo
        print(f"\nNúmero de documentos carregados: {len(documentos)}")
        for i, documento in enumerate(documentos):
            print(f"\nConteúdo do documento {i + 1}:")
            print("-" * 50)
            print(documento.page_content)
            print("-" * 50)
            
    except Exception as e:
        print(f"Erro ao processar documentos do Notion: {str(e)}")

if __name__ == "__main__":
    main()