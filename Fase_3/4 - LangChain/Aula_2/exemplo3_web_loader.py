# Exemplo 3: Web Base Loader
from langchain.document_loaders import WebBaseLoader

def carregar_pagina_web(url):
    """
    Carrega e processa o conteúdo de uma página web.
    
    Args:
        url (str): URL da página web a ser carregada
    
    Returns:
        list: Lista de documentos/seções da página
    """
    print(f"\nCarregando URL: {url}")
    loader = WebBaseLoader(url)
    documentos = loader.load()
    
    return documentos

def main():
    # URL de exemplo
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    try:
        # Carrega a página web
        documentos = carregar_pagina_web(url)
        
        # Exibe informações e conteúdo
        print(f"\nNúmero de seções carregadas: {len(documentos)}")
        for i, documento in enumerate(documentos):
            print(f"\nConteúdo da seção {i + 1}:")
            print("-" * 50)
            print(documento.page_content)
            print("-" * 50)
            
    except Exception as e:
        print(f"Erro ao processar a página web: {str(e)}")

if __name__ == "__main__":
    main()