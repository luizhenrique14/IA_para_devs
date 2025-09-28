# Exemplo 1: PDF Loader
from langchain.document_loaders import PyPDFLoader
import os

def carregar_pdf(caminho_pdf):
    """
    Carrega e processa um arquivo PDF.
    
    Args:
        caminho_pdf (str): Caminho para o arquivo PDF
    
    Returns:
        list: Lista de documentos/páginas do PDF
    """
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_pdf}")
        
    print(f"\nCarregando PDF: {caminho_pdf}")
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    
    return documentos

def main():
    # Caminho para o arquivo PDF de exemplo
    caminho_pdf = "dados/pythonlearn.pdf"
    
    try:
        # Carrega o PDF
        documentos = carregar_pdf(caminho_pdf)
        
        # Exibe informações e conteúdo
        print(f"\nNúmero de páginas carregadas: {len(documentos)}")
        for i, documento in enumerate(documentos):
            print(f"\nConteúdo da página {i + 1}:")
            print("-" * 50)
            print(documento.page_content)
            print("-" * 50)
            
    except Exception as e:
        print(f"Erro ao processar o PDF: {str(e)}")

if __name__ == "__main__":
    main()