# Exemplo 2: CSV Loader
from langchain.document_loaders import CSVLoader
import os

def carregar_csv(caminho_csv):
    """
    Carrega e processa um arquivo CSV.
    
    Args:
        caminho_csv (str): Caminho para o arquivo CSV
    
    Returns:
        list: Lista de documentos/linhas do CSV
    """
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")
        
    print(f"\nCarregando CSV: {caminho_csv}")
    loader = CSVLoader(caminho_csv)
    documentos = loader.load()
    
    return documentos

def main():
    # Caminho para o arquivo CSV de exemplo
    caminho_csv = "dados/exemplo.csv"
    
    try:
        # Carrega o CSV
        documentos = carregar_csv(caminho_csv)
        
        # Exibe informações e conteúdo
        print(f"\nNúmero de linhas carregadas: {len(documentos)}")
        for i, documento in enumerate(documentos):
            print(f"\nConteúdo da linha {i + 1}:")
            print("-" * 50)
            print(documento.page_content)
            print("-" * 50)
            
    except Exception as e:
        print(f"Erro ao processar o CSV: {str(e)}")

if __name__ == "__main__":
    main()