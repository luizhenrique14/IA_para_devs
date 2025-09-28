# Exemplo 5: Wikipedia Loader - Busca Simples
from langchain.document_loaders import WikipediaLoader

def pesquisar_wikipedia(query: str):
    """
    Faz uma busca simples na Wikipedia em português.
    
    Args:
        query (str): O termo que você quer pesquisar na Wikipedia
    """
    print(f"\nPesquisando na Wikipedia: {query}")
    
    # Configura o loader para buscar em português
    loader = WikipediaLoader(
        query=query,
        lang="pt"  # Busca em português
    )
    
    try:
        # Carrega o artigo
        documentos = loader.load()
        
        # Exibe o número de seções encontradas
        print(f"\nForam encontradas {len(documentos)} seções no artigo")
        
        # Exibe o conteúdo de cada seção
        for i, doc in enumerate(documentos):
            print(f"\n{'=' * 50}")
            print(f"Seção {i + 1}:")
            print(f"{'=' * 50}")
            print(doc.page_content)
            
    except Exception as e:
        print(f"Erro ao buscar '{query}': {str(e)}")

def main():
    # Pesquisa sobre o Santos FC
    pesquisar_wikipedia("Santos Futebol Clube")

if __name__ == "__main__":
    main()