# Exemplo 7: YouTube Loader
import os
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv

def carregar_transcricao_youtube(url: str):
    """
    Carrega a transcrição de um vídeo do YouTube.

    Args:
        url (str): A URL completa do vídeo do YouTube.
    """
    print(f"\nCarregando transcrição do vídeo: {url}")

    try:
        # Tenta carregar a transcrição em português, com fallback para inglês
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # Carrega metadados como título e autor
            language=["pt", "en"],
            translation="pt",  # Se a transcrição original não for 'pt', traduz para 'pt'
        )

        # Carrega o documento (transcrição)
        documentos = loader.load()

        # Verifica se algum documento foi carregado
        if not documentos:
            print("Não foi possível carregar a transcrição para este vídeo.")
            return

        # Exibe o conteúdo e os metadados
        for doc in documentos:
            print("\n" + "="*50)
            print("Metadados do Vídeo:")
            print(f"- Título: {doc.metadata.get('title', 'N/A')}")
            print(f"- Autor: {doc.metadata.get('author', 'N/A')}")
            print(f"- Duração: {doc.metadata.get('length', 'N/A')} segundos")
            
            print("\nTranscrição:")
            print(doc.page_content)
            print("="*50)

    except Exception as e:
        print(f"Ocorreu um erro ao tentar carregar o vídeo: {e}")
        print("Verifique se a URL está correta e se o vídeo possui transcrições disponíveis.")

def main():
    """
    Função principal que executa o loader do YouTube.
    """
    load_dotenv()

    # URL do vídeo do YouTube a ser processado
    video_url = "https://www.youtube.com/watch?v=3SJTjoU7TGw"
    
    carregar_transcricao_youtube(video_url)

if __name__ == "__main__":
    main()