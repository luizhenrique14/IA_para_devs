# Exemplos de Prompts com LangChain

Este diretório contém exemplos de como usar a biblioteca LangChain para criar e executar prompts com modelos de linguagem da OpenAI. Cada arquivo `exemplo_*.py` demonstra um caso de uso específico.

## Estrutura do Projeto

- `exemplo_1_descricao_produto.py`: Gera uma descrição de produto.
- `exemplo_2_extracao_avaliacao.py`: Extrai informações de uma avaliação de produto.
- `exemplo_3_sumarizacao.py`: Resume um texto longo.
- `exemplo_4_introducao_ia.py`: Gera uma introdução sobre Inteligência Artificial.
- `exemplo_5_gerar_perguntas.py`: Cria perguntas a partir de um texto.
- `exemplo_6_extrair_datas_locais.py`: Extrai datas e locais de um texto.
- `requirements.txt`: Lista as dependências Python necessárias.
- `.env.example`: Arquivo de exemplo para configurar sua chave de API.

## Configuração

1.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure sua chave de API da OpenAI:**
    - Renomeie o arquivo `.env.example` para `.env`.
    - Abra o arquivo `.env` e substitua `sua-chave-aqui` pela sua chave de API da OpenAI.

    ```
    OPENAI_API_KEY="sua-chave-aqui"
    ```

## Como Executar os Exemplos

Cada script pode ser executado individualmente. Por exemplo, para executar o exemplo de descrição de produto:

```bash
python exemplo_1_descricao_produto.py
```

Isso executará o script, que chamará a API da OpenAI e imprimirá a resposta no terminal.
