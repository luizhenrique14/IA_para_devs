# Exemplos de Loaders do LangChain

Este repositório contém exemplos práticos de diferentes loaders disponíveis no LangChain para processar e carregar dados de várias fontes.

## Estrutura do Projeto

```
.
├── dados/                  # Diretório para arquivos de exemplo
│   ├── exemplo1.pdf       # PDF de exemplo para PyPDFLoader
│   └── exemplo.csv        # CSV de exemplo para CSVLoader
├── exemplo1_pdf_loader.py  # Exemplo de uso do PyPDFLoader
├── exemplo2_csv_loader.py  # Exemplo de uso do CSVLoader
├── exemplo3_web_loader.py  # Exemplo de uso do WebBaseLoader
├── exemplo4_notion_loader.py # Exemplo de uso do NotionDirectoryLoader
├── requirements.txt        # Dependências do projeto
└── .env.example           # Template para variáveis de ambiente
```

## Pré-requisitos

1. Python 3.8 ou superior
2. pip (gerenciador de pacotes Python)

## Configuração

1. Clone o repositório
2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure as variáveis de ambiente:
   ```bash
   cp .env.example .env
   ```
   Edite o arquivo `.env` com suas credenciais.

## Exemplos Disponíveis

### 1. PDF Loader (exemplo1_pdf_loader.py)
Demonstra como carregar e processar arquivos PDF usando o PyPDFLoader.

```python
python exemplo1_pdf_loader.py
```

### 2. CSV Loader (exemplo2_csv_loader.py)
Mostra como carregar e processar arquivos CSV usando o CSVLoader.

```python
python exemplo2_csv_loader.py
```

### 3. Web Loader (exemplo3_web_loader.py)
Exemplo de como fazer scraping de páginas web usando o WebBaseLoader.

```python
python exemplo3_web_loader.py
```

### 4. Notion Loader (exemplo4_notion_loader.py)
Demonstra como carregar documentos do Notion usando o NotionDirectoryLoader.

```python
python exemplo4_notion_loader.py
```

## Notas Importantes

- Certifique-se de ter os arquivos de exemplo necessários na pasta `dados/`
- Para o Notion Loader, você precisa configurar um token de API no arquivo `.env`
- Alguns loaders podem requerer dependências adicionais específicas

## Tratamento de Erros

Todos os exemplos incluem tratamento de erros básico e mensagens informativas para ajudar na depuração.

## Contribuindo

Sinta-se à vontade para contribuir com melhorias nos exemplos ou adicionar novos loaders!