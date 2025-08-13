# Assistente de Receitas Saudáveis

Este é um assistente de receitas saudáveis que utiliza IA para fornecer recomendações de receitas com base em documentos de referência. O sistema usa processamento de linguagem natural e busca semântica para encontrar e recomendar receitas relevantes.

## Pré-requisitos

- Python 3.10 ou superior
- Uma chave de API da OpenAI
- Ambiente virtual Python (recomendado)

## Configuração do Ambiente

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
   - Crie um arquivo `.env` na raiz do projeto
   - Adicione sua chave da API OpenAI:
   ```
   OPENAI_API_KEY=sua_chave_aqui
   ```

## Estrutura do Projeto

- `chroma_db.py`: Script para processar documentos e criar o banco de dados vetorial
- `Assistente de Receitas Saudáveis.py`: Interface Streamlit do assistente
- `receitas/`: Diretório contendo os documentos Word com as receitas
- `requirements.txt`: Lista de dependências do projeto
- `.env`: Arquivo de configuração (não versionado)
- `chroma/`: Diretório do banco de dados vetorial (gerado automaticamente)

## Executando o Projeto

1. Primeiro, processe os documentos e crie o banco de dados vetorial:
```bash
python chroma_db.py
```

2. Em seguida, inicie a interface do Streamlit:
```bash
streamlit run "Assistente de Receitas Saudáveis.py"
```

O aplicativo estará disponível em `http://localhost:8501`

## Uso

1. Após iniciar o aplicativo, você verá uma interface de chat
2. Digite sua pergunta ou pedido de receita na caixa de texto
3. O assistente irá:
   - Buscar receitas relevantes no banco de dados
   - Gerar uma resposta personalizada usando o GPT-3.5
   - Apresentar a receita ou recomendação solicitada

## Logs

O sistema mantém dois arquivos de log para monitoramento:
- `chroma_db.log`: Registra o processamento dos documentos e criação do banco de dados
- `assistente_receitas.log`: Registra as interações e respostas do assistente

## Dependências Principais

```
streamlit==1.48.1
streamlit-chat==0.1.1
python-dotenv==1.1.1
openai==1.99.9
langchain==0.3.27
langchain-community==0.3.27
langchain-openai==0.3.30
langchain-chroma==0.2.5
chromadb==1.0.16
unstructured==0.18.11
python-docx==1.2.0
```

## Observações

- As receitas devem estar em formato .docx no diretório `receitas/`
- O banco de dados vetorial é recriado cada vez que `chroma_db.py` é executado
- A qualidade das respostas depende da qualidade e quantidade dos documentos fornecidos

## Solução de Problemas

1. Se encontrar erro de módulos não encontrados:
   - Verifique se está no ambiente virtual correto
   - Reinstale as dependências: `pip install -r requirements.txt`

2. Se o OpenAI API retornar erros:
   - Verifique se sua chave API está correta no arquivo `.env`
   - Confirme se sua conta tem créditos disponíveis

3. Se os documentos não forem carregados:
   - Verifique se estão no formato .docx
   - Confirme se estão no diretório `receitas/`
