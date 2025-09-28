# README.md - Exemplos LangChain
Este diretório contém exemplos práticos de uso da biblioteca LangChain com OpenAI.

## Configuração

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Configure o arquivo .env:
   - Copie o arquivo `env.example` para `.env`
   - Adicione sua chave API da OpenAI no arquivo `.env`

## Exemplos Disponíveis

1. **exemplo1_requisicao_simples.py**
   - Demonstra uma requisição básica à OpenAI
   - Para executar: `python exemplo1_requisicao_simples.py`

2. **exemplo2_output_parser.py**
   - Mostra como usar Output Parsers para estruturar respostas em JSON
   - Para executar: `python exemplo2_output_parser.py`

3. **exemplo3_memoria.py**
   - Implementa memória de conversação
   - Para executar: `python exemplo3_memoria.py`

4. **exemplo4_chains.py**
   - Demonstra encadeamento de etapas de processamento
   - Para executar: `python exemplo4_chains.py`

5. **exemplo5_gpt4.py**
   - Exemplo de uso do GPT-4 com configurações específicas
   - Para executar: `python exemplo5_gpt4.py`

## Estrutura do Projeto
```
.
├── requirements.txt      # Dependências do projeto
├── env.example          # Exemplo de arquivo de configuração
├── .env                 # Suas configurações (criar baseado no env.example)
└── exemplos/*.py        # Arquivos de exemplo
```